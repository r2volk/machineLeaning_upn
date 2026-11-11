"""04 · Una única apertura del test con modelo y protocolo ya congelados."""
from datetime import datetime, timezone

import joblib
import numpy as np

from .p00_configuracion import DATOS, MODELO, RAIZ, SALIDAS, SEMILLA, guardar_json, huella, leer_json
from .p01_datos import cargar, verificar_datos, xy
from .p02_modelos import mascara_aceptacion, metricas, selectivas


def intervalo_por_grupos(filas, predichas, repeticiones=1000):
    """Bootstrap de predicciones guardadas, remuestreando escenarios, no frases sueltas."""
    rng = np.random.default_rng(SEMILLA)
    grupos = sorted({f['grupo'] for f in filas})
    indices = {g: [i for i, f in enumerate(filas) if f['grupo'] == g] for g in grupos}
    reales = np.array([f['etiqueta'] for f in filas])
    valores = []
    from sklearn.metrics import f1_score
    from .p00_configuracion import CLASES
    for _ in range(repeticiones):
        muestra = [i for g in rng.choice(grupos, size=len(grupos), replace=True) for i in indices[g]]
        valores.append(f1_score(reales[muestra], predichas[muestra], labels=CLASES, average='macro', zero_division=0))
    return dict(inferior=float(np.quantile(valores, .025)), superior=float(np.quantile(valores, .975)),
                repeticiones=repeticiones, metodo='Percentil 95%, bootstrap por escenario; exploratorio')


def evaluar_una_vez():
    """Reserva test con creación exclusiva ANTES de cargarlo. Una segunda llamada no predice."""
    salida = SALIDAS / 'test_final.json'
    if salida.exists():
        return leer_json(salida)
    seleccion = leer_json(SALIDAS / 'seleccion.json')
    verificar_datos()
    if huella(DATOS / 'manifiesto.json') != seleccion['manifiesto_sha256']:
        raise RuntimeError('El manifiesto fue alterado')
    for nombre, esperado in seleccion['codigo_sha256'].items():
        if huella(RAIZ / nombre) != esperado:
            raise RuntimeError(f'Código modificado después de selección: {nombre}')
    if huella(MODELO) != seleccion['modelo_sha256']:
        raise RuntimeError('El modelo no coincide con el seleccionado')
    # No se borra si ocurre un error: una ejecución interrumpida exige una auditoría.
    reserva = SALIDAS / 'test_reservado.json'
    with reserva.open('x', encoding='utf-8') as archivo:
        archivo.write('{"estado":"reservado", "politica":"no reabrir ni borrar"}')
    modelo = joblib.load(MODELO)['pipeline']
    filas = cargar('test')
    textos, reales = xy(filas)
    probabilidades = modelo.predict_proba(textos)  # ÚNICA inferencia sobre test.
    predichas = modelo.classes_[np.argmax(probabilidades, axis=1)]
    conocidos = modelo['vectorizador'].transform(textos).getnnz(axis=1) > 0
    aceptar = mascara_aceptacion(probabilidades, seleccion['umbral'], conocidos)
    resultado = dict(modelo=seleccion['ganador'], n=len(filas),
                     fecha_utc=datetime.now(timezone.utc).isoformat(),
                     metricas=metricas(reales, predichas),
                     selectivas=selectivas(reales, predichas, aceptar),
                     intervalo_macro_f1=intervalo_por_grupos(filas, predichas),
                     predicciones=[dict(**f, predicha=str(p), aceptada=bool(a), scores=s.tolist())
                                   for f, p, a, s in zip(filas, predichas, aceptar, probabilidades)],
                     modelo_sha256=huella(MODELO), seleccion_sha256=huella(SALIDAS / 'seleccion.json'))
    guardar_json(salida, resultado)
    guardar_json(reserva, dict(estado='completado', aperturas=1, test_final_sha256=huella(salida)))
    return resultado
