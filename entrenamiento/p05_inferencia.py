"""05 · Usar el modelo congelado; nunca entrenar con conversaciones ni abrir test."""
import joblib
import numpy as np

from .p00_configuracion import DATOS, MODELO, SALIDAS, huella, leer_json
from .p01_datos import cargar, verificar_datos, xy
from .p02_modelos import crear_modelo, mascara_aceptacion


def cargar_modelo():
    """Carga el artefacto local verificado; si falta, lo reconstruye solo con train."""
    seleccion = leer_json(SALIDAS / 'seleccion.json')
    if not MODELO.exists():
        # Recuperación en memoria: no sobrescribe la evidencia del experimento original.
        verificar_datos()
        pipeline = crear_modelo(seleccion['ganador']).fit(*xy(cargar('train')))
        return dict(pipeline=pipeline, umbral=seleccion['umbral'], nombre=seleccion['ganador'])
    if huella(MODELO) != seleccion['modelo_sha256']:
        raise RuntimeError('Artefacto alterado. Restaurar modelo.joblib de la entrega original.')
    return joblib.load(MODELO)


def explicar(pipeline, vector, indice_clase, indice_alternativa):
    """Aportes x_j*(peso_clase-peso_alternativa); asociaciones, nunca causas psicológicas."""
    clasificador = pipeline['clasificador']
    if hasattr(clasificador, 'coef_'):
        pesos = clasificador.coef_
    elif hasattr(clasificador, 'feature_log_prob_'):
        pesos = clasificador.feature_log_prob_
    else:
        return []
    nombres = pipeline['vectorizador'].get_feature_names_out()
    diferencias = pesos[indice_clase] - pesos[indice_alternativa]
    aportes = [(nombres[j], float(valor*diferencias[j])) for j, valor in zip(vector.indices, vector.data)]
    return [dict(termino=t, aporte=v) for t, v in sorted(aportes, key=lambda par: par[1], reverse=True)[:5] if v > 0]


def predecir(artefacto, texto):
    """Predicción de texto independiente con scores, abstención y explicación local."""
    pipeline = artefacto['pipeline']
    vector = pipeline['vectorizador'].transform([texto])
    probabilidades = pipeline['clasificador'].predict_proba(vector)[0]
    orden = np.argsort(probabilidades)
    primero, segundo = int(orden[-1]), int(orden[-2])
    aceptada = bool(mascara_aceptacion(probabilidades.reshape(1, -1), artefacto['umbral'], [vector.nnz > 0])[0])
    clases = pipeline.classes_
    return dict(etiqueta=str(clases[primero]) if aceptada else 'incierto',
                candidata=str(clases[primero]), alternativa=str(clases[segundo]),
                aceptada=aceptada, score=float(probabilidades[primero]),
                margen=float(probabilidades[primero]-probabilidades[segundo]),
                scores={str(c): float(p) for c, p in zip(clases, probabilidades)},
                evidencia=explicar(pipeline, vector, primero, segundo),
                motivo='sin_vocabulario' if vector.nnz == 0 else 'umbral_o_margen' if not aceptada else 'senal_suficiente',
                modelo=artefacto['nombre'])
