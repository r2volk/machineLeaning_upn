# evalua una sola vez el modelo congelado con el test de la fuente
import json
from collections import Counter, defaultdict

import joblib
import numpy as np

from .p00_configuracion import guardar_json, huella, leer_json
from .p01_datos import normalizar, xy
from .p02_modelos import mascara_aceptacion, metricas, selectivas
from .p07_datos_ampliados import DESTINO, cargar_go, leer_csv
from .p08_comparacion_ampliada import MODELO_AMPLIADO, RESULTADOS


# revisa la seleccion y reserva el unico uso del test

def evaluar():
    seleccion = leer_json(RESULTADOS / 'seleccion.json')
    if huella(MODELO_AMPLIADO) != seleccion['modelo_sha256']:
        raise RuntimeError('El modelo no coincide con la seleccion congelada.')
    for nombre, esperado in seleccion['datos_sha256'].items():
        if huella(DESTINO / nombre) != esperado:
            raise RuntimeError('Los datos no coinciden con la seleccion congelada.')
    marca = RESULTADOS / 'test_reservado.lock'
    try:
        with marca.open('x', encoding='utf-8') as archivo:
            archivo.write('Test reservado para una sola ejecucion. No borrar para repetirlo.\n')
    except FileExistsError as error:
        raise RuntimeError('El test ya fue reservado o ejecutado.') from error

    # elimina coincidencias exactas con train y validacion
    reservadas = {normalizar(f['texto']) for nombre in
                  ['train.csv', 'validacion_go.csv', 'validacion_sintetica.csv']
                  for f in leer_csv(nombre)}
    test_fuente = cargar_go('test')
    etiquetas = defaultdict(set)
    for fila in test_fuente:
        etiquetas[normalizar(fila['texto'])].add(fila['etiqueta'])
    vistos = set(reservadas)
    test = []
    for fila in test_fuente:
        clave = normalizar(fila['texto'])
        if clave in vistos or len(etiquetas[clave]) > 1:
            continue
        vistos.add(clave)
        test.append(fila)

    # obtiene las probabilidades una sola vez para todas las medidas
    artefacto = joblib.load(MODELO_AMPLIADO)
    pipeline = artefacto['pipeline']
    textos, reales = xy(test)
    vectores = pipeline['vectorizador'].transform(textos)
    probabilidades = pipeline['clasificador'].predict_proba(vectores)
    predichas = pipeline.classes_[np.argmax(probabilidades, axis=1)]
    aceptadas = mascara_aceptacion(probabilidades, artefacto['umbral'],
                                  np.asarray(vectores.getnnz(axis=1) > 0))
    resultado = {
        'modelo': seleccion['ganador'], 'modelo_sha256': seleccion['modelo_sha256'],
        'fuente': 'go_emotions_es_mt test etiquetas simples seleccionadas',
        'n': len(test), 'descartados': len(test_fuente) - len(test),
        'clases': dict(Counter(reales)), 'metricas': metricas(reales, predichas),
        'selectivas': selectivas(reales, predichas, aceptadas),
    }
    guardar_json(RESULTADOS / 'test_final.json', resultado)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    evaluar()
