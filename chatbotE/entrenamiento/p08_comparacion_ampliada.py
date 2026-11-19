# compara candidatos con validacion y congela el ganador
import json
import time

import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .p00_configuracion import CLASES, MODELO, SALIDAS, guardar_json, huella
from .p01_datos import xy
from .p02_modelos import crear_modelo
from .p07_datos_ampliados import DESTINO, leer_csv

RESULTADOS = SALIDAS / 'ampliacion'
MODELO_AMPLIADO = MODELO.parent / 'modelo_ampliado.joblib'


# calcula medidas por clase sin usar el test

def medir(modelo, filas):
    textos, reales = xy(filas)
    predichas = modelo.predict(textos)
    detalle = classification_report(reales, predichas, labels=CLASES,
                                    output_dict=True, zero_division=0)
    return {
        'n': len(filas),
        'macro_f1': float(f1_score(reales, predichas, labels=CLASES,
                                   average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(reales, predichas)),
        'f1_clases': {clase: detalle[clase]['f1-score'] for clase in CLASES},
    }


# crea el mismo algoritmo sencillo con pesos de clase opcionales

def modelo_nuevo(balanceado=False):
    modelo = crear_modelo('logistica_bigramas')
    if balanceado:
        modelo.set_params(clasificador__class_weight='balanced')
    return modelo


# compara conjuntos y selecciona con el macro f1 de validacion go

def comparar():
    if (RESULTADOS / 'test_reservado.lock').exists():
        raise RuntimeError('El test ya fue usado; conserva la seleccion congelada.')
    train = leer_csv('train.csv')
    go = leer_csv('validacion_go.csv')
    sintetica = leer_csv('validacion_sintetica.csv')
    original = joblib.load(MODELO)['pipeline']
    configuraciones = [
        ('original', None, False),
        ('sintetico', [f for f in train if f['origen'] != 'go_emotions_es_mt'], False),
        ('go', [f for f in train if f['origen'] != 'sintetico_claude'], False),
        ('combinado', train, False),
        ('combinado_balanceado', train, True),
    ]
    candidatos = []
    modelos = {}
    for nombre, filas, balanceado in configuraciones:
        inicio = time.monotonic()
        modelo = original if filas is None else modelo_nuevo(balanceado).fit(*xy(filas))
        resultado = {
            'nombre': nombre,
            'ejemplos_entrenamiento': 240 if filas is None else len(filas),
            'validacion_go': medir(modelo, go),
            'validacion_sintetica': medir(modelo, sintetica),
            'segundos': time.monotonic() - inicio,
        }
        modelos[nombre] = modelo
        candidatos.append(resultado)
        print(json.dumps(resultado, ensure_ascii=False), flush=True)
    ganador = max(candidatos, key=lambda c: c['validacion_go']['macro_f1'])
    # guarda la configuracion del ganador para el aprendizaje local
    if ganador['nombre'] not in {'combinado', 'combinado_balanceado'}:
        raise RuntimeError('El ganador requiere revisar el protocolo de aprendizaje local.')
    artefacto = {'pipeline': modelos[ganador['nombre']], 'umbral': 0.3,
                 'nombre': ganador['nombre']}
    MODELO_AMPLIADO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, MODELO_AMPLIADO)
    guardar_json(RESULTADOS / 'seleccion.json', {
        'ganador': ganador['nombre'], 'criterio': 'macro f1 en validacion go',
        'umbral': 0.3, 'umbral_nota': 'heredado del experimento anterior, no calibrado para esta fuente',
        'modelo_sha256': huella(MODELO_AMPLIADO), 'candidatos': candidatos,
        'datos_sha256': {nombre: huella(DESTINO / nombre)
                         for nombre in ['train.csv', 'validacion_go.csv', 'validacion_sintetica.csv']},
    })
    print('Ganador:', ganador['nombre'], flush=True)


if __name__ == '__main__':
    comparar()
