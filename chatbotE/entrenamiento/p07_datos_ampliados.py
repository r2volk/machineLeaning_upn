# prepara datos adicionales sin leer el test de la fuente
import csv
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from .p00_configuracion import DATOS_COMPARTIDOS, SALIDAS, SEMILLA, guardar_json
from .p01_datos import cargar, normalizar

FUENTES = DATOS_COMPARTIDOS / 'fuentes'
DESTINO = DATOS_COMPARTIDOS / 'preparados'
ETIQUETAS_GO = {2: 'enojo', 17: 'alegria', 19: 'ansiedad', 25: 'tristeza', 27: 'neutral'}
CAMPOS = ['texto', 'etiqueta', 'grupo', 'origen']


# carga solamente las etiquetas simples que se pueden relacionar
# con las cinco categorias del proyecto

def cargar_go(particion):
    import pyarrow.parquet as pq

    ruta = next((FUENTES / 'go_emotions_es').glob(particion + '-*.parquet'))
    filas = pq.read_table(ruta, columns=['id', 'labels', 'text_es']).to_pylist()
    salida = []
    for fila in filas:
        etiquetas = fila['labels']
        texto = fila['text_es'] or ''
        if len(etiquetas) != 1 or etiquetas[0] not in ETIQUETAS_GO:
            continue
        if not any(len(palabra) >= 2 for palabra in normalizar(texto).split()):
            continue
        salida.append(dict(texto=texto, etiqueta=ETIQUETAS_GO[etiquetas[0]],
                           grupo='go_' + fila['id'], origen='go_emotions_es_mt'))
    return salida


# escribe solo las columnas compartidas por las fuentes

def guardar_csv(nombre, filas):
    DESTINO.mkdir(parents=True, exist_ok=True)
    with (DESTINO / nombre).open('w', encoding='utf-8', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS, extrasaction='ignore')
        escritor.writeheader()
        escritor.writerows(filas)


# lee una particion ya preparada

def leer_csv(nombre):
    with (DESTINO / nombre).open(encoding='utf-8', newline='') as archivo:
        return list(csv.DictReader(archivo))


# agrupa parafrasis y frases con el mismo inicio y etiqueta

def dividir_sintetico(filas):
    padres = {fila['grupo']: fila['grupo'] for fila in filas}

    def raiz(grupo):
        while grupo != padres[grupo]:
            padres[grupo] = padres[padres[grupo]]
            grupo = padres[grupo]
        return grupo

    inicios = {}
    for fila in filas:
        palabras = normalizar(fila['texto']).split()
        clave = (fila['etiqueta'], *palabras[:4])
        if len(palabras) >= 4 and clave in inicios:
            padres[raiz(fila['grupo'])] = raiz(inicios[clave])
        else:
            inicios[clave] = fila['grupo']
    grupos = [raiz(fila['grupo']) for fila in filas]
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    train, validacion = next(divisor.split(filas, [f['etiqueta'] for f in filas], grupos))
    return [filas[i] for i in train], [filas[i] for i in validacion]


# elimina duplicados de entrenamiento y textos con etiquetas contradictorias

def depurar(filas, reservadas):
    etiquetas = defaultdict(set)
    for fila in filas:
        etiquetas[normalizar(fila['texto'])].add(fila['etiqueta'])
    vistos = set(reservadas)
    limpias = []
    for fila in filas:
        clave = normalizar(fila['texto'])
        if not clave or clave in vistos or len(etiquetas[clave]) > 1:
            continue
        vistos.add(clave)
        limpias.append(fila)
    return limpias


# guarda entrenamiento y validaciones sin tocar el test

def preparar():
    if (SALIDAS / 'ampliacion' / 'test_reservado.lock').exists():
        raise RuntimeError('El test ya fue reservado; conserva los datos congelados.')
    with (FUENTES / 'emociones_animo_10000.csv').open(encoding='utf-8-sig', newline='') as archivo:
        sinteticas = list(csv.DictReader(archivo))
    sint_train, sint_val = dividir_sintetico(sinteticas)
    go_val = cargar_go('validation')
    reservadas = {normalizar(f['texto']) for f in sint_val + go_val}
    original = cargar('train')
    go_train = cargar_go('train')
    train = depurar(original + go_train + sint_train, reservadas)
    guardar_csv('train.csv', train)
    guardar_csv('validacion_go.csv', go_val)
    guardar_csv('validacion_sintetica.csv', sint_val)
    guardar_json(DESTINO / 'resumen.json', {
        'train': dict(Counter(f['origen'] for f in train)),
        'train_clases': dict(Counter(f['etiqueta'] for f in train)),
        'validacion_go_clases': dict(Counter(f['etiqueta'] for f in go_val)),
        'validacion_sintetica_clases': dict(Counter(f['etiqueta'] for f in sint_val)),
        'descartadas_train': len(original + go_train + sint_train) - len(train),
    })
    print('Datos preparados:', len(train), 'entrenamiento,', len(go_val),
          'validacion go y', len(sint_val), 'validacion sintetica')


if __name__ == '__main__':
    preparar()
