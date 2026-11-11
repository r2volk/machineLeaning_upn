"""01 · Leer escenarios, auditar y separar grupos ANTES de vectorizar."""
import csv
import random
import re
import unicodedata
from collections import Counter

from .p00_configuracion import CLASES, DATOS, SALIDAS, SEMILLA, guardar_json, huella, leer_json


def normalizar(texto):
    """Minúsculas, tildes y signos; conserva negaciones y no usa diccionarios externos."""
    texto = unicodedata.normalize('NFKC', texto).casefold()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    return ' '.join(re.findall(r'[a-z]+', texto))


def leer_escenarios(ruta=DATOS / 'escenarios.txt'):
    """Convierte cada línea curada en dos filas con la misma etiqueta y grupo."""
    filas, etiqueta, contador = [], None, Counter()
    for linea in ruta.read_text(encoding='utf-8').splitlines():
        if not linea.strip() or linea.startswith('#'):
            continue
        if linea.startswith('['):
            etiqueta = linea.strip('[]')
            if etiqueta not in CLASES:
                raise ValueError('Etiqueta desconocida')
            continue
        partes = [p.strip() for p in linea.split('|')]
        if etiqueta is None or len(partes) != 2 or not all(partes):
            raise ValueError('Cada escenario debe tener dos paráfrasis y una clase')
        contador[etiqueta] += 1
        grupo = f'{etiqueta}_{contador[etiqueta]:03d}'
        for numero, texto in enumerate(partes, 1):
            filas.append(dict(id=f'{grupo}_{numero}', grupo=grupo, texto=texto,
                              etiqueta=etiqueta, origen='sintetico_asistido_IA'))
    textos = [normalizar(f['texto']) for f in filas]
    if len(textos) != len(set(textos)):
        raise ValueError('Hay textos normalizados duplicados; revisar antes de dividir')
    return filas


def preparar():
    """Congela una división estratificada por grupos 60/20/20, sin ajustar modelos."""
    manifiesto = DATOS / 'manifiesto.json'
    if manifiesto.exists():
        verificar_datos()
        return leer_json(manifiesto)
    if (SALIDAS / 'test_reservado.json').exists():
        raise RuntimeError('El test ya está sellado; no se puede cambiar la partición')
    filas = leer_escenarios()
    asignacion = {}
    rng = random.Random(SEMILLA)
    for clase in CLASES:
        grupos = sorted({f['grupo'] for f in filas if f['etiqueta'] == clase})
        rng.shuffle(grupos)
        n = len(grupos)
        for i, grupo in enumerate(grupos):
            asignacion[grupo] = 'train' if i < int(.6*n) else 'validacion' if i < int(.8*n) else 'test'
    for fila in filas:
        fila['particion'] = asignacion[fila['grupo']]
    for particion in ['train', 'validacion', 'test']:
        destino = DATOS / f'{particion}.csv'
        with destino.open('w', encoding='utf-8', newline='') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(f for f in filas if f['particion'] == particion)
    resumen = dict(semilla=SEMILLA, origen='100% sintético; autoría asistida por IA; sin participantes humanos',
                   n_filas=len(filas), n_grupos=len(asignacion), asignacion=asignacion,
                   conteos={p: dict(Counter(f['etiqueta'] for f in filas if f['particion'] == p))
                            for p in ['train', 'validacion', 'test']},
                   sha256={nombre: huella(DATOS / nombre) for nombre in
                           ['escenarios.txt', 'train.csv', 'validacion.csv', 'test.csv']})
    guardar_json(manifiesto, resumen)
    return resumen


def verificar_datos():
    """Comprueba bytes congelados; no lee etiquetas de test para modelar ni evaluar."""
    manifiesto = leer_json(DATOS / 'manifiesto.json')
    for nombre, esperado in manifiesto['sha256'].items():
        if huella(DATOS / nombre) != esperado:
            raise RuntimeError(f'Datos modificados después de congelar: {nombre}')


def cargar(particion):
    """Lee una partición explícita. Comparación e inferencia nunca solicitan test."""
    if particion not in {'train', 'validacion', 'test'}:
        raise ValueError('Partición no permitida')
    with (DATOS / f'{particion}.csv').open(encoding='utf-8', newline='') as archivo:
        return list(csv.DictReader(archivo))


def xy(filas):
    """Separa los únicos predictores (texto) de las etiquetas; excluye grupo e id."""
    return [f['texto'] for f in filas], [f['etiqueta'] for f in filas]
