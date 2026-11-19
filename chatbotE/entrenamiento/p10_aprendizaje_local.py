# administra ejemplos voluntarios y una version local del modelo
import json
import os
import threading
from pathlib import Path

import joblib

from seguridad import detectar_riesgo
from .p00_configuracion import CLASES, huella, leer_json
from .p01_datos import normalizar, xy
from .p07_datos_ampliados import DESTINO, leer_csv
from .p08_comparacion_ampliada import MODELO_AMPLIADO, RESULTADOS, medir, modelo_nuevo

LOCAL = Path(__file__).resolve().parent.parent / '.animo_local'
CANDADO = threading.RLock()


# crea un espacio privado en el equipo

def carpeta_local():
    LOCAL.mkdir(mode=0o700, parents=True, exist_ok=True)
    return LOCAL


# recupera los ejemplos guardados de forma explicita

def ejemplos():
    ruta = LOCAL / 'ejemplos.json'
    return leer_json(ruta) if ruta.exists() else []


# escribe datos privados sin dejar un archivo incompleto

def escribir_privado(ruta, datos):
    temporal = ruta.with_suffix(ruta.suffix + '.tmp')
    descriptor = os.open(temporal, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
    temporal.replace(ruta)


# carga el modelo base y solo usa una version local integra

def cargar_actual():
    seleccion = leer_json(RESULTADOS / 'seleccion.json')
    if huella(MODELO_AMPLIADO) != seleccion['modelo_sha256']:
        raise RuntimeError('El modelo ampliado no coincide con su huella.')
    base = joblib.load(MODELO_AMPLIADO)
    ruta = LOCAL / 'modelo_local.joblib'
    metadatos = LOCAL / 'modelo_local.json'
    if ruta.exists() and metadatos.exists():
        datos = leer_json(metadatos)
        if datos.get('sha256') != huella(ruta):
            raise RuntimeError('El modelo local no coincide con su huella.')
        return joblib.load(ruta)
    return base


# registra un mensaje corregido solo si el usuario lo solicita

def guardar_ejemplo(texto, etiqueta):
    if not isinstance(texto, str) or not isinstance(etiqueta, str):
        raise ValueError('Escribe un texto y elige una emoción.')
    texto = texto.strip()
    if not texto or len(texto) > 1500:
        raise ValueError('El texto debe tener entre 1 y 1500 caracteres.')
    if etiqueta not in CLASES:
        raise ValueError('Elige una de las cinco emociones.')
    if detectar_riesgo(texto):
        raise ValueError('Por privacidad y seguridad, este mensaje no se guarda como ejemplo.')
    clave = normalizar(texto)
    if not any(len(palabra) >= 2 for palabra in clave.split()):
        raise ValueError('Escribe al menos una palabra de dos letras.')
    with CANDADO:
        existentes = ejemplos()
        if len(existentes) >= 10000:
            raise ValueError('Se alcanzó el límite de ejemplos locales.')
        protegidos = {normalizar(f['texto']) for nombre in
                      ['train.csv', 'validacion_go.csv', 'validacion_sintetica.csv']
                      for f in leer_csv(nombre)}
        protegidos.update(normalizar(f['texto']) for f in existentes)
        if clave in protegidos:
            raise ValueError('Este texto ya existe en los datos o en tus ejemplos.')
        existentes.append({'texto': texto, 'etiqueta': etiqueta, 'origen': 'usuario_local'})
        escribir_privado(carpeta_local() / 'ejemplos.json', existentes)
        return len(existentes)


# entrena desde cero con los datos anteriores y las correcciones
# el test permanece intacto y no se utiliza para elegir cambios

def entrenar_local():
    with CANDADO:
        correcciones = ejemplos()
        if not correcciones:
            raise ValueError('Guarda al menos un ejemplo antes de entrenar.')
        seleccion = leer_json(RESULTADOS / 'seleccion.json')
        for nombre, esperado in seleccion['datos_sha256'].items():
            if huella(DESTINO / nombre) != esperado:
                raise RuntimeError('Los datos preparados fueron modificados.')
        balanceado = seleccion['ganador'] == 'combinado_balanceado'
        modelo = modelo_nuevo(balanceado).fit(*xy(leer_csv('train.csv') + correcciones))
        resultado = {
            'ejemplos_locales': len(correcciones),
            'validacion_go': medir(modelo, leer_csv('validacion_go.csv')),
            'validacion_sintetica': medir(modelo, leer_csv('validacion_sintetica.csv')),
        }
        artefacto = {'pipeline': modelo, 'umbral': seleccion['umbral'],
                     'nombre': 'logistica_bigramas_local'}
        carpeta = carpeta_local()
        temporal = carpeta / 'modelo_local.tmp'
        descriptor = os.open(temporal, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'wb') as archivo:
            joblib.dump(artefacto, archivo)
        destino = carpeta / 'modelo_local.joblib'
        temporal.replace(destino)
        resultado['sha256'] = huella(destino)
        escribir_privado(carpeta / 'modelo_local.json', resultado)
        return artefacto, resultado


# elimina correcciones y tambien el modelo que aprendio de ellas

def borrar_local():
    with CANDADO:
        for nombre in ['ejemplos.json', 'modelo_local.joblib', 'modelo_local.json']:
            (LOCAL / nombre).unlink(missing_ok=True)
        return cargar_actual()


# informa cantidades y medidas sin exponer las frases guardadas

def estado_local():
    with CANDADO:
        ruta = LOCAL / 'modelo_local.json'
        return {'ejemplos': len(ejemplos()), 'modelo_local': ruta.exists(),
                'evaluacion': leer_json(ruta) if ruta.exists() else None}
