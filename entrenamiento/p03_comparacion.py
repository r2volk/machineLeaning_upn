"""03 · Comparar en validación, elegir umbral y congelar el modelo; NO abrir test."""
import platform
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning

from .p00_configuracion import DATOS, MODELO, RAIZ, SALIDAS, SEMILLA, UMBRALES, guardar_json, huella, leer_json
from .p01_datos import cargar, preparar, xy
from .p02_modelos import NOMBRES, crear_modelo, mascara_aceptacion, metricas, selectivas


def comparar():
    """Entrena cinco candidatos en train; desempata por el orden predeclarado."""
    if (SALIDAS / 'seleccion.json').exists():
        return leer_json(SALIDAS / 'seleccion.json')
    if (SALIDAS / 'test_reservado.json').exists():
        raise RuntimeError('No se permite seleccionar después de abrir test')
    preparar()
    train, validacion = cargar('train'), cargar('validacion')
    x_train, y_train = xy(train)
    x_val, y_val = xy(validacion)
    candidatos, modelos = [], {}
    for nombre in NOMBRES:
        modelo = crear_modelo(nombre)
        inicio = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            modelo.fit(x_train, y_train)
        segundos = time.perf_counter() - inicio
        predicciones = modelo.predict(x_val)
        candidatos.append(dict(nombre=nombre, segundos_ajuste=segundos,
                               train=metricas(y_train, modelo.predict(x_train)),
                               validacion=metricas(y_val, predicciones),
                               vocabulario=len(modelo['vectorizador'].vocabulary_)))
        modelos[nombre] = modelo
    ganador = max(candidatos, key=lambda c: c['validacion']['macro_f1'])['nombre']
    modelo = modelos[ganador]
    probabilidades = modelo.predict_proba(x_val)
    predichas = modelo.classes_[np.argmax(probabilidades, axis=1)]
    conocidos = modelo['vectorizador'].transform(x_val).getnnz(axis=1) > 0
    umbrales = [dict(umbral=u, **selectivas(y_val, predichas, mascara_aceptacion(probabilidades, u, conocidos)))
                for u in UMBRALES]
    # Regla previa: maximizar cobertura con accuracy >= .80 y cobertura >= .50.
    elegibles = [u for u in umbrales if u['cobertura'] >= .50 and u['accuracy_aceptados'] >= .80]
    if elegibles:
        umbral = max(elegibles, key=lambda u: (u['cobertura'], -u['umbral']))['umbral']
        motivo = 'Mayor cobertura entre umbrales con accuracy selectiva >= .80 y cobertura >= .50'
    else:
        umbral, motivo = .50, 'Ningún umbral cumplió el objetivo; fallback .50 predeclarado'
    curvas = curva_aprendizaje(ganador, train, x_val, y_val)
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(pipeline=modelo, umbral=umbral, nombre=ganador), MODELO)
    seleccion = dict(ganador=ganador, umbral=umbral, motivo_umbral=motivo, umbrales=umbrales,
                     candidatos=candidatos, curva=curvas, semilla=SEMILLA,
                     python=platform.python_version(), sklearn=sklearn.__version__,
                     creado_utc=datetime.now(timezone.utc).isoformat(),
                     modelo_sha256=huella(MODELO), manifiesto_sha256=huella(DATOS / 'manifiesto.json'),
                     codigo_sha256={p.name: huella(p) for p in sorted(RAIZ.glob('p0[0-5]*.py'))},
                     predicciones_validacion=[dict(**f, predicha=str(p)) for f, p in zip(validacion, predichas)])
    guardar_json(SALIDAS / 'seleccion.json', seleccion)
    return seleccion


def curva_aprendizaje(nombre, train, x_val, y_val):
    """Ablación de tamaño con subconjuntos anidados de grupos, usando solo validación."""
    rng = np.random.default_rng(SEMILLA)
    grupos = {}
    for clase in sorted({f['etiqueta'] for f in train}):
        lista = sorted({f['grupo'] for f in train if f['etiqueta'] == clase})
        grupos[clase] = rng.permutation(lista)
    resultados = []
    for fraccion in [.25, .50, .75, 1.0]:
        elegidos = {g for lista in grupos.values() for g in lista[:max(1, int(len(lista)*fraccion))]}
        filas = [f for f in train if f['grupo'] in elegidos]
        modelo = crear_modelo(nombre).fit(*xy(filas))
        resultados.append(dict(fraccion=fraccion, filas=len(filas), grupos=len(elegidos),
                               macro_f1=metricas(y_val, modelo.predict(x_val))['macro_f1']))
    return resultados
