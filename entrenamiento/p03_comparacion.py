# importa la consulta de la version de python
import platform

# importa el reloj para medir la duracion del ajuste
import time

# importa el control de avisos de las librerias
import warnings

# importa la fecha y la zona horaria del registro
from datetime import datetime, timezone

# importa la carga y el guardado de modelos locales
import joblib

# importa operaciones sobre arreglos y matrices
import numpy as np

# importa la consulta de la version de la libreria de modelos
import sklearn

# importa el aviso de falta de convergencia del ajuste
from sklearn.exceptions import ConvergenceWarning

# importa las rutas constantes y utilidades compartidas
from .p00_configuracion import (
    DATOS,
    MODELO,
    RAIZ,
    SALIDAS,
    SEMILLA,
    UMBRALES,
    guardar_json,
    huella,
    leer_json,
)

# importa la lectura limpieza y verificacion de datos
from .p01_datos import cargar, preparar, xy

# importa los modelos y las reglas de evaluacion
from .p02_modelos import NOMBRES, crear_modelo, mascara_aceptacion, metricas, selectivas


# elige un modelo y un umbral usando validacion
def comparar():
    # comprueba si ya hay una seleccion guardada
    if (SALIDAS / "seleccion.json").exists():
        # reutiliza la seleccion sin volver a entrenar
        return leer_json(SALIDAS / "seleccion.json")
    # comprueba si el test ya fue reservado
    if (SALIDAS / "test_reservado.json").exists():
        # impide elegir modelos despues de abrir el test
        raise RuntimeError("No se permite seleccionar después de abrir test")
    # prepara los datos o verifica los existentes
    preparar()
    # carga entrenamiento y validacion sin consultar test
    train, validacion = cargar("train"), cargar("validacion")
    # separa textos y etiquetas de entrenamiento
    x_train, y_train = xy(train)
    # separa textos y etiquetas de validacion
    x_val, y_val = xy(validacion)
    # prepara los resultados y los modelos candidatos
    candidatos, modelos = [], {}
    # recorre los cinco candidatos predefinidos
    for nombre in NOMBRES:
        # crea el candidato sin entrenarlo todavia
        modelo = crear_modelo(nombre)
        # registra el inicio para medir el tiempo de ajuste
        inicio = time.perf_counter()
        # limita el tratamiento de avisos a este bloque
        with warnings.catch_warnings():
            # convierte un aviso de falta de convergencia en un error
            warnings.simplefilter("error", ConvergenceWarning)
            # aprende vocabulario y pesos solo con entrenamiento
            modelo.fit(x_train, y_train)
        # calcula los segundos usados durante el ajuste
        segundos = time.perf_counter() - inicio
        # predice las etiquetas de validacion
        predicciones = modelo.predict(x_val)
        # registra las metricas del candidato y su vocabulario
        candidatos.append(
            dict(
                # registra el nombre interno del candidato
                nombre=nombre,
                # registra el tiempo usado al entrenar
                segundos_ajuste=segundos,
                # mide el resultado sobre los datos de entrenamiento
                train=metricas(y_train, modelo.predict(x_train)),
                # mide el resultado sobre validacion
                validacion=metricas(y_val, predicciones),
                # cuenta las caracteristicas que aprendio el vectorizador
                vocabulario=len(modelo["vectorizador"].vocabulary_),
            )
        )
        # conserva el modelo ya entrenado para no repetir el ajuste
        modelos[nombre] = modelo
    # elige el mayor promedio de las medidas de clase en validacion
    ganador = max(candidatos, key=lambda c: c["validacion"]["macro_f1"])["nombre"]
    # recupera el modelo ganador ya entrenado
    modelo = modelos[ganador]
    # obtiene las puntuaciones del ganador en validacion
    probabilidades = modelo.predict_proba(x_val)
    # traduce la mayor puntuacion de cada fila a su clase
    predichas = modelo.classes_[np.argmax(probabilidades, axis=1)]
    # detecta mensajes con alguna caracteristica conocida
    conocidos = modelo["vectorizador"].transform(x_val).getnnz(axis=1) > 0
    # mide cobertura y aciertos para cada umbral de abstencion
    umbrales = [
        dict(
            # conserva el limite elegido para aceptar etiquetas
            umbral=u,
            **selectivas(
                y_val, predichas, mascara_aceptacion(probabilidades, u, conocidos)
            )
        )
        for u in UMBRALES
    ]

    # conserva umbrales con cobertura y aciertos minimos suficientes
    elegibles = [
        u
        for u in umbrales
        if u["cobertura"] >= 0.50 and u["accuracy_aceptados"] >= 0.80
    ]
    # comprueba si algun umbral cumple la regla previa
    if elegibles:
        # elige la mayor cobertura y desempata con el menor umbral
        umbral = max(elegibles, key=lambda u: (u["cobertura"], -u["umbral"]))["umbral"]
        # registra el motivo de la eleccion
        motivo = "Mayor cobertura entre umbrales con accuracy selectiva >= .80 y cobertura >= .50"
    else:
        # usa el umbral de respaldo si ninguno cumple la regla
        umbral, motivo = (
            0.50,
            "Ningún umbral cumplió el objetivo; fallback .50 predeclarado",
        )
    # explora el efecto de usar distintas cantidades de entrenamiento
    curvas = curva_aprendizaje(ganador, train, x_val, y_val)
    # crea la carpeta donde se guardara el modelo
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    # guarda el ganador con su nombre y umbral
    joblib.dump(dict(pipeline=modelo, umbral=umbral, nombre=ganador), MODELO)
    # reune la seleccion las versiones las huellas y la evidencia
    seleccion = dict(
        # registra el candidato elegido con validacion
        ganador=ganador,
        # conserva el limite elegido para aceptar etiquetas
        umbral=umbral,
        # explica como se eligio el limite de aceptacion
        motivo_umbral=motivo,
        # conserva las alternativas de abstencion evaluadas
        umbrales=umbrales,
        # conserva los resultados de todos los candidatos
        candidatos=candidatos,
        # conserva los resultados al variar la cantidad de entrenamiento
        curva=curvas,
        # registra la semilla para repetir el procedimiento
        semilla=SEMILLA,
        # registra la version de python usada
        python=platform.python_version(),
        # registra la version de la libreria de modelos
        sklearn=sklearn.__version__,
        # registra la fecha del experimento con zona horaria
        creado_utc=datetime.now(timezone.utc).isoformat(),
        # guarda la huella del modelo entrenado
        modelo_sha256=huella(MODELO),
        # guarda la huella de la division usada
        manifiesto_sha256=huella(DATOS / "manifiesto.json"),
        # guarda las huellas de los modulos del experimento
        codigo_sha256={p.name: huella(p) for p in sorted(RAIZ.glob("p0[0-5]*.py"))},
        # conserva las etiquetas predichas sobre validacion
        predicciones_validacion=[
            dict(**f, predicha=str(p)) for f, p in zip(validacion, predichas)
        ],
    )
    # guarda los resultados para reutilizarlos despues
    guardar_json(SALIDAS / "seleccion.json", seleccion)
    # devuelve la seleccion realizada
    return seleccion


# explora el efecto de la cantidad de entrenamiento
def curva_aprendizaje(nombre, train, x_val, y_val):
    # crea un generador con semilla fija para los subconjuntos
    rng = np.random.default_rng(SEMILLA)
    # prepara los grupos ordenados por clase
    grupos = {}
    # recorre las clases presentes en entrenamiento
    for clase in sorted({f["etiqueta"] for f in train}):
        # reune los grupos de la clase actual
        lista = sorted({f["grupo"] for f in train if f["etiqueta"] == clase})
        # baraja sus grupos sin separar las dos frases
        grupos[clase] = rng.permutation(lista)
    # prepara los puntos de la curva
    resultados = []
    # prueba cuatro proporciones de los grupos de entrenamiento
    for fraccion in [0.25, 0.50, 0.75, 1.0]:
        # elige grupos completos y conserva al menos uno por clase
        elegidos = {
            g
            for lista in grupos.values()
            for g in lista[: max(1, int(len(lista) * fraccion))]
        }
        # selecciona las frases de esos grupos
        filas = [f for f in train if f["grupo"] in elegidos]
        # ajusta un modelo auxiliar con el subconjunto elegido
        modelo = crear_modelo(nombre).fit(*xy(filas))
        # registra el tamano usado y el resultado en validacion
        resultados.append(
            dict(
                # registra la proporcion de entrenamiento utilizada
                fraccion=fraccion,
                # registra la cantidad de frases de este subconjunto
                filas=len(filas),
                # registra la cantidad de escenarios completos usados
                grupos=len(elegidos),
                # promedia la medida de las clases con el mismo peso
                macro_f1=metricas(y_val, modelo.predict(x_val))["macro_f1"],
            )
        )
    # devuelve los puntos de la curva de aprendizaje
    return resultados
