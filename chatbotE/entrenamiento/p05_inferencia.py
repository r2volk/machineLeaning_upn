# importa la carga y el guardado de modelos locales
import joblib

# importa operaciones sobre arreglos y matrices
import numpy as np

# importa las rutas constantes y utilidades compartidas
from .p00_configuracion import DATOS, MODELO, SALIDAS, huella, leer_json

# importa la lectura limpieza y verificacion de datos
from .p01_datos import cargar, verificar_datos, xy

# importa los modelos y las reglas de evaluacion
from .p02_modelos import crear_modelo, mascara_aceptacion


# carga el modelo verificado o lo recupera con entrenamiento
def cargar_modelo():
    # lee el nombre y el umbral del modelo seleccionado
    seleccion = leer_json(SALIDAS / "seleccion.json")
    # comprueba si falta el modelo guardado
    if not MODELO.exists():

        # verifica los datos antes de reconstruir el modelo
        verificar_datos()
        # reconstruye en memoria usando solamente entrenamiento
        pipeline = crear_modelo(seleccion["ganador"]).fit(*xy(cargar("train")))
        # devuelve el modelo recuperado sin escribir otro archivo
        return dict(
            # conserva juntos el vectorizador y el clasificador
            pipeline=pipeline,
            umbral=seleccion["umbral"],
            nombre=seleccion["ganador"],
        )
    # compara la huella del modelo con la seleccion guardada
    if huella(MODELO) != seleccion["modelo_sha256"]:
        # rechaza el modelo si su contenido fue alterado
        raise RuntimeError(
            "Artefacto alterado. Restaurar modelo.joblib de la entrega original."
        )
    # carga el archivo local ya verificado
    return joblib.load(MODELO)


# busca palabras que favorecen a una clase frente a otra
def explicar(pipeline, vector, indice_clase, indice_alternativa):
    # extrae el clasificador del conjunto de pasos
    clasificador = pipeline["clasificador"]
    # comprueba si el modelo tiene coeficientes de logistica
    if hasattr(clasificador, "coef_"):
        # usa los pesos de cada clase
        pesos = clasificador.coef_
    # comprueba si dispone de pesos logaritmicos de bayes
    elif hasattr(clasificador, "feature_log_prob_"):
        # usa las asociaciones de palabras de bayes
        pesos = clasificador.feature_log_prob_
    else:
        # devuelve una lista vacia si el modelo no permite estos aportes
        return []
    # recupera los nombres de las caracteristicas aprendidas
    nombres = pipeline["vectorizador"].get_feature_names_out()
    # resta los pesos de la primera clase y la alternativa
    diferencias = pesos[indice_clase] - pesos[indice_alternativa]
    # multiplica cada valor presente por su diferencia de pesos
    aportes = [
        (nombres[j], float(valor * diferencias[j]))
        for j, valor in zip(vector.indices, vector.data)
    ]
    # devuelve hasta cinco aportes positivos de mayor a menor
    return [
        dict(termino=t, aporte=v)
        for t, v in sorted(aportes, key=lambda par: par[1], reverse=True)[:5]
        if v > 0
    ]


# clasifica un mensaje usando el modelo ya entrenado
def predecir(artefacto, texto):
    # recupera los pasos del modelo entrenado
    pipeline = artefacto["pipeline"]
    # convierte el mensaje usando el vocabulario ya aprendido
    vector = pipeline["vectorizador"].transform([texto])
    # obtiene las puntuaciones de las clases para ese vector
    probabilidades = pipeline["clasificador"].predict_proba(vector)[0]
    # ordena los indices de clase por su puntuacion
    orden = np.argsort(probabilidades)
    # recupera los indices de las dos mejores clases
    primero, segundo = int(orden[-1]), int(orden[-2])
    # comprueba puntuacion margen y presencia de vocabulario
    aceptada = bool(
        mascara_aceptacion(
            probabilidades.reshape(1, -1), artefacto["umbral"], [vector.nnz > 0]
        )[0]
    )
    # recupera el orden real de las etiquetas del modelo
    clases = pipeline.classes_
    # devuelve la etiqueta o incertidumbre con su evidencia
    return dict(
        # guarda la clase asignada o la salida visible
        etiqueta=str(clases[primero]) if aceptada else "incierto",
        # informa la clase con mayor puntuacion aunque se rechace
        candidata=str(clases[primero]),
        # informa la segunda clase segun su puntuacion
        alternativa=str(clases[segundo]),
        # indica si se cumplen las reglas para mostrar la etiqueta
        aceptada=aceptada,
        # informa la mayor puntuacion de clase
        score=float(probabilidades[primero]),
        # resta la segunda puntuacion de la primera
        margen=float(probabilidades[primero] - probabilidades[segundo]),
        # relaciona cada clase con su puntuacion
        scores={str(c): float(p) for c, p in zip(clases, probabilidades)},
        # calcula los aportes de palabras de esta prediccion
        evidencia=explicar(pipeline, vector, primero, segundo),
        # explica por que se acepta o se pide mas contexto
        motivo=(
            "sin_vocabulario"
            if vector.nnz == 0
            else "umbral_o_margen" if not aceptada else "senal_suficiente"
        ),
        # identifica el modelo que produjo el resultado
        modelo=artefacto["nombre"],
    )
