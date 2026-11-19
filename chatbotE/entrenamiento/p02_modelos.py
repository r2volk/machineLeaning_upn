# importa operaciones sobre arreglos y matrices
import numpy as np

# importa el clasificador de referencia constante
from sklearn.dummy import DummyClassifier

# importa los vectorizadores de conteos y pesos de palabras
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# importa el clasificador de regresion logistica
from sklearn.linear_model import LogisticRegression

# importa las medidas para comparar etiquetas
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# importa el clasificador multinomial de bayes
from sklearn.naive_bayes import MultinomialNB

# importa la cadena de transformacion y clasificacion
from sklearn.pipeline import Pipeline

# importa las rutas constantes y utilidades compartidas
from .p00_configuracion import CLASES, MARGEN, SEMILLA

# importa la lectura limpieza y verificacion de datos
from .p01_datos import normalizar

# fija los candidatos y el orden de desempate
NOMBRES = [
    "logistica_tfidf",
    "logistica_conteos",
    "logistica_bigramas",
    "naive_bayes",
    "referencia",
]


# construye un candidato sin ajustarlo todavia
def crear_modelo(nombre):
    # comprueba que se haya pedido un candidato conocido
    if nombre not in NOMBRES:
        # rechaza nombres de modelos no declarados
        raise ValueError("Modelo desconocido")
    # comparte la limpieza y admite palabras de dos o mas letras
    opciones = dict(
        # aplica la misma limpieza al entrenar y al predecir
        preprocessor=normalizar,
        lowercase=False,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    # elige conteos o pesos de frecuencia y el rango de palabras
    vectorizador = (
        CountVectorizer(**opciones)
        if nombre == "logistica_conteos"
        else TfidfVectorizer(
            **opciones,
            # incluye parejas de palabras solo en el candidato de bigramas
            ngram_range=(1, 2) if nombre == "logistica_bigramas" else (1, 1),
            # reduce el efecto de repetir una misma palabra
            sublinear_tf=True
        )
    )
    # detecta el candidato de referencia constante
    if nombre == "referencia":
        # crea una referencia que siempre predice la clase mas frecuente
        clasificador = DummyClassifier(strategy="most_frequent")
    # detecta el candidato de bayes
    elif nombre == "naive_bayes":
        # crea bayes multinomial con suavizado
        clasificador = MultinomialNB(alpha=1.0)
    else:
        # crea logistica con regularizacion y limite de iteraciones
        clasificador = LogisticRegression(C=4.0, max_iter=2000, random_state=SEMILLA)
    # encadena la transformacion del texto y el clasificador
    return Pipeline([("vectorizador", vectorizador), ("clasificador", clasificador)])


# compara etiquetas reales y predichas
def metricas(reales, predichas):
    # calcula aciertos medidas por clase y matriz en un orden fijo
    return dict(
        # calcula la proporcion total de aciertos
        accuracy=float(accuracy_score(reales, predichas)),
        # promedia la medida de las clases con el mismo peso
        macro_f1=float(
            f1_score(reales, predichas, labels=CLASES, average="macro", zero_division=0)
        ),
        # informa las medidas y la cantidad de ejemplos de cada clase
        por_clase=classification_report(
            reales, predichas, labels=CLASES, output_dict=True, zero_division=0
        ),
        # cuenta las combinaciones de clase real y predicha
        matriz=confusion_matrix(reales, predichas, labels=CLASES).tolist(),
    )


# decide si mostrar una etiqueta o pedir mas contexto
def mascara_aceptacion(probabilidades, umbral, conocidos=None):
    # ordena las puntuaciones de cada mensaje
    ordenadas = np.sort(probabilidades, axis=1)
    # exige una puntuacion suficiente y margen entre las dos mejores
    aceptar = (ordenadas[:, -1] >= umbral) & (
        (ordenadas[:, -1] - ordenadas[:, -2]) >= MARGEN
    )
    # comprueba si tambien se informo del vocabulario conocido
    if conocidos is not None:
        # descarta los mensajes sin caracteristicas conocidas
        aceptar &= np.asarray(conocidos)
    # devuelve que mensajes cumplen las condiciones
    return aceptar


# calcula cobertura y aciertos entre los casos aceptados
def selectivas(reales, predichas, aceptar):
    # convierte etiquetas a arreglos para aplicar la seleccion
    reales, predichas = np.array(reales), np.array(predichas)
    # informa cobertura y aciertos entre las etiquetas aceptadas
    return dict(
        # cuenta las etiquetas que se pueden mostrar
        aceptados=int(aceptar.sum()),
        # cuenta todos los mensajes evaluados
        total=len(reales),
        # calcula la fraccion de mensajes aceptados
        cobertura=float(np.mean(aceptar)),
        # calcula aciertos entre aceptados o deja el valor sin definir
        accuracy_aceptados=(
            float(np.mean(reales[aceptar] == predichas[aceptar]))
            if aceptar.any()
            else None
        ),
    )
