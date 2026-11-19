# definicion de los modelos candidatos y del ganador con parametros fijos
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from entrenamiento.p01_config import SEMILLA
from entrenamiento.p03_preprocesamiento import PATRON_TOKEN


def construir_modelo_ganador():
    # construye tf-idf con trigramas y regresion logistica multinomial balanceada
    return Pipeline(
        [
            (
                "vectorizador",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    min_df=2,
                    sublinear_tf=True,
                    token_pattern=PATRON_TOKEN,
                ),
            ),
            (
                "clasificador",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )


def construir_catalogo_modelos():
    # devuelve seis modelos comparables fijando hiperparametros entre n-gramas
    return [
        {
            "clave": "referencia_bow_nb",
            "nombre": "Referencia simple: Bolsa de palabras (1,1) + Naive Bayes",
            "familia": "Naive Bayes",
            "ngramas": "(1, 1)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Linea base de conteo simple de palabras sueltas con clasificador "
                "MultinomialNB (Semana 11 Sec. 5.3 y Semana 12 Sec. 2.3)."
            ),
            "pipeline": Pipeline(
                [
                    (
                        "vectorizador",
                        CountVectorizer(
                            ngram_range=(1, 1),
                            min_df=2,
                            token_pattern=PATRON_TOKEN,
                        ),
                    ),
                    ("clasificador", MultinomialNB(alpha=1.0)),
                ]
            ),
        },
        {
            "clave": "tfidf_lr_unigramas",
            "nombre": "TF-IDF (1,1) + Regresión Logística balanceada",
            "familia": "Regresión Logística",
            "ngramas": "(1, 1)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Mismos parametros que bigramas y trigramas pero solo con palabras "
                "sueltas para aislar el aporte de los grupos de palabras."
            ),
            "pipeline": Pipeline(
                [
                    (
                        "vectorizador",
                        TfidfVectorizer(
                            ngram_range=(1, 1),
                            min_df=2,
                            sublinear_tf=True,
                            token_pattern=PATRON_TOKEN,
                        ),
                    ),
                    (
                        "clasificador",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=SEMILLA,
                        ),
                    ),
                ]
            ),
        },
        {
            "clave": "tfidf_lr_bigramas",
            "nombre": "TF-IDF (1,2) + Regresión Logística balanceada",
            "familia": "Regresión Logística",
            "ngramas": "(1, 2)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Grupos de hasta dos palabras con parametros fijos para comparar "
                "directamente contra trigramas."
            ),
            "pipeline": Pipeline(
                [
                    (
                        "vectorizador",
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            min_df=2,
                            sublinear_tf=True,
                            token_pattern=PATRON_TOKEN,
                        ),
                    ),
                    (
                        "clasificador",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=SEMILLA,
                        ),
                    ),
                ]
            ),
        },
        {
            "clave": "tfidf_lr_trigramas",
            "nombre": "TF-IDF (1,3) + Regresión Logística balanceada (Ganador)",
            "familia": "Regresión Logística",
            "ngramas": "(1, 3)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Grupos de hasta tres palabras con exactamente los mismos parametros "
                "que bigramas y probabilidades nativas por softmax."
            ),
            "pipeline": construir_modelo_ganador(),
        },
        {
            "clave": "tfidf_cnb_trigramas",
            "nombre": "TF-IDF (1,3) + Complement Naive Bayes",
            "familia": "Naive Bayes",
            "ngramas": "(1, 3)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Variante probabilistica de Naive Bayes adaptada a clases "
                "desbalanceadas con el mismo vocabulario de trigramas."
            ),
            "pipeline": Pipeline(
                [
                    (
                        "vectorizador",
                        TfidfVectorizer(
                            ngram_range=(1, 3),
                            min_df=2,
                            sublinear_tf=True,
                            token_pattern=PATRON_TOKEN,
                        ),
                    ),
                    ("clasificador", ComplementNB(alpha=1.0)),
                ]
            ),
        },
        {
            "clave": "tfidf_svm_trigramas",
            "nombre": "TF-IDF (1,3) + SVM Lineal balanceado",
            "familia": "SVM",
            "ngramas": "(1, 3)",
            "probabilidades_nativas": False,
            "descripcion": (
                "Clasificador de margen maximo (Semana 09 Sec. 2.4 y Semana 12 Sec. 2.3); "
                "no entrega probabilidades directas para abstencion sin calibracion extra."
            ),
            "pipeline": Pipeline(
                [
                    (
                        "vectorizador",
                        TfidfVectorizer(
                            ngram_range=(1, 3),
                            min_df=2,
                            sublinear_tf=True,
                            token_pattern=PATRON_TOKEN,
                        ),
                    ),
                    (
                        "clasificador",
                        LinearSVC(
                            C=1.0,
                            class_weight="balanced",
                            random_state=SEMILLA,
                        ),
                    ),
                ]
            ),
        },
    ]
