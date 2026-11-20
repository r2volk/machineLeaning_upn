# definicion de los modelos candidatos y del ganador con parametros fijos
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from entrenamiento.p01_config import (
    C_REGRESION_LOGISTICA,
    MAX_DF_TFIDF,
    SEMILLA,
)
from entrenamiento.p03_preprocesamiento import PATRON_TOKEN_MODELO


def construir_vectorizador_hibrido(max_df=MAX_DF_TFIDF):
    # une dos formas de leer texto: grupos de 1 a 3 palabras y pedacitos de 3 a 5 letras
    # asi capta frases completas y tambien conecta raices como robo y robaron
    return FeatureUnion(
        [
            (
                "palabras",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=max_df,
                    sublinear_tf=True,
                    token_pattern=PATRON_TOKEN_MODELO,
                ),
            ),
            (
                "caracteres",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def construir_modelo_ganador(c_reg=C_REGRESION_LOGISTICA, max_df=MAX_DF_TFIDF):
    # arma el pipeline ganador: vectorizador hibrido seguido de regresion logistica
    # usa class_weight balanced para que la clase neutral no opaque a las demas
    return Pipeline(
        [
            ("vectorizador", construir_vectorizador_hibrido(max_df=max_df)),
            (
                "clasificador",
                LogisticRegression(
                    C=c_reg,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )


def construir_catalogo_modelos():
    # construye los 9 modelos comparados en validacion con la misma semilla fija
    # incluye linea base, variantes de n-gramas, svm, red neuronal con svd y ensamble
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
                            token_pattern=PATRON_TOKEN_MODELO,
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
                            token_pattern=PATRON_TOKEN_MODELO,
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
                            token_pattern=PATRON_TOKEN_MODELO,
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
            "nombre": "TF-IDF (1,3) + Regresión Logística balanceada",
            "familia": "Regresión Logística",
            "ngramas": "(1, 3)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Grupos de hasta tres palabras con exactamente los mismos parametros "
                "que bigramas y probabilidades nativas por softmax."
            ),
            "pipeline": Pipeline(
                [
                    (
                        "vectorizador",
                        TfidfVectorizer(
                            ngram_range=(1, 3),
                            min_df=2,
                            sublinear_tf=True,
                            token_pattern=PATRON_TOKEN_MODELO,
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
            "clave": "tfidf_hibrido_lr",
            "nombre": "TF-IDF palabras (1,3) + subpalabras (3,5) + Regresión Logística (Ganador)",
            "familia": "Regresión Logística",
            "ngramas": "(1,3) + char (3,5)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Combina trigramas de palabras con alcance de negacion y subpalabras "
                "de caracteres para conectar raices y conjugaciones en espanol."
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
                            token_pattern=PATRON_TOKEN_MODELO,
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
                            token_pattern=PATRON_TOKEN_MODELO,
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
        {
            "clave": "tfidf_svd_mlp",
            "nombre": "TF-IDF híbrido + SVD (150) + Red Neuronal MLP",
            "familia": "Red Neuronal",
            "ngramas": "(1,3) + char (3,5)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Reduccion de dimensionalidad con TruncatedSVD y Perceptron Multicapa "
                "(Semana 14 Sec. 3.5 y 6.3) evaluado en las mismas particiones."
            ),
            "pipeline": Pipeline(
                [
                    ("vectorizador", construir_vectorizador_hibrido()),
                    ("reduccion", TruncatedSVD(n_components=150, random_state=SEMILLA)),
                    ("normalizador", Normalizer(copy=False)),
                    (
                        "clasificador",
                        MLPClassifier(
                            hidden_layer_sizes=(64,),
                            max_iter=40,
                            early_stopping=True,
                            random_state=SEMILLA,
                        ),
                    ),
                ]
            ),
        },
        {
            "clave": "tfidf_ensamble_voting",
            "nombre": "TF-IDF híbrido + Ensamble suave (LR + ComplementNB)",
            "familia": "Ensamble",
            "ngramas": "(1,3) + char (3,5)",
            "probabilidades_nativas": True,
            "descripcion": (
                "Metodo ensamblado por votacion suave (Semana 12 Sec. 2.3) combinando "
                "Regresion Logistica balanceada y Complement Naive Bayes."
            ),
            "pipeline": Pipeline(
                [
                    ("vectorizador", construir_vectorizador_hibrido()),
                    (
                        "clasificador",
                        VotingClassifier(
                            estimators=[
                                (
                                    "lr",
                                    LogisticRegression(
                                        C=C_REGRESION_LOGISTICA,
                                        class_weight="balanced",
                                        max_iter=1000,
                                        random_state=SEMILLA,
                                    ),
                                ),
                                ("cnb", ComplementNB(alpha=0.5)),
                            ],
                            voting="soft",
                            weights=[3, 1],
                        ),
                    ),
                ]
            ),
        },
    ]
