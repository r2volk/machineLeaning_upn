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
                # lee palabras sueltas, parejas y trios ignorando las presentes en mas del 50 porciento
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    # exige aparecer al menos en 2 frases para evitar errores de tipeo
                    min_df=2,
                    max_df=max_df,
                    # usa escala logaritmica para que repetir una palabra no infle su peso
                    sublinear_tf=True,
                    token_pattern=PATRON_TOKEN_MODELO,
                ),
            ),
            (
                "caracteres",
                # parte cada palabra en trozos de 3 a 5 letras dentro de sus bordes
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
                # c en 3.0 controla cuanto se ajusta al entrenamiento y max_iter da margen para converger
                LogisticRegression(
                    C=c_reg,
                    class_weight="balanced",
                    max_iter=1000,
                    # fija la semilla en 42 para que el entrenamiento sea siempre repetible
                    random_state=SEMILLA,
                ),
            ),
        ]
    )


def construir_catalogo_modelos():
    # construye los 9 modelos comparados en validacion con la misma semilla fija
    # incluye linea base, variantes de n-gramas, svm, red neuronal con svd y ensamble
    return [
        # modelo 1: referencia mas simple con conteo basico de palabras y naive bayes
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
                        # solo cuenta cuantas veces aparece cada palabra suelta
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
        # modelo 2: pasa a tf idf y regresion logistica solo con palabras sueltas
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
        # modelo 3: agrega parejas de palabras seguidas para captar expresiones como no quiero
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
        # modelo 4: amplia hasta trios de palabras seguidas manteniendo los mismos parametros
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
        # modelo 5 ganador: combina palabras, subpalabras de letras y regresion logistica afinada
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
        # modelo 6: variante de naive bayes disenada para clases desbalanceadas
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
        # modelo 7: svm lineal que busca la frontera mas ancha pero sin porcentajes directos
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
        # modelo 8: reduccion de dimensionalidad a 150 columnas y red neuronal mlp
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
                    # comprime 84 mil columnas de texto a 150 ejes resumen para la red neuronal
                    ("reduccion", TruncatedSVD(n_components=150, random_state=SEMILLA)),
                    # escala cada fila comprimida antes de entrar a las neuronas
                    ("normalizador", Normalizer(copy=False)),
                    (
                        "clasificador",
                        # red con una capa oculta de 64 neuronas que frena sola si deja de mejorar
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
        # modelo 9: ensamble suave que promedia probabilidades de regresion logistica y naive bayes
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
                        # vota por probabilidades dando triple peso a regresion logistica
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
