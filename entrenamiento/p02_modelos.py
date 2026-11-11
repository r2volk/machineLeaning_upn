"""02 · Cinco candidatos pequeños, declarados antes de observar resultados."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .p00_configuracion import CLASES, MARGEN, SEMILLA
from .p01_datos import normalizar

NOMBRES = ['logistica_tfidf', 'logistica_conteos', 'logistica_bigramas', 'naive_bayes', 'referencia']


def crear_modelo(nombre):
    """Un Pipeline aprende vocabulario/IDF solo durante fit(train). C=4 es fijo."""
    if nombre not in NOMBRES:
        raise ValueError('Modelo desconocido')
    opciones = dict(preprocessor=normalizar, lowercase=False, token_pattern=r'(?u)\b[a-z]{2,}\b')
    vectorizador = CountVectorizer(**opciones) if nombre == 'logistica_conteos' else TfidfVectorizer(
        **opciones, ngram_range=(1, 2) if nombre == 'logistica_bigramas' else (1, 1), sublinear_tf=True)
    if nombre == 'referencia':
        clasificador = DummyClassifier(strategy='most_frequent')
    elif nombre == 'naive_bayes':
        clasificador = MultinomialNB(alpha=1.0)
    else:
        clasificador = LogisticRegression(C=4.0, max_iter=2000, random_state=SEMILLA)
    return Pipeline([('vectorizador', vectorizador), ('clasificador', clasificador)])


def metricas(reales, predichas):
    """Accuracy, macro-F1 y detalle de todas las clases en un orden fijo."""
    return dict(accuracy=float(accuracy_score(reales, predichas)),
                macro_f1=float(f1_score(reales, predichas, labels=CLASES, average='macro', zero_division=0)),
                por_clase=classification_report(reales, predichas, labels=CLASES,
                                                output_dict=True, zero_division=0),
                matriz=confusion_matrix(reales, predichas, labels=CLASES).tolist())


def mascara_aceptacion(probabilidades, umbral, conocidos=None):
    """Abstención heurística por score máximo, margen y vocabulario conocido."""
    ordenadas = np.sort(probabilidades, axis=1)
    aceptar = (ordenadas[:, -1] >= umbral) & ((ordenadas[:, -1] - ordenadas[:, -2]) >= MARGEN)
    if conocidos is not None:
        aceptar &= np.asarray(conocidos)
    return aceptar


def selectivas(reales, predichas, aceptar):
    """Siempre informa cobertura junto a accuracy selectiva: abstenerse tiene costo."""
    reales, predichas = np.array(reales), np.array(predichas)
    return dict(aceptados=int(aceptar.sum()), total=len(reales), cobertura=float(np.mean(aceptar)),
                accuracy_aceptados=float(np.mean(reales[aceptar] == predichas[aceptar])) if aceptar.any() else None)
