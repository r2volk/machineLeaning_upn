# configuracion central de rutas, clases y parametros del modelo
from os import getenv
from pathlib import Path

# carpeta principal del proyecto y carpeta padre donde viven datos y docs
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_ML = RAIZ_PROYECTO.parent
# permite cambiar la ruta de datos o docs con variables de entorno si hace falta
DATOS_COMPARTIDOS = Path(getenv("CHATBOT_DATA_DIR") or CARPETA_ML / "datos de entrenamiento")
DOCUMENTACION = Path(getenv("CHATBOT_DOCS_DIR") or CARPETA_ML / "documentacionCF")

# subcarpetas para datos preparados, correcciones locales y modelos guardados
DIR_DATOS = DATOS_COMPARTIDOS / "preparados"
DIR_LOCAL = RAIZ_PROYECTO / "datos"
DIR_MODELOS = RAIZ_PROYECTO / "modelos"

# rutas exactas de los archivos csv de entrenamiento, validacion y correcciones
RUTA_TRAIN = DIR_DATOS / "train.csv"
RUTA_CORPUS_IMPLICITO = RAIZ_PROYECTO / "entrenamiento" / "corpus_situaciones_implicitas.csv"
RUTA_VAL_GO = DIR_DATOS / "validacion_go.csv"
RUTA_VAL_SINTETICA = DIR_DATOS / "validacion_sintetica.csv"
RUTA_CORRECCIONES = DIR_LOCAL / "correcciones_usuario.json"

# rutas de los modelos congelados y los reportes json de metricas
RUTA_MODELO_BASE = DIR_MODELOS / "modelo_base.joblib"
RUTA_MODELO_ACTIVO = DIR_MODELOS / "modelo_activo.joblib"
RUTA_RESULTADOS = DIR_MODELOS / "resultados_experimentos.json"
RUTA_ESTADO_MODELO = DIR_MODELOS / "estado_modelo.json"

# orden fijo de etiquetas para que las matrices de confusion siempre coincidan
CLASES = ["alegria", "ansiedad", "enojo", "neutral", "tristeza"]

# nombres bonitos con tildes para mostrar las emociones en la pagina web
NOMBRES_CLASES = {
    "alegria": "Alegría",
    "ansiedad": "Ansiedad / nerviosismo",
    "enojo": "Enojo",
    "neutral": "Neutral",
    "tristeza": "Tristeza",
}

# semilla fija para que los experimentos den siempre el mismo resultado
SEMILLA = 42

# si la probabilidad maxima es menor a 0.45 o la ventaja menor a 0.10 pide contexto
UMBRAL_PROB_MINIMA = 0.45
UMBRAL_MARGEN_MINIMO = 0.10

# repite cada correccion del usuario 3 veces para que supere min_df sin distorsionar
PESO_REPETICION_CORRECCION = 3

# mejores valores de max_df y regularizacion c elegidos en validacion
MAX_DF_TFIDF = 0.5
C_REGRESION_LOGISTICA = 3.0
# lista de combinaciones probadas en validacion para justificar el ganador
REJILLA_AJUSTE_LR = [
    (1.0, 1.0),
    (0.5, 1.0),
    (0.5, 2.0),
    (0.5, 3.0),
    (0.5, 5.0),
]
