# configuracion central de rutas, clases y parametros del modelo
from os import getenv
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
# separa los corpus compartidos de las correcciones privadas
CARPETA_ML = RAIZ_PROYECTO.parent
DATOS_COMPARTIDOS = Path(getenv("CHATBOT_DATA_DIR") or CARPETA_ML / "datos de entrenamiento")
DOCUMENTACION = Path(getenv("CHATBOT_DOCS_DIR") or CARPETA_ML / "documentacionCF")
DIR_DATOS = DATOS_COMPARTIDOS / "preparados"
DIR_LOCAL = RAIZ_PROYECTO / "datos"
DIR_MODELOS = RAIZ_PROYECTO / "modelos"

RUTA_TRAIN = DIR_DATOS / "train.csv"
RUTA_VAL_GO = DIR_DATOS / "validacion_go.csv"
RUTA_VAL_SINTETICA = DIR_DATOS / "validacion_sintetica.csv"
RUTA_CORRECCIONES = DIR_LOCAL / "correcciones_usuario.json"

RUTA_MODELO_BASE = DIR_MODELOS / "modelo_base.joblib"
RUTA_MODELO_ACTIVO = DIR_MODELOS / "modelo_activo.joblib"
RUTA_RESULTADOS = DIR_MODELOS / "resultados_experimentos.json"
RUTA_ESTADO_MODELO = DIR_MODELOS / "estado_modelo.json"

# orden fijo de etiquetas para matrices de confusion y reportes
CLASES = ["alegria", "ansiedad", "enojo", "neutral", "tristeza"]

NOMBRES_CLASES = {
    "alegria": "Alegría",
    "ansiedad": "Ansiedad / nerviosismo",
    "enojo": "Enojo",
    "neutral": "Neutral",
    "tristeza": "Tristeza",
}

# semilla fija para reproducibilidad de experimentos
SEMILLA = 42

# umbrales calibrados en validacion para pedir contexto cuando la senal es debil
UMBRAL_PROB_MINIMA = 0.45
UMBRAL_MARGEN_MINIMO = 0.10

# repeticiones moderadas por correccion para superar min_df sin desestabilizar el modelo
PESO_REPETICION_CORRECCION = 3
