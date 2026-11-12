# importa el calculo de huellas de archivos
import hashlib

# importa la conversion entre texto json y datos de python
import json

# importa el manejo de rutas de archivos y carpetas
from pathlib import Path

# ubica la carpeta de entrenamiento
RAIZ = Path(__file__).resolve().parent
# ubica los datos del experimento
DATOS = RAIZ / "datos"
# ubica los resultados guardados
SALIDAS = RAIZ / "resultados"
# ubica el modelo entrenado
MODELO = RAIZ / "artefactos" / "modelo.joblib"
# fija el orden de las cinco clases
CLASES = ["alegria", "ansiedad", "enojo", "neutral", "tristeza"]
# fija la semilla para repetir la division
SEMILLA = 2026
# exige una diferencia minima entre las dos mejores clases
MARGEN = 0.10
# define los umbrales que se probaran en validacion
UMBRALES = [0.30, 0.40, 0.50, 0.60, 0.70]


# calcula una huella para detectar cambios en un archivo
def huella(ruta):
    # lee los bytes y devuelve su huella digital
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


# lee datos guardados en formato json
def leer_json(ruta):
    # convierte el texto del archivo en datos de python
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


# guarda datos sin dejar el destino a medio escribir
def guardar_json(ruta, contenido):
    # convierte el destino en una ruta
    ruta = Path(ruta)
    # crea la carpeta del archivo si hace falta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # prepara un archivo temporal junto al destino
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    # escribe los datos con sangria y conserva las letras del texto
    temporal.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # reemplaza el destino con el archivo ya escrito
    temporal.replace(ruta)
