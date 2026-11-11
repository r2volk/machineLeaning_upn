"""00 · Constantes y utilidades; leer antes del resto del experimento."""
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / 'datos'
SALIDAS = RAIZ / 'resultados'
MODELO = RAIZ / 'artefactos' / 'modelo.joblib'
CLASES = ['alegria', 'ansiedad', 'enojo', 'neutral', 'tristeza']
SEMILLA = 2026
MARGEN = 0.10
UMBRALES = [0.30, 0.40, 0.50, 0.60, 0.70]


def huella(ruta):
    """SHA-256 de un archivo: permite detectar cambios, no prueba autoría."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def leer_json(ruta):
    """Lee un archivo UTF-8 con datos JSON."""
    return json.loads(Path(ruta).read_text(encoding='utf-8'))


def guardar_json(ruta, contenido):
    """Escribe JSON legible mediante reemplazo atómico en el mismo disco."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + '.tmp')
    temporal.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding='utf-8')
    temporal.replace(ruta)
