# lectura y verificacion de los archivos csv de entrenamiento y validacion
import csv
from collections import Counter, defaultdict
from pathlib import Path

from entrenamiento.p01_config import (
    CLASES,
    RUTA_CORPUS_IMPLICITO,
    RUTA_TRAIN,
    RUTA_VAL_GO,
    RUTA_VAL_SINTETICA,
)


def cargar_csv_etiquetado(ruta):
    # lee un csv fila por fila verificando que tenga texto y una emocion valida
    # devuelve una lista de diccionarios listos para entrenar o evaluar
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {ruta_path}")

    filas = []
    # abre el csv en utf-8 para leer bien tildes y la letra enie
    with ruta_path.open("r", encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        columnas = set(lector.fieldnames or [])
        # revisa que existan las dos columnas obligatorias antes de leer
        if "texto" not in columnas or "etiqueta" not in columnas:
            raise ValueError(
                f"El archivo {ruta_path.name} debe tener columnas 'texto' y 'etiqueta'."
            )

        # recorre cada fila empezando en la linea 2 porque la 1 es el encabezado
        for numero, fila in enumerate(lector, start=2):
            texto = (fila.get("texto") or "").strip()
            etiqueta = (fila.get("etiqueta") or "").strip().lower()
            # salta filas vacias y detiene la carga si hay una etiqueta desconocida
            if not texto:
                continue
            if etiqueta not in CLASES:
                raise ValueError(
                    f"Etiqueta invalida '{etiqueta}' en {ruta_path.name} linea {numero}."
                )
            filas.append(
                {
                    "texto": texto,
                    "etiqueta": etiqueta,
                    "grupo": (fila.get("grupo") or "").strip(),
                    "origen": (fila.get("origen") or "desconocido").strip(),
                }
            )

    return filas


def resumir_particion(filas):
    # cuenta cuantas frases hay por emocion, por origen y cuantos grupos unicos existen
    por_clase = Counter(f["etiqueta"] for f in filas)
    por_origen = Counter(f["origen"] for f in filas)
    # arma una tabla cruzada para ver cuantas frases de cada clase aporta cada fuente
    cruce = defaultdict(Counter)
    for fila in filas:
        cruce[fila["origen"]][fila["etiqueta"]] += 1

    textos_unicos = {f["texto"].lower() for f in filas}
    grupos_unicos = {f["grupo"] for f in filas if f["grupo"]}

    return {
        "total": len(filas),
        "textos_unicos": len(textos_unicos),
        "grupos_unicos": len(grupos_unicos),
        "por_clase": {c: por_clase.get(c, 0) for c in CLASES},
        "por_origen": dict(sorted(por_origen.items())),
        "por_origen_y_clase": {
            origen: {c: conteos.get(c, 0) for c in CLASES}
            for origen, conteos in sorted(cruce.items())
        },
    }


def verificar_sin_fuga(train_filas, val_go_filas, val_sin_filas):
    # comprueba con intersecciones de conjuntos que no haya textos ni grupos repetidos
    # si todo da cero garantiza que validacion no se mezclo con entrenamiento
    t_textos = {f["texto"].lower() for f in train_filas}
    vg_textos = {f["texto"].lower() for f in val_go_filas}
    vs_textos = {f["texto"].lower() for f in val_sin_filas}

    # extrae tambien los identificadores de grupo para evitar parafrasis partidas
    t_grupos = {f["grupo"] for f in train_filas if f["grupo"]}
    vg_grupos = {f["grupo"] for f in val_go_filas if f["grupo"]}
    vs_grupos = {f["grupo"] for f in val_sin_filas if f["grupo"]}

    return {
        "solape_texto_train_val_go": len(t_textos & vg_textos),
        "solape_texto_train_val_sintetica": len(t_textos & vs_textos),
        "solape_texto_val_go_val_sintetica": len(vg_textos & vs_textos),
        "solape_grupo_train_val_go": len(t_grupos & vg_grupos),
        "solape_grupo_train_val_sintetica": len(t_grupos & vs_grupos),
    }


def cargar_entrenamiento_completo(
    ruta_train=RUTA_TRAIN,
    ruta_implicito=RUTA_CORPUS_IMPLICITO,
):
    # junta el csv base de entrenamiento con el corpus local de situaciones implicitas
    filas = cargar_csv_etiquetado(ruta_train)
    if ruta_implicito and Path(ruta_implicito).exists():
        filas.extend(cargar_csv_etiquetado(ruta_implicito))
    return filas


def cargar_particiones(
    ruta_train=RUTA_TRAIN,
    ruta_val_go=RUTA_VAL_GO,
    ruta_val_sintetica=RUTA_VAL_SINTETICA,
    ruta_implicito=RUTA_CORPUS_IMPLICITO,
):
    # carga entrenamiento y las dos validaciones y devuelve tambien el resumen auditable
    train_filas = cargar_entrenamiento_completo(ruta_train, ruta_implicito)
    val_go_filas = cargar_csv_etiquetado(ruta_val_go)
    val_sin_filas = cargar_csv_etiquetado(ruta_val_sintetica)

    resumen = {
        "train": resumir_particion(train_filas),
        "validacion_go": resumir_particion(val_go_filas),
        "validacion_sintetica": resumir_particion(val_sin_filas),
        "auditoria_fugas": verificar_sin_fuga(train_filas, val_go_filas, val_sin_filas),
        "nota_ansiedad": (
            "En GoEmotions la etiqueta original nervousness (nerviosismo) se usa "
            "como aproximacion a la categoria ansiedad y cuenta con pocos casos "
            "frente a los ejemplos sinteticos."
        ),
    }

    return train_filas, val_go_filas, val_sin_filas, resumen
