# gestion local de correcciones del usuario, reentrenamiento y restauracion
import json
import shutil
import uuid
from datetime import datetime, timezone

import joblib

from entrenamiento.p01_config import (
    CLASES,
    PESO_REPETICION_CORRECCION,
    RUTA_CORRECCIONES,
    RUTA_ESTADO_MODELO,
    RUTA_MODELO_ACTIVO,
    RUTA_MODELO_BASE,
    RUTA_RESULTADOS,
    RUTA_TRAIN,
    RUTA_VAL_GO,
    RUTA_VAL_SINTETICA,
)
from entrenamiento.p02_datos import cargar_csv_etiquetado, cargar_entrenamiento_completo
from entrenamiento.p03_preprocesamiento import (
    normalizar_texto,
    preparar_lote_modelo,
    preparar_texto_modelo,
)
from entrenamiento.p04_seguridad import evaluar_riesgo
from entrenamiento.p05_modelos import construir_modelo_ganador
from entrenamiento.p06_metricas import evaluar_modelo_en_particiones, redondear


def _textos_validacion_normalizados():
    # devuelve el conjunto de textos de validacion para evitar fuga al corregir
    val_go = cargar_csv_etiquetado(RUTA_VAL_GO)
    val_sin = cargar_csv_etiquetado(RUTA_VAL_SINTETICA)
    return {normalizar_texto(f["texto"]) for f in val_go + val_sin}


def cargar_correcciones(ruta=RUTA_CORRECCIONES):
    # lee el archivo json local con los ejemplos guardados por el usuario
    if not ruta.exists():
        return []
    with ruta.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)
    return datos if isinstance(datos, list) else []


def guardar_lista_correcciones(correcciones, ruta=RUTA_CORRECCIONES):
    # escribe de forma explicita la lista actualizada en disco local
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(correcciones, archivo, ensure_ascii=False, indent=2)


def agregar_correccion(texto, etiqueta, ruta=RUTA_CORRECCIONES):
    # valida una frase nueva del usuario antes de guardarla en el archivo local
    # bloquea mensajes de riesgo y frases que ya pertenecen a validacion
    texto_limpio = (texto or "").strip()
    etiqueta_limpia = (etiqueta or "").strip().lower()

    if not texto_limpio:
        raise ValueError("Escribe o pega una frase antes de guardar.")
    if etiqueta_limpia not in CLASES:
        raise ValueError(f"La emoción debe ser una de: {', '.join(CLASES)}.")

    # impide guardar frases de suicidio o autolesion para entrenar el modelo
    riesgo = evaluar_riesgo(texto_limpio)
    if riesgo["es_riesgo"]:
        raise ValueError(
            "Este mensaje contiene señales de riesgo de suicidio o autolesión y no "
            "puede guardarse para entrenar. Si necesitas apoyo, comunícate con la "
            "Línea 113 Salud (Opción 5) del MINSA."
        )

    normalizado = normalizar_texto(texto_limpio)
    if not normalizado:
        raise ValueError("La frase debe contener al menos una palabra válida.")

    # evita fugas impidiendo guardar textos que ya estan en validacion
    if normalizado in _textos_validacion_normalizados():
        raise ValueError(
            "Esa frase ya pertenece al conjunto de validación. No se guarda para "
            "evitar contaminar las métricas."
        )

    correcciones = cargar_correcciones(ruta)
    ahora = datetime.now(timezone.utc).isoformat()

    # si la frase ya estaba guardada actualiza su etiqueta en lugar de duplicar
    for item in correcciones:
        if normalizar_texto(item["texto"]) == normalizado:
            item["texto"] = texto_limpio
            item["etiqueta"] = etiqueta_limpia
            item["actualizado_utc"] = ahora
            guardar_lista_correcciones(correcciones, ruta)
            return item

    # crea un registro nuevo con id corto y fecha utc y lo escribe en disco
    nuevo = {
        "id": uuid.uuid4().hex[:10],
        "texto": texto_limpio,
        "etiqueta": etiqueta_limpia,
        "creado_utc": ahora,
    }
    correcciones.append(nuevo)
    guardar_lista_correcciones(correcciones, ruta)
    return nuevo


def eliminar_correccion(corr_id, ruta=RUTA_CORRECCIONES):
    # elimina una correccion puntual por su identificador local
    correcciones = cargar_correcciones(ruta)
    nuevas = [item for item in correcciones if item.get("id") != corr_id]
    eliminado = len(nuevas) != len(correcciones)
    if eliminado:
        guardar_lista_correcciones(nuevas, ruta)
    return eliminado


def limpiar_correcciones(ruta=RUTA_CORRECCIONES):
    # borra todas las correcciones locales almacenadas por el usuario
    guardar_lista_correcciones([], ruta)
    return True


def cargar_estado_modelo(ruta_estado=RUTA_ESTADO_MODELO):
    # lee el estado actual del modelo base frente al modelo activo
    if ruta_estado.exists():
        with ruta_estado.open("r", encoding="utf-8") as archivo:
            return json.load(archivo)
    return {
        "version_activa": "base",
        "numero_reentrenamientos": 0,
        "correcciones_incorporadas": 0,
        "metricas_base": None,
        "metricas_actual": None,
        "delta_validacion": {},
    }


def _metricas_base_referencia():
    # obtiene las metricas congeladas del modelo base desde resultados o estado
    if RUTA_ESTADO_MODELO.exists():
        estado = cargar_estado_modelo()
        if estado.get("metricas_base"):
            return estado["metricas_base"]
    if RUTA_RESULTADOS.exists():
        with RUTA_RESULTADOS.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
        for modelo in datos.get("modelos", []):
            if modelo["clave"] == datos.get("modelo_ganador_clave"):
                return modelo["metricas"]
    return None


def _calcular_delta(metricas_base, metricas_actual):
    # calcula la variacion en validacion entre el nuevo modelo y el modelo base
    return {
        "macro_f1_go": redondear(
            metricas_actual["validacion_go"]["macro_f1"]
            - metricas_base["validacion_go"]["macro_f1"]
        ),
        "exactitud_go": redondear(
            metricas_actual["validacion_go"]["exactitud"]
            - metricas_base["validacion_go"]["exactitud"]
        ),
        "macro_f1_sintetica": redondear(
            metricas_actual["validacion_sintetica"]["macro_f1"]
            - metricas_base["validacion_sintetica"]["macro_f1"]
        ),
        "exactitud_sintetica": redondear(
            metricas_actual["validacion_sintetica"]["exactitud"]
            - metricas_base["validacion_sintetica"]["exactitud"]
        ),
        "macro_f1_combinada": redondear(
            metricas_actual["validacion_combinada"]["macro_f1"]
            - metricas_base["validacion_combinada"]["macro_f1"]
        ),
        "promedio_macro_f1": redondear(
            metricas_actual["promedio_macro_f1"]
            - metricas_base["promedio_macro_f1"]
        ),
    }


def reentrenar_con_correcciones(
    ruta_correcciones=RUTA_CORRECCIONES,
    ruta_modelo_activo=RUTA_MODELO_ACTIVO,
    ruta_estado=RUTA_ESTADO_MODELO,
):
    # reentrena desde cero combinando train.csv con las correcciones seguras
    correcciones = cargar_correcciones(ruta_correcciones)
    seguras = [
        item
        for item in correcciones
        if not evaluar_riesgo(item.get("texto", ""))["es_riesgo"]
        and item.get("etiqueta") in CLASES
    ]
    if not seguras:
        raise ValueError(
            "No hay correcciones válidas guardadas para entrenar una nueva versión."
        )

    train_filas = cargar_entrenamiento_completo(RUTA_TRAIN)
    val_go_filas = cargar_csv_etiquetado(RUTA_VAL_GO)
    val_sin_filas = cargar_csv_etiquetado(RUTA_VAL_SINTETICA)

    x_train = preparar_lote_modelo([f["texto"] for f in train_filas])
    y_train = [f["etiqueta"] for f in train_filas]

    # repite cada correccion un numero acotado para superar min_df sin distorsionar
    for item in seguras:
        texto_norm = preparar_texto_modelo(item["texto"])
        if not texto_norm:
            continue
        for _ in range(PESO_REPETICION_CORRECCION):
            x_train.append(texto_norm)
            y_train.append(item["etiqueta"])

    x_go = preparar_lote_modelo([f["texto"] for f in val_go_filas])
    y_go = [f["etiqueta"] for f in val_go_filas]
    x_sin = preparar_lote_modelo([f["texto"] for f in val_sin_filas])
    y_sin = [f["etiqueta"] for f in val_sin_filas]

    nuevo_modelo = construir_modelo_ganador()
    nuevo_modelo.fit(x_train, y_train)
    metricas_actual = evaluar_modelo_en_particiones(
        nuevo_modelo, x_go, y_go, x_sin, y_sin
    )

    joblib.dump(nuevo_modelo, ruta_modelo_activo)

    estado_previo = cargar_estado_modelo(ruta_estado)
    numero = int(estado_previo.get("numero_reentrenamientos", 0)) + 1
    metricas_base = _metricas_base_referencia() or metricas_actual

    nuevo_estado = {
        "version_activa": f"reentrenado_v{numero}",
        "numero_reentrenamientos": numero,
        "correcciones_incorporadas": len(seguras),
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
        "metricas_base": metricas_base,
        "metricas_actual": metricas_actual,
        "delta_validacion": _calcular_delta(metricas_base, metricas_actual),
    }

    with ruta_estado.open("w", encoding="utf-8") as archivo:
        json.dump(nuevo_estado, archivo, ensure_ascii=False, indent=2)

    return nuevo_modelo, nuevo_estado


def restaurar_modelo_base(
    ruta_modelo_base=RUTA_MODELO_BASE,
    ruta_modelo_activo=RUTA_MODELO_ACTIVO,
    ruta_estado=RUTA_ESTADO_MODELO,
):
    # vuelve al modelo base congelado sin borrar el historial de correcciones
    if not ruta_modelo_base.exists():
        raise FileNotFoundError(
            "No se encontro el modelo base. Ejecuta entrenamiento/p08_experimentos.py."
        )

    shutil.copy2(ruta_modelo_base, ruta_modelo_activo)
    modelo_base = joblib.load(ruta_modelo_activo)

    estado_previo = cargar_estado_modelo(ruta_estado)
    metricas_base = _metricas_base_referencia()
    if metricas_base is None:
        val_go_filas = cargar_csv_etiquetado(RUTA_VAL_GO)
        val_sin_filas = cargar_csv_etiquetado(RUTA_VAL_SINTETICA)
        metricas_base = evaluar_modelo_en_particiones(
            modelo_base,
            preparar_lote_modelo([f["texto"] for f in val_go_filas]),
            [f["etiqueta"] for f in val_go_filas],
            preparar_lote_modelo([f["texto"] for f in val_sin_filas]),
            [f["etiqueta"] for f in val_sin_filas],
        )

    estado_restaurado = {
        "version_activa": "base",
        "numero_reentrenamientos": int(estado_previo.get("numero_reentrenamientos", 0)),
        "correcciones_incorporadas": 0,
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
        "metricas_base": metricas_base,
        "metricas_actual": metricas_base,
        "delta_validacion": {
            "macro_f1_go": 0.0,
            "exactitud_go": 0.0,
            "macro_f1_sintetica": 0.0,
            "exactitud_sintetica": 0.0,
            "macro_f1_combinada": 0.0,
            "promedio_macro_f1": 0.0,
        },
    }

    with ruta_estado.open("w", encoding="utf-8") as archivo:
        json.dump(estado_restaurado, archivo, ensure_ascii=False, indent=2)

    return modelo_base, estado_restaurado
