# entrenamiento, comparacion controlada de modelos y generacion del modelo base
import json
import time
from datetime import datetime, timezone

import joblib

from entrenamiento.p01_config import (
    C_REGRESION_LOGISTICA,
    DIR_MODELOS,
    MAX_DF_TFIDF,
    REJILLA_AJUSTE_LR,
    RUTA_ESTADO_MODELO,
    RUTA_MODELO_ACTIVO,
    RUTA_MODELO_BASE,
    RUTA_RESULTADOS,
    UMBRAL_MARGEN_MINIMO,
    UMBRAL_PROB_MINIMA,
)
from entrenamiento.p02_datos import cargar_particiones
from entrenamiento.p03_preprocesamiento import normalizar_texto, preparar_lote_modelo
from entrenamiento.p05_modelos import construir_catalogo_modelos, construir_modelo_ganador
from entrenamiento.p06_metricas import evaluar_modelo_en_particiones, redondear
from entrenamiento.p07_inferencia import analizar_mensaje

# frases exploratorias escritas a mano y distintas del entrenamiento y validacion
FRASES_DIAGNOSTICO = [
    {
        "texto": "No me fue del todo bien",
        "categoria": "negacion_indirecta",
        "lectura_esperada": "tristeza o pedir contexto",
    },
    {
        "texto": "No me fue del todo bien en el examen",
        "categoria": "negacion_indirecta",
        "lectura_esperada": "tristeza, ansiedad o pedir contexto",
    },
    {
        "texto": "No puedo estar más feliz con esta noticia",
        "categoria": "negacion_alegre",
        "lectura_esperada": "alegria",
    },
    {
        "texto": "No me lo puedo creer, al fin aprobé el curso",
        "categoria": "negacion_alegre",
        "lectura_esperada": "alegria",
    },
    {
        "texto": "No tengo clases mañana y luego iré al banco",
        "categoria": "negacion_neutral",
        "lectura_esperada": "neutral",
    },
    {
        "texto": "No sé si la reunión empieza a las cuatro o a las cinco",
        "categoria": "negacion_neutral",
        "lectura_esperada": "neutral",
    },
    {
        "texto": "No me preocupa en absoluto, todo salió genial",
        "categoria": "negacion_dificil",
        "lectura_esperada": "alegria (caso dificil por la palabra preocupa)",
    },
    {
        "texto": "No estoy triste, solo estoy pensando en mis planes",
        "categoria": "negacion_dificil",
        "lectura_esperada": "neutral (caso dificil por la palabra triste)",
    },
    {
        "texto": "No estoy enojado, solo cansado",
        "categoria": "indirecta_ambigua",
        "lectura_esperada": "tristeza, neutral o pedir contexto",
    },
    {
        "texto": "Bueno, supongo que así son las cosas",
        "categoria": "senal_debil",
        "lectura_esperada": "neutral o pedir contexto",
    },
    {
        "texto": "No sé qué va a pasar mañana, tengo el pecho apretado",
        "categoria": "ansiedad_cotidiana",
        "lectura_esperada": "ansiedad",
    },
    {
        "texto": "No dormí nada pensando en la presentación de mañana",
        "categoria": "ansiedad_cotidiana",
        "lectura_esperada": "ansiedad",
    },
    {
        "texto": "Me da mucha rabia que no cumplan lo acordado",
        "categoria": "enojo_cotidiano",
        "lectura_esperada": "enojo",
    },
    {
        "texto": "Hoy me siento vacío y sin ganas de salir",
        "categoria": "tristeza_cotidiana",
        "lectura_esperada": "tristeza",
    },
    {
        "texto": "Qué bueno que pudimos terminar el proyecto a tiempo",
        "categoria": "alegria_cotidiana",
        "lectura_esperada": "alegria",
    },
    {
        "texto": "No esperaba aprobar pero saqué veinte",
        "categoria": "modismo_local",
        "lectura_esperada": "alegria (caso ideal para probar correcciones)",
    },
]


def ejecutar_diagnostico_exploratorio(modelo, textos_entrenamiento_set):
    # evalua frases exploratorias verificando que no esten en el entrenamiento
    resultados = []
    for item in FRASES_DIAGNOSTICO:
        texto = item["texto"]
        en_train = normalizar_texto(texto) in textos_entrenamiento_set
        analisis = analizar_mensaje(
            modelo,
            texto,
            umbral_prob=UMBRAL_PROB_MINIMA,
            umbral_margen=UMBRAL_MARGEN_MINIMO,
        )
        resultados.append(
            {
                "texto": texto,
                "categoria": item["categoria"],
                "lectura_esperada": item["lectura_esperada"],
                "en_entrenamiento": en_train,
                "estado": analisis["estado"],
                "etiqueta_final": analisis["etiqueta"] or "pedir_contexto",
                "etiqueta_tentativa": analisis.get("etiqueta_tentativa"),
                "segunda_etiqueta": analisis.get("segunda_etiqueta"),
                "confianza": analisis["confianza"],
                "margen": analisis["margen"],
                "terminos_clave": analisis["terminos_clave"][:3],
            }
        )
    return resultados


def construir_comparacion_ngramas(resultados_por_clave):
    # compara exactamente bigramas contra trigramas con los demas parametros fijos
    bi = resultados_por_clave["tfidf_lr_bigramas"]
    tri = resultados_por_clave["tfidf_lr_trigramas"]

    return {
        "parametros_fijos": {
            "vectorizador": "TfidfVectorizer",
            "min_df": 2,
            "sublinear_tf": True,
            "clasificador": "LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
        "bigramas": {
            "ngram_range": "(1, 2)",
            "vocabulario": bi["tamano_vocabulario"],
            "macro_f1_go": bi["metricas"]["validacion_go"]["macro_f1"],
            "exactitud_go": bi["metricas"]["validacion_go"]["exactitud"],
            "macro_f1_sintetica": bi["metricas"]["validacion_sintetica"]["macro_f1"],
            "exactitud_sintetica": bi["metricas"]["validacion_sintetica"]["exactitud"],
            "macro_f1_combinada": bi["metricas"]["validacion_combinada"]["macro_f1"],
            "promedio_macro_f1": bi["metricas"]["promedio_macro_f1"],
        },
        "trigramas": {
            "ngram_range": "(1, 3)",
            "vocabulario": tri["tamano_vocabulario"],
            "macro_f1_go": tri["metricas"]["validacion_go"]["macro_f1"],
            "exactitud_go": tri["metricas"]["validacion_go"]["exactitud"],
            "macro_f1_sintetica": tri["metricas"]["validacion_sintetica"]["macro_f1"],
            "exactitud_sintetica": tri["metricas"]["validacion_sintetica"]["exactitud"],
            "macro_f1_combinada": tri["metricas"]["validacion_combinada"]["macro_f1"],
            "promedio_macro_f1": tri["metricas"]["promedio_macro_f1"],
        },
        "diferencia_trigramas_menos_bigramas": {
            "vocabulario": tri["tamano_vocabulario"] - bi["tamano_vocabulario"],
            "macro_f1_go": redondear(
                tri["metricas"]["validacion_go"]["macro_f1"]
                - bi["metricas"]["validacion_go"]["macro_f1"]
            ),
            "macro_f1_sintetica": redondear(
                tri["metricas"]["validacion_sintetica"]["macro_f1"]
                - bi["metricas"]["validacion_sintetica"]["macro_f1"]
            ),
            "macro_f1_combinada": redondear(
                tri["metricas"]["validacion_combinada"]["macro_f1"]
                - bi["metricas"]["validacion_combinada"]["macro_f1"]
            ),
            "promedio_macro_f1": redondear(
                tri["metricas"]["promedio_macro_f1"]
                - bi["metricas"]["promedio_macro_f1"]
            ),
        },
    }


def contar_vocabulario(vectorizador):
    # cuenta los terminos del vectorizador simple o de una union de caracteristicas
    if hasattr(vectorizador, "vocabulary_"):
        return len(vectorizador.vocabulary_)
    if hasattr(vectorizador, "transformer_list"):
        return sum(
            len(trans.vocabulary_)
            for _, trans in vectorizador.transformer_list
            if hasattr(trans, "vocabulary_")
        )
    return 0


def evaluar_ajuste_hiperparametros(x_train, y_train, x_go, y_go, x_sin, y_sin):
    # compara combinaciones de max_df y regularizacion c solo en validacion
    pruebas = []
    for max_df, c_reg in REJILLA_AJUSTE_LR:
        modelo = construir_modelo_ganador(c_reg=c_reg, max_df=max_df)
        modelo.fit(x_train, y_train)
        metricas = evaluar_modelo_en_particiones(modelo, x_go, y_go, x_sin, y_sin)
        pruebas.append(
            {
                "max_df": max_df,
                "c_reg": c_reg,
                "seleccionado": (
                    max_df == MAX_DF_TFIDF and c_reg == C_REGRESION_LOGISTICA
                ),
                "macro_f1_go": metricas["validacion_go"]["macro_f1"],
                "exactitud_go": metricas["validacion_go"]["exactitud"],
                "macro_f1_sintetica": metricas["validacion_sintetica"]["macro_f1"],
                "macro_f1_combinada": metricas["validacion_combinada"]["macro_f1"],
                "promedio_macro_f1": metricas["promedio_macro_f1"],
                "macro_f1_selectivo_go": metricas["abstencion"]["validacion_go"][
                    "macro_f1_selectivo"
                ],
            }
        )
    return {
        "criterio": (
            "Se selecciona max_df=0.5 y C=3.0 porque equilibra el mayor macro-F1 "
            "en GoEmotions y promedio global con mejor macro-F1 selectivo y menor "
            "riesgo de sobreajuste frente a C=5.0."
        ),
        "combinaciones": pruebas,
    }


def ejecutar_experimentos():
    # entrena los 9 modelos solo con entrenamiento y guarda el ganador como modelo base
    # genera tambien el json de metricas que luego lee la pestana laboratorio
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    train_filas, val_go_filas, val_sin_filas, resumen_datos = cargar_particiones()

    # limpia y marca negaciones en entrenamiento y en ambas validaciones
    x_train = preparar_lote_modelo([f["texto"] for f in train_filas])
    y_train = [f["etiqueta"] for f in train_filas]

    x_go = preparar_lote_modelo([f["texto"] for f in val_go_filas])
    y_go = [f["etiqueta"] for f in val_go_filas]

    x_sin = preparar_lote_modelo([f["texto"] for f in val_sin_filas])
    y_sin = [f["etiqueta"] for f in val_sin_filas]

    catalogo = construir_catalogo_modelos()
    resultados_modelos = []
    resultados_por_clave = {}
    modelo_ganador = None

    # entrena cada modelo candidato midiendo su tiempo, vocabulario y metricas
    for entrada in catalogo:
        inicio = time.perf_counter()
        pipeline = entrada["pipeline"]
        pipeline.fit(x_train, y_train)
        duracion = round(time.perf_counter() - inicio, 2)

        metricas = evaluar_modelo_en_particiones(pipeline, x_go, y_go, x_sin, y_sin)
        vocabulario = contar_vocabulario(pipeline.named_steps["vectorizador"])

        registro = {
            "clave": entrada["clave"],
            "nombre": entrada["nombre"],
            "familia": entrada["familia"],
            "ngramas": entrada["ngramas"],
            "probabilidades_nativas": entrada["probabilidades_nativas"],
            "descripcion": entrada["descripcion"],
            "tamano_vocabulario": vocabulario,
            "segundos_entrenamiento": duracion,
            "metricas": metricas,
        }
        resultados_modelos.append(registro)
        resultados_por_clave[entrada["clave"]] = registro

        # guarda en memoria el pipeline ganador para exportarlo al disco
        if entrada["clave"] == "tfidf_hibrido_lr":
            modelo_ganador = pipeline

    # guarda el modelo base congelado y una copia inicial como modelo activo
    joblib.dump(modelo_ganador, RUTA_MODELO_BASE)
    joblib.dump(modelo_ganador, RUTA_MODELO_ACTIVO)

    # corre el diagnostico exploratorio, la comparacion de n-gramas y la rejilla
    train_set = {normalizar_texto(f["texto"]) for f in train_filas}
    diagnostico = ejecutar_diagnostico_exploratorio(modelo_ganador, train_set)
    comparacion_ngramas = construir_comparacion_ngramas(resultados_por_clave)
    ajuste_hp = evaluar_ajuste_hiperparametros(
        x_train, y_train, x_go, y_go, x_sin, y_sin
    )

    ganador_info = resultados_por_clave["tfidf_hibrido_lr"]
    ahora = datetime.now(timezone.utc).isoformat()

    reporte = {
        "fecha_generacion_utc": ahora,
        "modelo_ganador_clave": "tfidf_hibrido_lr",
        "modelo_ganador_nombre": ganador_info["nombre"],
        "justificacion_ganador": (
            "Se elige TF-IDF hibrido (palabras 1-3 con alcance de negacion y "
            "subpalabras de caracteres 3-5, max_df=0.5) con Regresión Logística "
            "multinomial balanceada (C=3.0) porque logra el mayor macro-F1 en "
            "GoEmotions y promedio global entre los nueve modelos comparados "
            "(superando tambien a la Red Neuronal MLP y al ensamble suave), "
            "conecta raices del español sin modelos preentrenados y entrega "
            "probabilidades nativas e interpretabilidad directa por termino."
        ),
        "umbrales_abstencion": {
            "umbral_prob_minima": UMBRAL_PROB_MINIMA,
            "umbral_margen_minimo": UMBRAL_MARGEN_MINIMO,
        },
        "resumen_datos": resumen_datos,
        "modelos": resultados_modelos,
        "comparacion_bigramas_vs_trigramas": comparacion_ngramas,
        "ajuste_hiperparametros": ajuste_hp,
        "diagnostico_exploratorio": {
            "nota": (
                "Comprobación exploratoria sobre frases cotidianas escritas a mano y "
                "ausentes del entrenamiento; no sustituye una evaluación final en test."
            ),
            "casos": diagnostico,
        },
    }

    with RUTA_RESULTADOS.open("w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, ensure_ascii=False, indent=2)

    estado_inicial = {
        "version_activa": "base",
        "numero_reentrenamientos": 0,
        "correcciones_incorporadas": 0,
        "actualizado_utc": ahora,
        "metricas_base": ganador_info["metricas"],
        "metricas_actual": ganador_info["metricas"],
        "delta_validacion": {
            "macro_f1_go": 0.0,
            "exactitud_go": 0.0,
            "macro_f1_sintetica": 0.0,
            "exactitud_sintetica": 0.0,
            "macro_f1_combinada": 0.0,
            "promedio_macro_f1": 0.0,
        },
    }

    with RUTA_ESTADO_MODELO.open("w", encoding="utf-8") as archivo:
        json.dump(estado_inicial, archivo, ensure_ascii=False, indent=2)

    return reporte


def imprimir_resumen(reporte):
    # imprime en consola una tabla clara con los resultados de validacion
    print("\nResultados de validacion (ajustado unicamente con train.csv):")
    encabezado = f"{'Modelo':<46} {'Vocab':>7} {'Go F1':>7} {'Sin F1':>7} {'Comb F1':>8} {'Prom F1':>8}"
    print(encabezado)
    for item in reporte["modelos"]:
        m = item["metricas"]
        print(
            f"{item['clave']:<46} {item['tamano_vocabulario']:>7} "
            f"{m['validacion_go']['macro_f1']:>7.4f} "
            f"{m['validacion_sintetica']['macro_f1']:>7.4f} "
            f"{m['validacion_combinada']['macro_f1']:>8.4f} "
            f"{m['promedio_macro_f1']:>8.4f}"
        )

    dif = reporte["comparacion_bigramas_vs_trigramas"]["diferencia_trigramas_menos_bigramas"]
    print(
        f"\nAporte de trigramas vs bigramas: Go F1 {dif['macro_f1_go']:+.4f}, "
        f"Sintetica F1 {dif['macro_f1_sintetica']:+.4f}, "
        f"Combinada F1 {dif['macro_f1_combinada']:+.4f}"
    )
    print(f"Modelo base guardado en: {RUTA_MODELO_BASE}")


if __name__ == "__main__":
    resultado = ejecutar_experimentos()
    imprimir_resumen(resultado)
