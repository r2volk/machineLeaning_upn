# evaluacion final de una sola vez del modelo congelado sobre un archivo test
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from entrenamiento.p01_config import (
    DIR_MODELOS,
    RUTA_MODELO_BASE,
    RUTA_TRAIN,
    UMBRAL_MARGEN_MINIMO,
    UMBRAL_PROB_MINIMA,
)
from entrenamiento.p02_datos import (
    cargar_csv_etiquetado,
    cargar_entrenamiento_completo,
    resumir_particion,
)
from entrenamiento.p03_preprocesamiento import normalizar_texto, preparar_lote_modelo
from entrenamiento.p06_metricas import evaluar_abstencion, evaluar_predicciones
from entrenamiento.p07_inferencia import analizar_mensaje


def evaluar_conjunto_test(
    ruta_test,
    ruta_modelo=RUTA_MODELO_BASE,
    dir_salida=None,
    forzar=False,
):
    # evalua el modelo base congelado una sola vez sobre un csv de prueba final
    # guarda metricas en json y predicciones fila por fila en csv para auditoria
    salida = Path(dir_salida) if dir_salida else (DIR_MODELOS / "evaluacion_final")
    ruta_metricas = salida / "metricas_test.json"
    ruta_predicciones = salida / "predicciones_test.csv"

    # bloquea repetir la evaluacion por accidente para respetar la regla del test unico
    if ruta_metricas.exists() and not forzar:
        raise RuntimeError(
            "Ya existe una evaluacion final guardada en "
            f"{ruta_metricas}. El conjunto test debe usarse una sola vez; "
            "si necesitas repetirla conscientemente usa --forzar."
        )

    # carga el modelo congelado, el archivo de test y entrenamiento para auditar solape
    modelo = joblib.load(ruta_modelo)
    test_filas = cargar_csv_etiquetado(ruta_test)
    train_filas = cargar_entrenamiento_completo(RUTA_TRAIN)

    train_set = {normalizar_texto(f["texto"]) for f in train_filas}
    test_norm = [normalizar_texto(f["texto"]) for f in test_filas]
    test_limpios = preparar_lote_modelo([f["texto"] for f in test_filas])
    y_real = [f["etiqueta"] for f in test_filas]
    # predice sobre todo el test sin ajustar nada en el modelo
    y_pred = list(modelo.predict(test_limpios))

    metricas = evaluar_predicciones(y_real, y_pred)
    abstencion = (
        evaluar_abstencion(
            modelo,
            test_limpios,
            y_real,
            umbral_prob=UMBRAL_PROB_MINIMA,
            umbral_margen=UMBRAL_MARGEN_MINIMO,
        )
        if hasattr(modelo, "predict_proba")
        else None
    )

    # cuenta si alguna frase del test ya existia en entrenamiento
    solape_train = sum(1 for t in test_norm if t in train_set)

    # escribe las predicciones detalladas de cada mensaje para revision posterior
    salida.mkdir(parents=True, exist_ok=True)
    with ruta_predicciones.open("w", encoding="utf-8", newline="") as archivo:
        campos = [
            "texto",
            "etiqueta_real",
            "prediccion_directa",
            "decision_chatbot",
            "pide_contexto",
            "es_riesgo",
            "confianza",
            "margen",
            "acierto_directo",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for fila, pred_dir in zip(test_filas, y_pred):
            analisis = analizar_mensaje(modelo, fila["texto"])
            escritor.writerow(
                {
                    "texto": fila["texto"],
                    "etiqueta_real": fila["etiqueta"],
                    "prediccion_directa": pred_dir,
                    "decision_chatbot": analisis["etiqueta"] or analisis["estado"],
                    "pide_contexto": analisis["pide_contexto"],
                    "es_riesgo": analisis["es_riesgo"],
                    "confianza": analisis["confianza"],
                    "margen": analisis["margen"],
                    "acierto_directo": int(pred_dir == fila["etiqueta"]),
                }
            )

    reporte = {
        "fecha_evaluacion_utc": datetime.now(timezone.utc).isoformat(),
        "ruta_modelo": str(Path(ruta_modelo).resolve()),
        "ruta_test": str(Path(ruta_test).resolve()),
        "resumen_test": resumir_particion(test_filas),
        "solape_con_train": solape_train,
        "metricas_directas": metricas,
        "metricas_con_abstencion": abstencion,
        "archivo_predicciones": str(ruta_predicciones.resolve()),
    }

    with ruta_metricas.open("w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, ensure_ascii=False, indent=2)

    return reporte


def main():
    # analiza argumentos de consola y ejecuta la evaluacion final documentada
    parser = argparse.ArgumentParser(
        description="Evalua una sola vez el modelo congelado sobre un archivo CSV de test."
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Ruta al CSV de test con columnas texto y etiqueta.",
    )
    parser.add_argument(
        "--modelo",
        default=str(RUTA_MODELO_BASE),
        help="Ruta al modelo congelado (.joblib). Por defecto usa modelo_base.joblib.",
    )
    parser.add_argument(
        "--salida",
        default=str(DIR_MODELOS / "evaluacion_final"),
        help="Carpeta donde se guardan metricas_test.json y predicciones_test.csv.",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Sobrescribe una evaluacion previa si ya existe.",
    )
    args = parser.parse_args()

    reporte = evaluar_conjunto_test(
        ruta_test=args.test,
        ruta_modelo=args.modelo,
        dir_salida=args.salida,
        forzar=args.forzar,
    )
    m = reporte["metricas_directas"]
    print(f"Evaluacion final completada sobre {m['total']} mensajes.")
    print(f"Macro-F1: {m['macro_f1']:.4f} | Exactitud: {m['exactitud']:.4f}")
    if reporte["metricas_con_abstencion"]:
        a = reporte["metricas_con_abstencion"]
        print(
            f"Con abstencion: etiquetados {a['etiquetados']}/{a['total']} "
            f"({a['cobertura']*100:.1f}%), aciertos {a['aciertos_en_etiquetados']} "
            f"(exactitud selectiva {a['exactitud_selectiva']:.4f})"
        )
    print(f"Resultados guardados en: {args.salida}")


if __name__ == "__main__":
    main()
