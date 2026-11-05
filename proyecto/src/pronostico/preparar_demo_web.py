"""Prepara predicciones Kaggle para la pantalla de demostración del frontend.

No entrena modelos ni accede a la base de datos de Kapaq. Une las predicciones
ya generadas con test.csv usando el identificador oficial de Kaggle.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CARPETA_ML = Path(__file__).resolve().parents[3]
ARCHIVO_TEST = CARPETA_ML / "datasets" / "test.csv"
ARCHIVO_PREDICCIONES = (
    CARPETA_ML / "proyecto" / "resultados" / "predicciones_kaggle.csv"
)
DESTINO_POR_DEFECTO = (
    CARPETA_ML.parent
    / "Proyecto-Minimarket"
    / "frontend"
    / "src"
    / "assets"
    / "data"
    / "pronosticos-kaggle.json"
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepara el archivo de pronósticos Kaggle para el frontend."
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=DESTINO_POR_DEFECTO,
        help="Ruta del JSON que consumirá la web.",
    )
    args = parser.parse_args()

    if not ARCHIVO_TEST.exists():
        raise FileNotFoundError(f"No se encontró test.csv: {ARCHIVO_TEST}")
    if not ARCHIVO_PREDICCIONES.exists():
        raise FileNotFoundError(
            "Falta predicciones_kaggle.csv. Ejecuta primero "
            "proyecto/src/pronostico/predecir_test.py."
        )

    test = pd.read_csv(ARCHIVO_TEST, parse_dates=["date"])
    predicciones = pd.read_csv(ARCHIVO_PREDICCIONES)

    if not {"id", "date", "store", "item"}.issubset(test.columns):
        raise ValueError("test.csv no tiene las columnas Kaggle esperadas.")
    if not {"id", "sales"}.issubset(predicciones.columns):
        raise ValueError("El CSV de predicciones debe tener id y sales.")
    if test["id"].duplicated().any() or predicciones["id"].duplicated().any():
        raise ValueError("Hay IDs duplicados; se cancela la preparación.")
    if len(test) != len(predicciones) or set(test["id"]) != set(predicciones["id"]):
        raise ValueError("Los IDs de predicción no coinciden con test.csv.")

    combinado = test.merge(
        predicciones[["id", "sales"]],
        on="id",
        how="left",
        validate="one_to_one",
    ).sort_values(["date", "store", "item"])

    if combinado["sales"].isna().any() or not np.isfinite(
        combinado["sales"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Hay predicciones faltantes o no finitas.")
    if (combinado["sales"] < 0).any():
        raise ValueError("Hay predicciones negativas.")

    documento = {
        "fuente": "Kaggle — Store Item Demand Forecasting Challenge",
        "tipo": "demostracion_datos_externos",
        "modelo": "Random Forest",
        "nota": (
            "Estas predicciones usan datos externos de Kaggle. No son ventas, "
            "productos ni recomendaciones de inventario de Kapaq."
        ),
        "predicciones": [
            {
                "date": fila.date.strftime("%Y-%m-%d"),
                "store": int(fila.store),
                "item": int(fila.item),
                "sales": round(float(fila.sales), 2),
            }
            for fila in combinado.itertuples(index=False)
        ],
    }

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        json.dumps(documento, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"Filas preparadas: {len(documento['predicciones']):,}")
    print(f"Fechas: {combinado['date'].min():%Y-%m-%d} a "
          f"{combinado['date'].max():%Y-%m-%d}")
    print(f"Archivo para la demo web: {args.salida}")
    print("Origen Kaggle; no son ventas ni recomendaciones reales de Kapaq.")


if __name__ == "__main__":
    main()
 