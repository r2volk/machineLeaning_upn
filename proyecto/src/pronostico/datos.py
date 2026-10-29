from pathlib import Path

import pandas as pd


# Busca datasets/train.csv desde la ubicación de este archivo.
CARPETA_MACHINE_LEARNING = Path(__file__).resolve().parents[3]
ARCHIVO_TRAIN = CARPETA_MACHINE_LEARNING / "datasets" / "train.csv"

COLUMNAS_REQUERIDAS = {"date", "store", "item", "sales"}


def cargar_train(ruta=ARCHIVO_TRAIN):
    """Lee train.csv, comprueba sus columnas y ordena las observaciones."""
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    datos = pd.read_csv(ruta, parse_dates=["date"])

    columnas_faltantes = COLUMNAS_REQUERIDAS - set(datos.columns)
    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(columnas_faltantes)}"
        )

    datos = datos.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    return datos


def mostrar_resumen(datos):
    """Imprime información básica para entender el dataset."""
    print("RESUMEN DE train.csv")
    print(f"Filas: {len(datos):,}")
    print(f"Columnas: {list(datos.columns)}")
    print(
        f"Período: {datos['date'].min():%Y-%m-%d} "
        f"a {datos['date'].max():%Y-%m-%d}"
    )
    print(f"Tiendas: {datos['store'].nunique()}")
    print(f"Artículos: {datos['item'].nunique()}")
    print(f"Valores vacíos: {int(datos.isna().sum().sum())}")
    print(
        "Filas duplicadas por fecha, tienda y artículo: "
        f"{datos.duplicated(['date', 'store', 'item']).sum():,}"
    )
    print()
    print("Resumen de las ventas:")
    print(datos["sales"].describe())


if __name__ == "__main__":
    datos = cargar_train()
    mostrar_resumen(datos)