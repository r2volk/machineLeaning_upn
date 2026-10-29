from pathlib import Path

import numpy as np
import pandas as pd

from datos import cargar_train
from modelos import COLUMNAS_ENTRADA, crear_modelos, preparar_variables
from variables import agregar_variables_historicas


CARPETA_MACHINE_LEARNING = Path(__file__).resolve().parents[3]
CARPETA_DATASETS = CARPETA_MACHINE_LEARNING / "datasets"
CARPETA_RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"

ARCHIVO_TEST = CARPETA_DATASETS / "test.csv"
ARCHIVO_EJEMPLO = CARPETA_DATASETS / "sample_submission.csv"
ARCHIVO_SALIDA = CARPETA_RESULTADOS / "predicciones_kaggle.csv"


# 1. Entrenar el candidato con todo train.csv
train_original = cargar_train()
train_con_variables = agregar_variables_historicas(train_original)

train_entrenable = train_con_variables.dropna(
    subset=["lag_7", "lag_14", "media_7_dias", "media_28_dias"]
).copy()

X_train = preparar_variables(train_entrenable)
y_train = train_entrenable["sales"]

modelo = crear_modelos()["Random Forest"]

print("Entrenando Random Forest con train.csv completo...")
modelo.fit(X_train, y_train)


# 2. Preparar el historial real de ventas para cada tienda y artículo
historiales = {}

for (tienda, articulo), grupo in train_original.groupby(["store", "item"]):
    historiales[(int(tienda), int(articulo))] = (
        grupo.sort_values("date")["sales"].astype(float).tolist()
    )

fecha_base = train_entrenable["date"].min()


# 3. Leer test.csv y ordenar sus fechas para predecir en secuencia
test = pd.read_csv(ARCHIVO_TEST, parse_dates=["date"])
test_ordenado = test.sort_values(
    ["date", "store", "item"]
).copy()

predicciones_por_id = {}

for fecha, filas_del_dia in test_ordenado.groupby("date", sort=True):
    entradas = []

    # Las predicciones de un mismo día se calculan antes de actualizar
    # el historial, para que ningún producto use una predicción del mismo día.
    for fila in filas_del_dia.itertuples():
        clave = (int(fila.store), int(fila.item))
        historial = historiales[clave]

        entradas.append({
            "store": int(fila.store),
            "item": int(fila.item),
            "day_index": (fecha - fecha_base).days,
            "day_of_week": fecha.dayofweek,
            "month": fecha.month,
            "lag_7": historial[-7],
            "lag_14": historial[-14],
            "media_7_dias": np.mean(historial[-7:]),
            "media_28_dias": np.mean(historial[-28:]),
        })

    X_dia = pd.DataFrame(entradas, columns=COLUMNAS_ENTRADA)
    ventas_estimadas = np.maximum(modelo.predict(X_dia), 0)

    # Guardar el pronóstico y agregarlo al historial para el día siguiente.
    for fila, venta_estimada in zip(
        filas_del_dia.itertuples(),
        ventas_estimadas,
    ):
        clave = (int(fila.store), int(fila.item))
        venta_estimada = float(venta_estimada)

        predicciones_por_id[int(fila.id)] = venta_estimada
        historiales[clave].append(venta_estimada)


# 4. Crear la entrega respetando IDs y columnas de sample_submission.csv
entrega = test[["id"]].copy()
entrega["sales"] = entrega["id"].map(predicciones_por_id)
entrega = entrega.sort_values("id").reset_index(drop=True)

ejemplo = pd.read_csv(ARCHIVO_EJEMPLO)

if entrega["sales"].isna().any():
    raise ValueError("Quedaron filas de test.csv sin predicción.")

if not np.array_equal(entrega["id"].to_numpy(), ejemplo["id"].to_numpy()):
    raise ValueError("Los IDs no coinciden con sample_submission.csv.")

if (entrega["sales"] < 0).any():
    raise ValueError("Se encontraron predicciones negativas.")

CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
entrega.to_csv(ARCHIVO_SALIDA, index=False)

print()
print(f"Filas predichas: {len(entrega):,}")
print(f"Columnas: {list(entrega.columns)}")
print(f"Ventas predichas mínimas: {entrega['sales'].min():.2f}")
print(f"Ventas predichas máximas: {entrega['sales'].max():.2f}")
print(f"Archivo creado: {ARCHIVO_SALIDA}")
print("Son predicciones de Kaggle, no ventas reales de Kapaq.")