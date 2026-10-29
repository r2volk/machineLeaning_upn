from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


CARPETA_PROYECTO = Path(__file__).resolve().parent
ARCHIVO_DATOS = CARPETA_PROYECTO.parent / "datasets" / "train.csv"
DIAS_VALIDACION = 90

# Leer y ordenar los datos
datos = pd.read_csv(ARCHIVO_DATOS, parse_dates=["date"])
datos = datos.sort_values(["store", "item", "date"]).reset_index(drop=True)

# Referencia: repetir las ventas del mismo artículo y tienda hace 7 días
datos["ventas_hace_7_dias"] = (
    datos.groupby(["store", "item"])["sales"].shift(7)
)

# Variables construidas a partir de la fecha
datos["day_index"] = (datos["date"] - datos["date"].min()).dt.days
datos["day_of_week"] = datos["date"].dt.dayofweek
datos["month"] = datos["date"].dt.month
datos = datos.dropna(subset=["ventas_hace_7_dias"]).copy()

# Separación cronológica: fechas antiguas para entrenar y fechas recientes para evaluar
fecha_ultima = datos["date"].max()
fecha_inicio_validacion = fecha_ultima - pd.Timedelta(
    days=DIAS_VALIDACION - 1
)

entrenamiento = datos[datos["date"] < fecha_inicio_validacion].copy()
validacion = datos[datos["date"] >= fecha_inicio_validacion].copy()

# Entradas X y respuesta y
columnas_categoricas = ["store", "item"]
columnas_numericas = ["day_index", "day_of_week", "month"]
columnas_entrada = columnas_categoricas + columnas_numericas

X_entrenamiento = entrenamiento[columnas_entrada]
y_entrenamiento = entrenamiento["sales"]

X_validacion = validacion[columnas_entrada]
y_validacion = validacion["sales"]

# Convertir los IDs de tienda y artículo en categorías
preparacion = ColumnTransformer(
    transformers=[
        (
            "categorias",
            OneHotEncoder(handle_unknown="ignore"),
            columnas_categoricas,
        ),
        ("numericas", "passthrough", columnas_numericas),
    ]
)

# Modelo 1: regresión lineal
regresion = make_pipeline(
    preparacion,
    LinearRegression(),
)
regresion.fit(X_entrenamiento, y_entrenamiento)

pred_regresion = np.maximum(
    regresion.predict(X_validacion),
    0,
)

# Modelo 2: Random Forest
bosque = make_pipeline(
    preparacion,
    RandomForestRegressor(
        n_estimators=20,
        max_depth=12,
        min_samples_leaf=10,
        max_samples=0.10,
        n_jobs=2,
        random_state=42,
    ),
)

print("Entrenando Random Forest; puede tardar un poco...")
bosque.fit(X_entrenamiento, y_entrenamiento)

pred_bosque = np.maximum(
    bosque.predict(X_validacion),
    0,
)

# Referencia: ventas reales observadas hace 7 días
pred_referencia = validacion["ventas_hace_7_dias"].to_numpy()

# Calcular MAE y RMSE para los tres métodos
metodos = [
    ("Referencia: ventas de hace 7 días", pred_referencia),
    ("Regresión lineal", pred_regresion),
    ("Random Forest", pred_bosque),
]

resultados = []

for nombre, predicciones in metodos:
    mae = mean_absolute_error(y_validacion, predicciones)
    rmse = np.sqrt(mean_squared_error(y_validacion, predicciones))

    resultados.append({
        "Método": nombre,
        "MAE": mae,
        "RMSE": rmse,
    })

print()
print(
    f"Entrenamiento: hasta "
    f"{fecha_inicio_validacion - pd.Timedelta(days=1):%Y-%m-%d}"
)
print(
    f"Validación:    {fecha_inicio_validacion:%Y-%m-%d} "
    f"a {fecha_ultima:%Y-%m-%d}"
)
print(f"Filas de entrenamiento: {len(entrenamiento):,}")
print(f"Filas de validación:    {len(validacion):,}")
print()
print(
    pd.DataFrame(resultados).to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
        },
    )
)
print()
print("Los datos son de Kaggle; no representan ventas reales de Kapaq.")