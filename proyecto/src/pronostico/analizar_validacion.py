from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datos import cargar_train
from evaluacion import separar_cronologicamente
from modelos import crear_modelos, preparar_variables
from variables import agregar_variables_historicas


# Carpeta para guardar los resultados de este análisis
CARPETA_RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"

# Cargar datos y crear variables basadas en el historial de ventas
datos = cargar_train()
datos = agregar_variables_historicas(datos)

columnas_historicas = [
    "lag_7",
    "lag_14",
    "media_7_dias",
    "media_28_dias",
]
datos = datos.dropna(subset=columnas_historicas).copy()

# Reservar las últimas 90 fechas para validar
entrenamiento, validacion = separar_cronologicamente(datos)

X = preparar_variables(datos)
X_entrenamiento = X.loc[entrenamiento.index]
X_validacion = X.loc[validacion.index]

y_entrenamiento = entrenamiento["sales"]
y_validacion = validacion["sales"]

# Entrenar Random Forest con las fechas anteriores a la validación
modelo = crear_modelos()["Random Forest"]
print("Entrenando Random Forest para analizar la validación...")
modelo.fit(X_entrenamiento, y_entrenamiento)

predicciones = np.maximum(modelo.predict(X_validacion), 0)

# Comparar cada predicción con la venta real
detalle = validacion[["date", "store", "item", "sales"]].copy()
detalle = detalle.rename(columns={"sales": "venta_real"})
detalle["venta_predicha"] = predicciones
detalle["error"] = detalle["venta_predicha"] - detalle["venta_real"]
detalle["error_absoluto"] = detalle["error"].abs()
detalle["error_cuadratico"] = detalle["error"] ** 2

# Resumir el error de cada artículo en todas las tiendas y fechas
error_por_articulo = (
    detalle.groupby("item")
    .agg(
        MAE=("error_absoluto", "mean"),
        RMSE=("error_cuadratico", lambda valores: np.sqrt(valores.mean())),
        venta_real_promedio=("venta_real", "mean"),
        venta_predicha_promedio=("venta_predicha", "mean"),
    )
    .reset_index()
    .sort_values("MAE", ascending=False)
)

# Sumar ventas reales y predichas por día para poder graficarlas luego
ventas_diarias = (
    detalle.groupby("date")
    .agg(
        venta_real_total=("venta_real", "sum"),
        venta_predicha_total=("venta_predicha", "sum"),
    )
    .reset_index()
)

# Guardar tres archivos que se pueden abrir con Excel
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

archivo_detalle = CARPETA_RESULTADOS / "validacion_detalle_random_forest.csv"
archivo_articulos = CARPETA_RESULTADOS / "error_por_articulo.csv"
archivo_diario = CARPETA_RESULTADOS / "ventas_diarias_validacion.csv"

detalle.to_csv(archivo_detalle, index=False)
error_por_articulo.to_csv(archivo_articulos, index=False)
ventas_diarias.to_csv(archivo_diario, index=False)

# Métricas generales de este período
mae = mean_absolute_error(y_validacion, predicciones)
rmse = np.sqrt(mean_squared_error(y_validacion, predicciones))

print()
print(
    f"Período de validación: {validacion['date'].min():%Y-%m-%d} "
    f"a {validacion['date'].max():%Y-%m-%d}"
)
print(f"MAE general:  {mae:.4f} unidades")
print(f"RMSE general: {rmse:.4f} unidades")

print("\nCinco artículos con mayor MAE:")
print(error_por_articulo.head(5).to_string(
    index=False,
    formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "venta_real_promedio": "{:.2f}".format,
        "venta_predicha_promedio": "{:.2f}".format,
    },
))

print("\nArchivos guardados:")
print(archivo_detalle)
print(archivo_articulos)
print(archivo_diario)
print("\nEstos resultados corresponden a Kaggle, no a ventas reales de Kapaq.")