import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datos import cargar_train
from evaluacion import separar_cronologicamente
from modelos import preparar_variables, crear_modelos
from variables import agregar_variables_historicas


datos = cargar_train()

# Agregar ventas previas y promedios históricos.
datos = agregar_variables_historicas(datos)

# Eliminar el inicio de cada serie, que no tiene historial suficiente.
columnas_historicas = [
    "lag_7",
    "lag_14",
    "media_7_dias",
    "media_28_dias",
]
datos = datos.dropna(subset=columnas_historicas).copy()

# Separar por fecha: pasado para entrenar, fechas posteriores para evaluar.
entrenamiento, validacion = separar_cronologicamente(datos)

# Crear las entradas de los modelos y alinearlas con cada partición.
X = preparar_variables(datos)
X_entrenamiento = X.loc[entrenamiento.index]
X_validacion = X.loc[validacion.index]

y_entrenamiento = entrenamiento["sales"]
y_validacion = validacion["sales"]

resultados = []

# Referencia semanal: repetir las ventas de hace 7 días.
predicciones_referencia = validacion["lag_7"].to_numpy()

resultados.append({
    "Método": "Referencia: ventas de hace 7 días",
    "MAE": mean_absolute_error(y_validacion, predicciones_referencia),
    "RMSE": np.sqrt(
        mean_squared_error(y_validacion, predicciones_referencia)
    ),
})

# Entrenar y evaluar cada modelo.
for nombre, modelo in crear_modelos().items():
    print(f"Entrenando {nombre}...")

    modelo.fit(X_entrenamiento, y_entrenamiento)
    predicciones = np.maximum(modelo.predict(X_validacion), 0)

    resultados.append({
        "Método": nombre,
        "MAE": mean_absolute_error(y_validacion, predicciones),
        "RMSE": np.sqrt(
            mean_squared_error(y_validacion, predicciones)
        ),
    })

print()
print(
    f"Entrenamiento: hasta "
    f"{entrenamiento['date'].max():%Y-%m-%d}"
)
print(
    f"Validación:    {validacion['date'].min():%Y-%m-%d} "
    f"a {validacion['date'].max():%Y-%m-%d}"
)
print(f"Filas para entrenar: {len(entrenamiento):,}")
print(f"Filas para validar:  {len(validacion):,}")
print()
print(pd.DataFrame(resultados).to_string(
    index=False,
    formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
    },
))
print()
print("Los datos son de Kaggle; no son ventas reales de Kapaq.")