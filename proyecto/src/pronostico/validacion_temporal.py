import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datos import cargar_train
from modelos import crear_modelos, preparar_variables
from variables import agregar_variables_historicas


DIAS_POR_BLOQUE = 90
NUM_BLOQUES = 3


def calcular_errores(real, predicho):
    return {
        "MAE": mean_absolute_error(real, predicho),
        "RMSE": np.sqrt(mean_squared_error(real, predicho)),
    }


datos = cargar_train()
datos = agregar_variables_historicas(datos)

# Quitamos las fechas que aún no tienen suficiente historial.
datos = datos.dropna(
    subset=["lag_7", "lag_14", "media_7_dias", "media_28_dias"]
).copy()

X = preparar_variables(datos)
fecha_final = datos["date"].max()

# Tomamos los últimos 270 días y los dividimos en tres bloques consecutivos
# de 90 días. Cada bloque se evalúa usando solo fechas anteriores para entrenar.
dias_totales = DIAS_POR_BLOQUE * NUM_BLOQUES
primer_inicio = fecha_final - pd.Timedelta(days=dias_totales - 1)

resultados = []

for numero_bloque in range(NUM_BLOQUES):
    inicio = primer_inicio + pd.Timedelta(
        days=numero_bloque * DIAS_POR_BLOQUE
    )
    fin = inicio + pd.Timedelta(days=DIAS_POR_BLOQUE - 1)

    entrenamiento = datos[datos["date"] < inicio]
    validacion = datos[
        (datos["date"] >= inicio) & (datos["date"] <= fin)
    ]

    X_entrenamiento = X.loc[entrenamiento.index]
    X_validacion = X.loc[validacion.index]
    y_entrenamiento = entrenamiento["sales"]
    y_validacion = validacion["sales"]

    print(f"Evaluando bloque {numero_bloque + 1}/{NUM_BLOQUES}: "
          f"{inicio:%Y-%m-%d} a {fin:%Y-%m-%d}")

    # Referencia: repetir las ventas de hace siete días.
    errores = calcular_errores(
        y_validacion,
        validacion["lag_7"],
    )
    resultados.append({
        "Bloque": numero_bloque + 1,
        "Método": "Referencia semanal",
        **errores,
    })

    # Entrenar y evaluar regresión lineal y Random Forest.
    for nombre, modelo in crear_modelos().items():
        modelo.fit(X_entrenamiento, y_entrenamiento)
        predicciones = np.maximum(
            modelo.predict(X_validacion),
            0,
        )

        errores = calcular_errores(y_validacion, predicciones)
        resultados.append({
            "Bloque": numero_bloque + 1,
            "Método": nombre,
            **errores,
        })


tabla = pd.DataFrame(resultados)

print("\nRESULTADOS POR PERÍODO")
print(tabla.to_string(
    index=False,
    formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
    },
))

print("\nPROMEDIO DE LOS TRES PERÍODOS")
promedios = (
    tabla.groupby("Método")[["MAE", "RMSE"]]
    .mean()
    .reset_index()
)
print(promedios.to_string(
    index=False,
    formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
    },
))

print("\nLos resultados son del dataset de Kaggle, no de ventas reales de Kapaq.")