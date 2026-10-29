from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datos import cargar_train
from modelos import COLUMNAS_ENTRADA, crear_modelos, preparar_variables
from variables import agregar_variables_historicas


DIAS_POR_BLOQUE = 90
NUM_BLOQUES = 3

CARPETA_RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"


def calcular_errores(ventas_reales, predicciones):
    return {
        "MAE": mean_absolute_error(ventas_reales, predicciones),
        "RMSE": np.sqrt(
            mean_squared_error(ventas_reales, predicciones)
        ),
    }


def crear_historial(datos_anteriores):
    """Crea el historial real inicial por tienda y artículo."""
    historial = {}

    for (tienda, articulo), grupo in datos_anteriores.groupby(
        ["store", "item"]
    ):
        historial[(int(tienda), int(articulo))] = (
            grupo.sort_values("date")["sales"].astype(float).tolist()
        )

    return historial


# Cargar datos y construir variables que solo usan fechas pasadas
datos_originales = cargar_train()
datos_con_variables = agregar_variables_historicas(datos_originales)

datos_modelo = datos_con_variables.dropna(
    subset=["lag_7", "lag_14", "media_7_dias", "media_28_dias"]
).copy()

X_completo = preparar_variables(datos_modelo)
fecha_base = datos_modelo["date"].min()
fecha_final = datos_originales["date"].max()

# Últimos 270 días, separados en 3 bloques consecutivos de 90 días
dias_totales = DIAS_POR_BLOQUE * NUM_BLOQUES
primer_inicio = fecha_final - pd.Timedelta(days=dias_totales - 1)

resultados = []

for numero_bloque in range(NUM_BLOQUES):
    inicio = primer_inicio + pd.Timedelta(
        days=numero_bloque * DIAS_POR_BLOQUE
    )
    fin = inicio + pd.Timedelta(days=DIAS_POR_BLOQUE - 1)

    entrenamiento = datos_modelo[datos_modelo["date"] < inicio]
    validacion = datos_originales[
        (datos_originales["date"] >= inicio)
        & (datos_originales["date"] <= fin)
    ].copy()

    X_entrenamiento = X_completo.loc[entrenamiento.index]
    y_entrenamiento = entrenamiento["sales"]

    modelo = crear_modelos()["Random Forest"]

    print(
        f"Bloque {numero_bloque + 1}/{NUM_BLOQUES}: "
        f"{inicio:%Y-%m-%d} a {fin:%Y-%m-%d}"
    )
    print("  Entrenando Random Forest...")
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # Ambos métodos comienzan con las ventas reales previas al bloque.
    fechas_previas = datos_originales[
        datos_originales["date"] < inicio
    ]
    historial_rf = crear_historial(fechas_previas)
    historial_referencia = crear_historial(fechas_previas)

    reales_bloque = []
    pred_rf_bloque = []
    pred_referencia_bloque = []

    # Se predice día por día. Después de cada día, cada método agrega
    # sus propias predicciones al historial para pronosticar el siguiente.
    for fecha, ventas_del_dia in validacion.groupby("date", sort=True):
        ventas_del_dia = ventas_del_dia.sort_values(
            ["store", "item"]
        )

        filas_features_rf = []
        predicciones_referencia_dia = []

        for fila in ventas_del_dia.itertuples():
            clave = (int(fila.store), int(fila.item))
            historial_rf_producto = historial_rf[clave]
            historial_ref_producto = historial_referencia[clave]

            filas_features_rf.append({
                "store": int(fila.store),
                "item": int(fila.item),
                "day_index": (fecha - fecha_base).days,
                "day_of_week": fecha.dayofweek,
                "month": fecha.month,
                "lag_7": historial_rf_producto[-7],
                "lag_14": historial_rf_producto[-14],
                "media_7_dias": np.mean(historial_rf_producto[-7:]),
                "media_28_dias": np.mean(historial_rf_producto[-28:]),
            })

            # Referencia semanal recursiva: repite lo estimado hace 7 días.
            predicciones_referencia_dia.append(
                historial_ref_producto[-7]
            )

        X_dia = pd.DataFrame(
            filas_features_rf,
            columns=COLUMNAS_ENTRADA,
        )

        predicciones_rf_dia = np.maximum(
            modelo.predict(X_dia),
            0,
        )
        predicciones_referencia_dia = np.maximum(
            np.array(predicciones_referencia_dia),
            0,
        )

        reales_dia = ventas_del_dia["sales"].to_numpy()
        reales_bloque.extend(reales_dia)
        pred_rf_bloque.extend(predicciones_rf_dia)
        pred_referencia_bloque.extend(predicciones_referencia_dia)

        # Se agregan las predicciones, no las ventas reales, al historial.
        for fila, pred_rf, pred_ref in zip(
            ventas_del_dia.itertuples(),
            predicciones_rf_dia,
            predicciones_referencia_dia,
        ):
            clave = (int(fila.store), int(fila.item))
            historial_rf[clave].append(float(pred_rf))
            historial_referencia[clave].append(float(pred_ref))

    # Registrar resultados del bloque
    for nombre, predicciones in [
        ("Referencia semanal recursiva", pred_referencia_bloque),
        ("Random Forest recursivo", pred_rf_bloque),
    ]:
        errores = calcular_errores(reales_bloque, predicciones)

        resultados.append({
            "Bloque": numero_bloque + 1,
            "Inicio": inicio.date(),
            "Fin": fin.date(),
            "Método": nombre,
            **errores,
        })


# Mostrar resultados por bloque y promedio
tabla = pd.DataFrame(resultados)

print("\nRESULTADOS POR BLOQUE")
print(tabla.to_string(
    index=False,
    formatters={
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
    },
))

print("\nPROMEDIO DE LOS TRES BLOQUES")
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

CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
archivo_salida = (
    CARPETA_RESULTADOS / "validacion_recursiva_random_forest.csv"
)
tabla.to_csv(archivo_salida, index=False)

print(f"\nResultados guardados en: {archivo_salida}")
print("Los datos son de Kaggle, no ventas reales de Kapaq.")