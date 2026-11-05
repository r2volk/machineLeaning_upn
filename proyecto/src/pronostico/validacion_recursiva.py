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


def crear_historial(datos_anteriores):
    """Crea el historial inicial de ventas por tienda y artículo."""
    historial = {}

    for (tienda, articulo), grupo in datos_anteriores.groupby(
        ["store", "item"]
    ):
        historial[(int(tienda), int(articulo))] = (
            grupo.sort_values("date")["sales"].astype(float).tolist()
        )

    return historial


def pronosticar_recursivamente(
    datos_validacion,
    historial_inicial,
    pares,
    fecha_base,
    modelo=None,
):
    """
    Pronostica día a día. Usa las predicciones de cada día como historial
    para el siguiente, nunca las ventas reales del período de validación.
    Si modelo=None, aplica la referencia de ventas de hace 7 días.
    """
    historial = {
        clave: valores.copy()
        for clave, valores in historial_inicial.items()
    }

    ventas_reales = []
    predicciones = []

    for fecha, ventas_dia in datos_validacion.groupby("date", sort=True):
        ventas_dia = ventas_dia.sort_values(["store", "item"])

        filas = []

        for tienda, articulo in pares:
            ventas_previas = historial[(tienda, articulo)]

            filas.append({
                "store": tienda,
                "item": articulo,
                "day_index": (fecha - fecha_base).days,
                "day_of_week": fecha.dayofweek,
                "month": fecha.month,
                "lag_7": ventas_previas[-7],
                "lag_14": ventas_previas[-14],
                "media_7_dias": np.mean(ventas_previas[-7:]),
                "media_28_dias": np.mean(ventas_previas[-28:]),
            })

        X_dia = pd.DataFrame(filas, columns=COLUMNAS_ENTRADA)

        if modelo is None:
            predicciones_dia = X_dia["lag_7"].to_numpy()
        else:
            predicciones_dia = modelo.predict(X_dia)

        predicciones_dia = np.maximum(predicciones_dia, 0)
        reales_dia = ventas_dia["sales"].to_numpy()

        ventas_reales.extend(reales_dia)
        predicciones.extend(predicciones_dia)

        # Guardar en el historial las predicciones, no las ventas reales.
        for (tienda, articulo), prediccion in zip(pares, predicciones_dia):
            historial[(tienda, articulo)].append(float(prediccion))

    return np.array(ventas_reales), np.array(predicciones)


# Cargar Kaggle y construir variables históricas.
datos_originales = cargar_train()
datos = agregar_variables_historicas(datos_originales)

datos_modelo = datos.dropna(
    subset=["lag_7", "lag_14", "media_7_dias", "media_28_dias"]
).copy()

# Preparar las entradas una vez para conservar la misma escala de fechas.
X_completo = preparar_variables(datos_modelo)
fecha_base = datos_modelo["date"].min()
fecha_final = datos_originales["date"].max()

pares = sorted({
    (int(fila.store), int(fila.item))
    for fila in datos_originales[
        ["store", "item"]
    ].drop_duplicates().itertuples(index=False)
})

# Últimos 270 días: tres bloques cronológicos de 90 días.
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

    if entrenamiento.empty or validacion.empty:
        raise ValueError(f"El bloque {numero_bloque + 1} está vacío.")

    X_entrenamiento = X_completo.loc[entrenamiento.index]
    y_entrenamiento = entrenamiento["sales"]

    datos_anteriores = datos_originales[
        datos_originales["date"] < inicio
    ]
    historial_inicial = crear_historial(datos_anteriores)

    if any(len(historial_inicial[par]) < 28 for par in pares):
        raise ValueError("Faltan 28 días de historial para algún producto.")

    print(
        f"\nBloque {numero_bloque + 1}/{NUM_BLOQUES}: "
        f"{inicio:%Y-%m-%d} a {fin:%Y-%m-%d}"
    )

    # Primero probamos la referencia semanal.
    reales, predichas = pronosticar_recursivamente(
        datos_validacion=validacion,
        historial_inicial=historial_inicial,
        pares=pares,
        fecha_base=fecha_base,
        modelo=None,
    )

    resultados.append({
        "Bloque": numero_bloque + 1,
        "Inicio": inicio.date(),
        "Fin": fin.date(),
        "Método": "Referencia semanal recursiva",
        "MAE": mean_absolute_error(reales, predichas),
        "RMSE": np.sqrt(mean_squared_error(reales, predichas)),
    })

    # Entrenar y evaluar regresión lineal y Random Forest.
    for nombre, modelo in crear_modelos().items():
        print(f"  Entrenando {nombre}...")
        modelo.fit(X_entrenamiento, y_entrenamiento)

        reales, predichas = pronosticar_recursivamente(
            datos_validacion=validacion,
            historial_inicial=historial_inicial,
            pares=pares,
            fecha_base=fecha_base,
            modelo=modelo,
        )

        resultados.append({
            "Bloque": numero_bloque + 1,
            "Inicio": inicio.date(),
            "Fin": fin.date(),
            "Método": f"{nombre} recursivo",
            "MAE": mean_absolute_error(reales, predichas),
            "RMSE": np.sqrt(mean_squared_error(reales, predichas)),
        })


tabla = pd.DataFrame(resultados)

print("\nRESULTADOS RECURSIVOS POR BLOQUE")
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
archivo_salida = CARPETA_RESULTADOS / "comparacion_recursiva_90_dias.csv"
tabla.to_csv(archivo_salida, index=False)

print(f"\nResultados guardados en: {archivo_salida}")
print("Los datos son de Kaggle, no ventas reales de Kapaq.")