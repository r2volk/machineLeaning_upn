import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datos import cargar_train
from evaluacion import separar_cronologicamente


def agregar_ventas_semana_anterior(datos):
    """Añade las ventas del mismo artículo y tienda hace 7 días."""
    datos = datos.sort_values(
        ["store", "item", "date"]
    ).copy()

    datos["ventas_hace_7_dias"] = (
        datos.groupby(["store", "item"])["sales"].shift(7)
    )

    return datos


if __name__ == "__main__":
    datos = cargar_train()
    datos = agregar_ventas_semana_anterior(datos)

    # Las primeras 7 observaciones de cada artículo/tienda aún no
    # tienen una semana anterior con la que hacer la referencia.
    datos = datos.dropna(subset=["ventas_hace_7_dias"]).copy()

    entrenamiento, validacion = separar_cronologicamente(datos)

    ventas_reales = validacion["sales"]
    predicciones = validacion["ventas_hace_7_dias"]

    mae = mean_absolute_error(ventas_reales, predicciones)
    rmse = np.sqrt(mean_squared_error(ventas_reales, predicciones))

    print("REFERENCIA: ventas de hace 7 días")
    print(f"Filas de entrenamiento: {len(entrenamiento):,}")
    print(f"Filas de validación:    {len(validacion):,}")
    print(f"MAE:  {mae:.4f} unidades")
    print(f"RMSE: {rmse:.4f} unidades")