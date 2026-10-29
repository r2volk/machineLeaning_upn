from datos import cargar_train


def agregar_variables_historicas(datos):
    """Añade ventas pasadas y promedios calculados sin usar el día objetivo."""
    datos = datos.sort_values(
        ["store", "item", "date"]
    ).copy()

    ventas_por_producto = datos.groupby(
        ["store", "item"],
        sort=False,
    )["sales"]

    # Venta del mismo artículo y tienda hace 7 y 14 días
    datos["lag_7"] = ventas_por_producto.shift(7)
    datos["lag_14"] = ventas_por_producto.shift(14)

    # Promedios de los días anteriores.
    # shift(1) excluye las ventas del propio día que se quiere predecir.
    datos["media_7_dias"] = ventas_por_producto.transform(
        lambda ventas: ventas.shift(1).rolling(
            window=7,
            min_periods=7,
        ).mean()
    )

    datos["media_28_dias"] = ventas_por_producto.transform(
        lambda ventas: ventas.shift(1).rolling(
            window=28,
            min_periods=28,
        ).mean()
    )

    return datos


if __name__ == "__main__":
    datos = cargar_train()
    datos = agregar_variables_historicas(datos)

    columnas = [
        "date",
        "store",
        "item",
        "sales",
        "lag_7",
        "lag_14",
        "media_7_dias",
        "media_28_dias",
    ]

    # Mostramos las primeras filas que ya tienen historial suficiente.
    ejemplo = datos.dropna(subset=columnas[4:]).head(5)

    print("EJEMPLO DE VARIABLES HISTÓRICAS")
    print(ejemplo[columnas].to_string(index=False))