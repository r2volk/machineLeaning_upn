import pandas as pd

from datos import cargar_train


def separar_cronologicamente(datos, dias_validacion=90):
    """Reserva los últimos días para validar; entrena solo con fechas anteriores."""
    if dias_validacion <= 0:
        raise ValueError("dias_validacion debe ser mayor que cero")

    fecha_final = datos["date"].max()
    fecha_inicio_validacion = (
        fecha_final - pd.Timedelta(days=dias_validacion - 1)
    )

    entrenamiento = datos[datos["date"] < fecha_inicio_validacion].copy()
    validacion = datos[datos["date"] >= fecha_inicio_validacion].copy()

    if entrenamiento.empty or validacion.empty:
        raise ValueError("La separación dejó entrenamiento o validación vacíos")

    return entrenamiento, validacion


if __name__ == "__main__":
    datos = cargar_train()
    entrenamiento, validacion = separar_cronologicamente(datos)

    print("SEPARACIÓN CRONOLÓGICA")
    print(
        f"Entrenamiento: {entrenamiento['date'].min():%Y-%m-%d} a "
        f"{entrenamiento['date'].max():%Y-%m-%d}"
    )
    print(
        f"Validación:    {validacion['date'].min():%Y-%m-%d} a "
        f"{validacion['date'].max():%Y-%m-%d}"
    )
    print(f"Filas para entrenar: {len(entrenamiento):,}")
    print(f"Filas para validar:  {len(validacion):,}")