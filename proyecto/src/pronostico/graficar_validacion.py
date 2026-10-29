from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CARPETA_RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"

ARCHIVO_DIARIO = CARPETA_RESULTADOS / "ventas_diarias_validacion.csv"
ARCHIVO_ARTICULOS = CARPETA_RESULTADOS / "error_por_articulo.csv"


# Gráfico 1: total real y predicho para cada día de validación
diario = pd.read_csv(ARCHIVO_DIARIO, parse_dates=["date"])

plt.figure(figsize=(12, 5))
plt.plot(
    diario["date"],
    diario["venta_real_total"],
    label="Ventas reales",
    linewidth=1.5,
)
plt.plot(
    diario["date"],
    diario["venta_predicha_total"],
    label="Ventas predichas",
    linewidth=1.5,
)
plt.title("Kaggle: ventas diarias reales y predichas en validación")
plt.xlabel("Fecha")
plt.ylabel("Unidades vendidas (total)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

grafico_diario = CARPETA_RESULTADOS / "ventas_reales_vs_predichas.png"
plt.savefig(grafico_diario, dpi=150)
plt.show()


# Gráfico 2: diez artículos con mayor MAE
articulos = pd.read_csv(ARCHIVO_ARTICULOS)
peores = (
    articulos.sort_values("MAE", ascending=False)
    .head(10)
    .sort_values("MAE")
)

plt.figure(figsize=(9, 5))
plt.barh(
    peores["item"].astype(str),
    peores["MAE"],
)
plt.title("Kaggle: diez artículos con mayor MAE")
plt.xlabel("MAE (unidades)")
plt.ylabel("Artículo")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

grafico_articulos = CARPETA_RESULTADOS / "mae_por_articulo.png"
plt.savefig(grafico_articulos, dpi=150)
plt.show()

print("Gráficos guardados:")
print(grafico_diario)
print(grafico_articulos)