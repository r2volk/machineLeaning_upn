from pathlib import Path

import pandas as pd

from datos import cargar_train
from evaluacion import separar_cronologicamente
from modelos import crear_modelos, preparar_variables
from variables import agregar_variables_historicas


CARPETA_RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"

datos = cargar_train()
datos = agregar_variables_historicas(datos)
datos = datos.dropna(
    subset=["lag_7", "lag_14", "media_7_dias", "media_28_dias"]
).copy()

entrenamiento, validacion = separar_cronologicamente(datos)

X = preparar_variables(datos)
X_entrenamiento = X.loc[entrenamiento.index]
y_entrenamiento = entrenamiento["sales"]

modelo = crear_modelos()["Random Forest"]

print("Entrenando Random Forest para analizar sus variables...")
modelo.fit(X_entrenamiento, y_entrenamiento)

# Recuperar el bosque y los nombres de las variables tras el preprocesamiento
preparacion = modelo.named_steps["columntransformer"]
bosque = modelo.named_steps["randomforestregressor"]

nombres = preparacion.get_feature_names_out()
importancias = bosque.feature_importances_

tabla = pd.DataFrame({
    "variable": nombres,
    "importancia": importancias,
}).sort_values("importancia", ascending=False)

CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
archivo_salida = CARPETA_RESULTADOS / "importancia_variables_random_forest.csv"
tabla.to_csv(archivo_salida, index=False)

print("\nQuince variables con mayor importancia:")
print(tabla.head(15).to_string(
    index=False,
    formatters={"importancia": "{:.4f}".format},
))
print(f"\nTabla completa guardada en: {archivo_salida}")