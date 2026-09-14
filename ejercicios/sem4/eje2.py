import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Datos simulados (X = metros cuadrados, y = precio)
X = np.array([[50], [60], [70], [80], [100]])
y = np.array([100, 120, 140, 160, 200])

# Creando el modelo de regresión lineal
modelo_lineal = LinearRegression()
modelo_lineal.fit(X, y)

# Predicción
y_pred = modelo_lineal.predict(X)

# Visualización
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred, color='red', label='Regresión Lineal')
plt.xlabel("Metros Cuadrados")
plt.ylabel("Precio")
plt.legend()
plt.show()
