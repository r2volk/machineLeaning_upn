import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Datos (no lineales)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # relación cuadrática (y = x^2)

# Transformación polinómica (grado 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Modelo
modelo = LinearRegression()
modelo.fit(X_poly, y)

# Predicción
y_pred = modelo.predict(X_poly)

# Gráfica
plt.scatter(X, y, label="Datos reales")
plt.plot(X, y_pred, label="Regresión No Lineal (Polinómica)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
