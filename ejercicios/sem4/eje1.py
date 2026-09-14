import numpy as np
from scipy.optimize import minimize

# Función a minimizar: f(x) = (x-3)^2 + (y-4)^2
def funcion(x):
    return (x[0] - 3)**2 + (x[1] - 4)**2

# Valor inicial
x0 = np.array([0, 0])

# Llamamos a la función de optimización
resultado = minimize(funcion, x0)

# Mostramos el resultado
print("Resultado de la optimización:", resultado.x)
