"""
===========================================================
ACTIVIDADES QUE DEBE DESARROLLAR EL ESTUDIANTE
===========================================================

1. Ejecutar el código base y analizar la predicción obtenida.

2. Ampliar el dataset agregando al menos 5 nuevos registros
   en X e y.

3. Modificar el parámetro test_size (por ejemplo: 0.2 -> 0.3)
   y comparar los resultados.

4. Cambiar el valor de n_estimators (10, 100, 200) y analizar
   el impacto en el modelo.

5. Agregar una nueva variable predictora al arreglo X.

6. Volver a ejecutar el modelo después de cada modificación.

7. Comparar el Error Cuadrático Medio (MSE) antes y después
   de los cambios.

===========================================================
"""


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Datos simulados
X = np.array([[50, 2], [60, 3], [70, 4], [80, 5], [100, 7]])
y = np.array([200, 220, 240, 260, 300])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar el modelo
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo_rf.predict(X_test)

# Mostrar predicción y calcular el error cuadrático medio
print("Predicción:", y_pred)
print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred):.2f}")
