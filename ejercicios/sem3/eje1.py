import random

Q = {
    "A":{
        "avanzar": 0.0,
        "esperar":0.0
    },
    "B":{
        "avanzar": 0.0,
        "regresar":0.0
    }
}

alpha = 0.5     #taza de aprendizaje
gamma = 0.9     
epsilon = 0.2    #exploracion del robot

def ejecutarAccion(estado,accion):
    if estado == "A" and accion == "avanzar":
        return "B",0
    
    if estado == "A" and accion == "esperar":
        return "A",-1

    if estado == "B" and accion == "avanzar":
        return "C",10

    if estado == "B" and accion == "regresar":
        return "A",0

for episodio in range(100):

    estado = "A"

    while estado != "C":

        # Explorar o utilizar lo aprendido
        if random.random() < epsilon:
            accion = random.choice(list(Q[estado].keys()))
        else:
            accion = max(Q[estado], key=Q[estado].get)

        # Ejecutar la acción
        nuevoEstado, recompensa = ejecutarAccion(estado, accion)

        # Si llegó a la meta, no hay acciones futuras
        if nuevoEstado == "C":
            mejor_valor_futuro = 0
        else:
            mejor_valor_futuro = max(Q[nuevoEstado].values())

        # Fórmula de Q-Learning
        Q[estado][accion] = Q[estado][accion] + alpha * (
            recompensa
            + gamma * mejor_valor_futuro
            - Q[estado][accion]
        )

        estado = nuevoEstado


# Mostrar la tabla Q aprendida
print("Tabla Q aprendida:")

for estado, acciones in Q.items():
    print(estado, acciones)