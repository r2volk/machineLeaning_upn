# Definimos el agente con su comportamiento básico
def agente_aspiradora(percepcion):
    if percepcion == "sucio":
        return "limpiar"
    elif percepcion == "obstaculo":
        return "girar"
    elif percepcion == "limpio":
        return "avanzar"
    else:
        return "esperar"  # Si no hay información, el agente espera


# Simulamos el entorno
entorno = ["sucio", "limpio", "obstaculo", "sucio", "bateria baja"]

# El agente realiza las acciones en base a la percepción
acciones = [agente_aspiradora(p) for p in entorno]

# Imprimimos los resultados
for i, (p, a) in enumerate(zip(entorno, acciones)):
    print(f"Paso {i+1}: Percepción = {p} -> Acción = {a}")
