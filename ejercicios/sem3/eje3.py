import random

# lista de peliculas con genero, duracion y anio
peliculas = [
    {"titulo": "Dune: Parte Dos", "genero": "Ciencia Ficcion", "duracion": 166, "anio": 2024},
    {"titulo": "Interestelar", "genero": "Ciencia Ficcion", "duracion": 169, "anio": 2014},
    {"titulo": "Avengers: Endgame", "genero": "Accion", "duracion": 181, "anio": 2019},
    {"titulo": "John Wick 4", "genero": "Accion", "duracion": 169, "anio": 2023},
    {"titulo": "Superbad", "genero": "Comedia", "duracion": 113, "anio": 2007},
    {"titulo": "Que paso ayer", "genero": "Comedia", "duracion": 100, "anio": 2009},
    {"titulo": "Forrest Gump", "genero": "Drama", "duracion": 142, "anio": 1994},
    {"titulo": "Titanic", "genero": "Drama", "duracion": 194, "anio": 1997},
    {"titulo": "It", "genero": "Terror", "duracion": 135, "anio": 2017},
    {"titulo": "Un lugar en silencio", "genero": "Terror", "duracion": 90, "anio": 2018},
]

# sacar los generos que hay
generos = []
for p in peliculas:
    if p["genero"] not in generos:
        generos.append(p["genero"])

# preguntar genero preferido por opciones
print("Elige tu genero preferido:")
for i in range(len(generos)):
    print(i + 1, "-", generos[i])

op = int(input("Opcion (1-5): "))
genero_preferido = generos[op - 1]

# preguntar historial por opciones
print("Elige 3 peliculas que ya viste:")
for i in range(len(peliculas)):
    print(i + 1, "-", peliculas[i]["titulo"])

entrada = input("Opciones separadas por coma (ej: 2,3,5): ")
partes = entrada.split(",")

historial = []
for x in partes:
    num = int(x)
    historial.append(peliculas[num - 1]["titulo"])

# tabla Q vacia en 0
Q = {}
for g in generos:
    Q[g] = {}
    for p in peliculas:
        Q[g][p["titulo"]] = 0.0

alpha = 0.5     #taza de aprendizaje
gamma = 0.9
epsilon = 0.2    #exploracion

# promedio de duracion de lo que ya vio
suma = 0
cont = 0
for p in peliculas:
    if p["titulo"] in historial:
        suma = suma + p["duracion"]
        cont = cont + 1

if cont > 0:
    prom_duracion = suma / cont
else:
    prom_duracion = 120

# funcion para dar puntos a cada pelicula
def calcularPuntos(peli, genero_pref):
    puntos = 0

    # si es del genero que le gusta
    if peli["genero"] == genero_pref:
        puntos = puntos + 3

    # si es del genero que mas vio en el historial
    for vista in historial:
        for p in peliculas:
            if p["titulo"] == vista:
                if p["genero"] == peli["genero"]:
                    puntos = puntos + 1

    # si ya la vio se castiga
    if peli["titulo"] in historial:
        puntos = puntos - 10

    # si es reciente
    if peli["anio"] >= 2015:
        puntos = puntos + 1

    # si dura parecido a lo que ve
    if abs(peli["duracion"] - prom_duracion) <= 30:
        puntos = puntos + 1

    return puntos

for episodio in range(200):

    estado = genero_preferido

    # Explorar o utilizar lo aprendido
    if random.random() < epsilon:
        accion = random.choice(list(Q[estado].keys()))
    else:
        accion = max(Q[estado], key=Q[estado].get)

    # Buscar la pelicula elegida
    peli_elegida = None
    for p in peliculas:
        if p["titulo"] == accion:
            peli_elegida = p

    # Ejecutar la acción
    recompensa = calcularPuntos(peli_elegida, estado)

    # Si es recomendacion final, no hay acciones futuras
    mejor_valor_futuro = 0

    # Fórmula de Q-Learning
    Q[estado][accion] = Q[estado][accion] + alpha * (
        recompensa
        + gamma * mejor_valor_futuro
        - Q[estado][accion]
    )

# Mostrar la tabla Q aprendida
print("Genero preferido:", genero_preferido)
print("Historial:", historial)
print("Tabla Q aprendida:")

for titulo, valor in Q[genero_preferido].items():
    print(titulo, valor)

# Buscar la mejor no vista
mejor = None
mejor_valor = -1000
for titulo, valor in Q[genero_preferido].items():
    if titulo not in historial:
        if valor > mejor_valor:
            mejor_valor = valor
            mejor = titulo

print("Recomendacion final:", mejor)
