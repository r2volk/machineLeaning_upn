# normalizacion de texto en espanol cuidando las negaciones
import re
import unicodedata

# patron que incluye letras en espanol y prefijos neg_ usados por el clasificador
PATRON_TOKEN_MODELO = r"(?u)\b[a-zñ_]+\b"

# palabras que activan el marcado de negacion y palabras donde se corta el alcance
NEGADORES = {"no", "nunca", "jamas", "tampoco", "ni"}
CORTES_NEGACION = {"pero", "aunque", "sino", "solo", "porque", "que", "y", "sin"}
VENTANA_NEGACION = 2


def normalizar_texto(texto):
    # limpia una frase dejandola en minusculas, sin tildes y conservando la letra enie
    # esta funcion se usa intacta para las reglas de seguridad y control de fugas
    if not isinstance(texto, str):
        return ""

    # guarda temporalmente la enie para que no pierda su virgulilla al quitar tildes
    limpio = texto.lower().strip()
    limpio = limpio.replace("ñ", "__enie__")
    # separa letras y tildes con nfd y descarta solo las marcas de tilde
    limpio = "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", limpio)
        if unicodedata.category(caracter) != "Mn"
    )
    limpio = limpio.replace("__enie__", "ñ")

    # acorta letras repetidas tres o mas veces como holaaaa sin romper perro o calle
    limpio = re.sub(r"([a-zñ])\1{2,}", r"\1", limpio)

    # cambia signos y numeros por espacios y deja un solo espacio entre palabras
    limpio = re.sub(r"[^a-zñ\s]", " ", limpio)
    limpio = re.sub(r"\s+", " ", limpio).strip()
    return limpio


def preparar_texto_modelo(texto, ventana=VENTANA_NEGACION):
    # prepara un mensaje para el modelo de ml marcando negaciones con el prefijo neg_
    # separa por signos de puntuacion para que un no no afecte a otra oracion
    if not isinstance(texto, str):
        return ""

    clausulas = re.split(r"[,.;:!?¡¿]+", texto)
    tokens_salida = []
    for clausula in clausulas:
        tokens = normalizar_texto(clausula).split()
        pendientes = 0
        for token in tokens:
            if token in NEGADORES:
                # conserva el negador y activa el marcado para las siguientes palabras
                tokens_salida.append(token)
                pendientes = ventana
            elif pendientes > 0 and token in CORTES_NEGACION:
                # detiene el alcance si aparece una conjuncion como pero o solo
                tokens_salida.append(token)
                pendientes = 0
            elif pendientes > 0:
                # agrega neg_ para distinguir por ejemplo triste de neg_triste
                tokens_salida.append("neg_" + token)
                pendientes -= 1
            else:
                tokens_salida.append(token)
    return " ".join(tokens_salida)


def preparar_lote_modelo(textos):
    # aplica preparar_texto_modelo a una lista completa de frases
    return [preparar_texto_modelo(texto) for texto in textos]
