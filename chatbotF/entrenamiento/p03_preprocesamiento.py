# normalizacion de texto en espanol cuidando las negaciones
import re
import unicodedata

# expresion regular para tokens de letras en espanol sin descartar palabras cortas
PATRON_TOKEN = r"(?u)\b[a-zñ]+\b"


def normalizar_texto(texto):
    # pasa a minusculas y elimina tildes conservando la letra enie
    if not isinstance(texto, str):
        return ""

    limpio = texto.lower().strip()
    limpio = limpio.replace("ñ", "__enie__")
    limpio = "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", limpio)
        if unicodedata.category(caracter) != "Mn"
    )
    limpio = limpio.replace("__enie__", "ñ")

    # reduce alargamientos expresivos de tres o mas letras repetidas como holaaaa
    limpio = re.sub(r"([a-zñ])\1{2,}", r"\1", limpio)

    # reemplaza signos y numeros por espacios para no pegar palabras distintas
    limpio = re.sub(r"[^a-zñ\s]", " ", limpio)
    limpio = re.sub(r"\s+", " ", limpio).strip()
    return limpio


def normalizar_lote(textos):
    # aplica la misma limpieza determinista a una lista completa de mensajes
    return [normalizar_texto(texto) for texto in textos]
