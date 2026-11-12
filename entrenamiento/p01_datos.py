# importa herramientas para leer y escribir tablas de texto
import csv

# importa el generador para barajar grupos con semilla fija
import random

# importa las busquedas por patrones de texto
import re

# importa las herramientas de normalizacion de caracteres
import unicodedata

# importa el contador de elementos por clase
from collections import Counter

# importa las rutas constantes y utilidades compartidas
from .p00_configuracion import (
    CLASES,
    DATOS,
    SALIDAS,
    SEMILLA,
    guardar_json,
    huella,
    leer_json,
)


# limpia el texto sin aprender parametros
def normalizar(texto):
    # unifica caracteres equivalentes y convierte a minusculas
    texto = unicodedata.normalize("NFKC", texto).casefold()
    # elimina las marcas de acento y convierte la letra enie en ene
    texto = "".join(
        c
        for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    # conserva las letras y une las palabras con espacios
    return " ".join(re.findall(r"[a-z]+", texto))


# convierte escenarios en frases con etiqueta y grupo
def leer_escenarios(ruta=DATOS / "escenarios.txt"):
    # prepara las filas la clase actual y el contador de escenarios
    filas, etiqueta, contador = [], None, Counter()
    # lee el archivo y recorre sus lineas
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        # detecta lineas vacias o comentarios del corpus
        if not linea.strip() or linea.startswith("#"):
            # omite la linea y sigue con la siguiente
            continue
        # detecta un encabezado de clase
        if linea.startswith("["):
            # retira los corchetes para obtener la etiqueta
            etiqueta = linea.strip("[]")
            # comprueba que la etiqueta este permitida
            if etiqueta not in CLASES:
                # detiene la lectura si la clase es desconocida
                raise ValueError("Etiqueta desconocida")
            # pasa a la siguiente linea despues del encabezado
            continue
        # separa las dos frases y retira los espacios sobrantes
        partes = [p.strip() for p in linea.split("|")]
        # comprueba que haya una clase y dos frases completas
        if etiqueta is None or len(partes) != 2 or not all(partes):
            # rechaza un escenario con formato incorrecto
            raise ValueError("Cada escenario debe tener dos paráfrasis y una clase")
        # aumenta el numero de escenarios de esta clase
        contador[etiqueta] += 1
        # crea un identificador de grupo con ceros a la izquierda
        grupo = f"{etiqueta}_{contador[etiqueta]:03d}"
        # recorre las dos frases del mismo escenario
        for numero, texto in enumerate(partes, 1):
            # guarda cada frase con su clase grupo y origen
            filas.append(
                dict(
                    # identifica la frase dentro de su escenario
                    id=f"{grupo}_{numero}",
                    # conserva juntas las frases del mismo escenario
                    grupo=grupo,
                    # guarda la frase original
                    texto=texto,
                    # guarda la clase asignada o la salida visible
                    etiqueta=etiqueta,
                    # declara como se redactaron los datos
                    origen="sintetico_asistido_IA",
                )
            )
    # normaliza las frases antes de buscar duplicados
    textos = [normalizar(f["texto"]) for f in filas]
    # compara la cantidad de frases con la de textos unicos
    if len(textos) != len(set(textos)):
        # rechaza los textos que coinciden despues de normalizar
        raise ValueError("Hay textos normalizados duplicados; revisar antes de dividir")
    # devuelve todas las frases estructuradas
    return filas


# crea o verifica la division por escenarios completos
def preparar():
    # ubica el manifiesto con la division y las huellas
    manifiesto = DATOS / "manifiesto.json"
    # comprueba si ya existe una division congelada
    if manifiesto.exists():
        # verifica que los datos existentes no hayan cambiado
        verificar_datos()
        # reutiliza el manifiesto sin cambiar las particiones
        return leer_json(manifiesto)
    # comprueba si el test ya fue reservado
    if (SALIDAS / "test_reservado.json").exists():
        # impide crear otra division despues de reservar el test
        raise RuntimeError("El test ya está sellado; no se puede cambiar la partición")
    # lee y valida los escenarios originales
    filas = leer_escenarios()
    # prepara el mapa de grupos a particiones
    asignacion = {}
    # crea un generador aleatorio con semilla fija
    rng = random.Random(SEMILLA)
    # divide los grupos de cada clase por separado
    for clase in CLASES:
        # reune y ordena los grupos de la clase actual
        grupos = sorted({f["grupo"] for f in filas if f["etiqueta"] == clase})
        # baraja los grupos con el generador local
        rng.shuffle(grupos)
        # cuenta los grupos disponibles de esta clase
        n = len(grupos)
        # recorre los grupos ya barajados con su posicion
        for i, grupo in enumerate(grupos):
            # asigna cada grupo a entrenamiento validacion o test
            asignacion[grupo] = (
                "train"
                if i < int(0.6 * n)
                else "validacion" if i < int(0.8 * n) else "test"
            )
    # recorre las frases para completar su particion
    for fila in filas:
        # asigna a cada frase la particion de su grupo
        fila["particion"] = asignacion[fila["grupo"]]
    # genera un archivo para cada particion
    for particion in ["train", "validacion", "test"]:
        # construye la ruta del archivo de esta particion
        destino = DATOS / f"{particion}.csv"
        # abre el archivo para escribirlo sin lineas vacias extra
        with destino.open("w", encoding="utf-8", newline="") as archivo:
            # usa las claves de las filas como columnas del archivo
            escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
            # escribe los nombres de las columnas
            escritor.writeheader()
            # escribe solo las frases de la particion actual
            escritor.writerows(f for f in filas if f["particion"] == particion)
    # reune el origen los conteos las asignaciones y las huellas
    resumen = dict(
        # registra la semilla para repetir el procedimiento
        semilla=SEMILLA,
        # declara como se redactaron los datos
        origen="100% sintético; autoría asistida por IA; sin participantes humanos",
        # registra el total de mensajes
        n_filas=len(filas),
        # registra el total de escenarios
        n_grupos=len(asignacion),
        # conserva la particion elegida para cada grupo
        asignacion=asignacion,
        # cuenta mensajes por particion y etiqueta
        conteos={
            p: dict(Counter(f["etiqueta"] for f in filas if f["particion"] == p))
            for p in ["train", "validacion", "test"]
        },
        # guarda las huellas de los datos congelados
        sha256={
            nombre: huella(DATOS / nombre)
            for nombre in ["escenarios.txt", "train.csv", "validacion.csv", "test.csv"]
        },
    )
    # guarda el manifiesto del experimento
    guardar_json(manifiesto, resumen)
    # devuelve el resumen de la division
    return resumen


# detecta cambios en los datos congelados
def verificar_datos():
    # lee las huellas esperadas del manifiesto
    manifiesto = leer_json(DATOS / "manifiesto.json")
    # recorre los archivos que quedaron registrados
    for nombre, esperado in manifiesto["sha256"].items():
        # compara la huella actual con la guardada
        if huella(DATOS / nombre) != esperado:
            # detiene el proceso si detecta datos modificados
            raise RuntimeError(f"Datos modificados después de congelar: {nombre}")


# lee una particion especifica del corpus
def cargar(particion):
    # comprueba que la particion solicitada este permitida
    if particion not in {"train", "validacion", "test"}:
        # rechaza nombres ajenos a las tres particiones
        raise ValueError("Partición no permitida")
    # abre el archivo de la particion solicitada
    with (DATOS / f"{particion}.csv").open(encoding="utf-8", newline="") as archivo:
        # devuelve cada fila como un diccionario
        return list(csv.DictReader(archivo))


# separa los textos de las etiquetas deseadas
def xy(filas):
    # devuelve los textos y sus etiquetas en el mismo orden
    return [f["texto"] for f in filas], [f["etiqueta"] for f in filas]
