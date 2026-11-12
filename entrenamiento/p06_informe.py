# importa herramientas para leer y escribir tablas de texto
import csv

# importa las herramientas para crear graficos
import matplotlib

# permite crear imagenes sin abrir ventanas graficas
matplotlib.use("Agg")
# importa las funciones para dibujar y guardar figuras
import matplotlib.pyplot as plt

# importa operaciones sobre arreglos y matrices
import numpy as np

# importa las rutas constantes y utilidades compartidas
from .p00_configuracion import CLASES, SALIDAS, leer_json


# dibuja tres graficos desde resultados guardados
def graficar(seleccion, final):
    # ubica la carpeta de las figuras
    carpeta = SALIDAS / "figuras"
    # crea la carpeta si todavia no existe
    carpeta.mkdir(exist_ok=True)
    # configura la tipografia los bordes y el fondo de los graficos
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fafbf7",
        }
    )
    # crea la figura para comparar candidatos
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    # extrae los resultados de los cinco modelos
    candidatos = seleccion["candidatos"]
    # separa los nombres largos para facilitar su lectura
    etiquetas = [c["nombre"].replace("_", "\n") for c in candidatos]
    # asigna una posicion a cada candidato
    x = np.arange(len(candidatos))
    # dibuja las barras del resultado en entrenamiento
    ax.bar(
        x - 0.18,
        [c["train"]["macro_f1"] for c in candidatos],
        0.36,
        # identifica los datos en la leyenda
        label="Train",
        # elige un color que permita distinguir los datos
        color="#aebf8d",
    )
    # dibuja al lado las barras del resultado en validacion
    ax.bar(
        x + 0.18,
        [c["validacion"]["macro_f1"] for c in candidatos],
        0.36,
        # identifica los datos en la leyenda
        label="Validación",
        # elige un color que permita distinguir los datos
        color="#305b49",
    )
    # define nombres escala y titulo de la comparacion
    ax.set(
        # ubica las marcas del eje horizontal
        xticks=x,
        # muestra los nombres del eje horizontal
        xticklabels=etiquetas,
        # fija el intervalo visible del eje vertical
        ylim=(0, 1.05),
        # describe el eje vertical
        ylabel="Macro-F1",
        # anade el titulo del grafico
        title="Comparación previa al test · corpus sintético",
    )
    # muestra la leyenda para distinguir ambas particiones
    ax.legend()
    # guarda el grafico de comparacion
    guardar_figura(fig, carpeta / "comparacion")
    # crea otra figura para la matriz de confusion
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    # convierte los conteos guardados en una matriz numerica
    matriz = np.array(final["metricas"]["matriz"])
    # representa los conteos con intensidad de color
    im = ax.imshow(matriz, cmap="YlGn", vmin=0)
    # recorre las clases reales por fila
    for i in range(len(CLASES)):
        # recorre las clases predichas por columna
        for j in range(len(CLASES)):
            # escribe el conteo con un color de texto que contraste
            ax.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                # elige un color que permita distinguir los datos
                color="white" if matriz[i, j] > matriz.max() * 0.65 else "#20392f",
            )
    # identifica los ejes y las clases de la matriz
    ax.set(
        # ubica las marcas del eje horizontal
        xticks=range(len(CLASES)),
        # muestra los nombres del eje horizontal
        xticklabels=CLASES,
        # ubica las marcas del eje vertical
        yticks=range(len(CLASES)),
        # muestra los nombres del eje vertical
        yticklabels=CLASES,
        # describe el eje horizontal
        xlabel="Predicción",
        # describe el eje vertical
        ylabel="Etiqueta real",
        # anade el titulo del grafico
        title="Matriz de confusión · test final sintético",
    )
    # muestra la escala de color como cantidad de mensajes
    fig.colorbar(im, ax=ax, label="Número de mensajes")
    # guarda la matriz de confusion
    guardar_figura(fig, carpeta / "matriz_confusion")
    # crea otra figura para la curva de aprendizaje
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    # relaciona la cantidad de entrenamiento con el resultado
    ax.plot(
        [c["filas"] for c in seleccion["curva"]],
        [c["macro_f1"] for c in seleccion["curva"]],
        "o-",
        # elige un color que permita distinguir los datos
        color="#305b49",
    )
    # define los ejes y aclara que se reutiliza la validacion
    ax.set(
        # fija el intervalo visible del eje vertical
        ylim=(0, 1),
        # describe el eje horizontal
        xlabel="Mensajes de entrenamiento (grupos completos)",
        # describe el eje vertical
        ylabel="Macro-F1 de validación",
        # anade el titulo del grafico
        title="Curva exploratoria · misma validación en cada punto",
    )
    # muestra una cuadricula tenue como referencia
    ax.grid(alpha=0.2)
    # guarda la curva de aprendizaje
    guardar_figura(fig, carpeta / "curva_aprendizaje")


# exporta el grafico y libera su memoria
def guardar_figura(figura, destino):
    # guarda una imagen para verla o insertarla en documentos
    figura.savefig(destino.with_suffix(".png"), dpi=160)
    # guarda una version vectorial que permite ampliacion
    figura.savefig(destino.with_suffix(".svg"))
    # cierra la figura para liberar memoria
    plt.close(figura)


# regenera graficos y lista errores sin repetir el test
def generar_informe():
    # lee la comparacion de modelos ya realizada
    seleccion = leer_json(SALIDAS / "seleccion.json")
    # lee las predicciones finales guardadas
    final = leer_json(SALIDAS / "test_final.json")
    # genera los tres graficos sin volver a predecir
    graficar(seleccion, final)
    # reune los mensajes cuya etiqueta predicha fue incorrecta
    errores = [f for f in final["predicciones"] if f["etiqueta"] != f["predicha"]]
    # abre el archivo donde se listaran los errores
    with (SALIDAS / "errores_test.csv").open(
        "w", encoding="utf-8", newline=""
    ) as archivo:
        # define las columnas del archivo de errores
        escritor = csv.DictWriter(
            archivo,
            # elige las columnas que se escribiran en el archivo
            fieldnames=["id", "grupo", "texto", "etiqueta", "predicha", "aceptada"],
        )
        # escribe los encabezados de las columnas
        escritor.writeheader()
        # escribe los campos elegidos de cada prediccion incorrecta
        escritor.writerows({k: f[k] for k in escritor.fieldnames} for f in errores)
