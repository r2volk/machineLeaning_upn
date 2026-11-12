# importa la fecha y la zona horaria del registro
from datetime import datetime, timezone

# importa la carga y el guardado de modelos locales
import joblib

# importa operaciones sobre arreglos y matrices
import numpy as np

# importa las rutas constantes y utilidades compartidas
from .p00_configuracion import (
    DATOS,
    MODELO,
    SALIDAS,
    SEMILLA,
    guardar_json,
    huella,
    leer_json,
)

# importa la lectura limpieza y verificacion de datos
from .p01_datos import cargar, verificar_datos, xy

# importa los modelos y las reglas de evaluacion
from .p02_modelos import mascara_aceptacion, metricas, selectivas


# estima variabilidad remuestreando predicciones por escenario
def intervalo_por_grupos(filas, predichas, repeticiones=1000):
    # crea el generador para repetir el remuestreo
    rng = np.random.default_rng(SEMILLA)
    # reune los escenarios presentes en las predicciones
    grupos = sorted({f["grupo"] for f in filas})
    # localiza las dos frases que pertenecen a cada escenario
    indices = {g: [i for i, f in enumerate(filas) if f["grupo"] == g] for g in grupos}
    # extrae las etiquetas reales en el orden de las frases
    reales = np.array([f["etiqueta"] for f in filas])
    # prepara las medidas de las replicas
    valores = []
    # importa las medidas para comparar etiquetas
    from sklearn.metrics import f1_score

    # importa las rutas constantes y utilidades compartidas
    from .p00_configuracion import CLASES

    # repite el remuestreo la cantidad indicada
    for _ in range(repeticiones):
        # elige escenarios con reemplazo e incluye sus dos frases
        muestra = [
            i
            for g in rng.choice(grupos, size=len(grupos), replace=True)
            for i in indices[g]
        ]
        # calcula la medida de la replica con predicciones existentes
        valores.append(
            f1_score(
                reales[muestra],
                predichas[muestra],
                labels=CLASES,
                average="macro",
                zero_division=0,
            )
        )
    # devuelve los extremos del intervalo y el metodo usado
    return dict(
        # toma el extremo inferior del intervalo exploratorio
        inferior=float(np.quantile(valores, 0.025)),
        # toma el extremo superior del intervalo exploratorio
        superior=float(np.quantile(valores, 0.975)),
        # registra cuantas replicas se calcularon
        repeticiones=repeticiones,
        # describe el procedimiento usado para el intervalo
        metodo="Percentil 95%, bootstrap por escenario; exploratorio",
    )


# reserva el test y guarda una unica evaluacion
def evaluar_una_vez():
    # ubica el archivo de evaluacion final
    salida = SALIDAS / "test_final.json"
    # comprueba si el resultado ya existe
    if salida.exists():
        # devuelve la evidencia sin cargar modelo ni abrir test
        return leer_json(salida)
    # lee el modelo elegido antes de test
    seleccion = leer_json(SALIDAS / "seleccion.json")
    # verifica las huellas de los datos congelados
    verificar_datos()
    # comprueba que el manifiesto sea el usado para elegir el modelo
    if huella(DATOS / "manifiesto.json") != seleccion["manifiesto_sha256"]:
        # rechaza un manifiesto distinto del registrado
        raise RuntimeError("El manifiesto fue alterado")
    # comprueba que el modelo coincida con el elegido
    if huella(MODELO) != seleccion["modelo_sha256"]:
        # rechaza un modelo diferente del registrado
        raise RuntimeError("El modelo no coincide con el seleccionado")

    # ubica el marcador que reserva una unica apertura del test
    reserva = SALIDAS / "test_reservado.json"
    # crea la reserva y falla si ya existe
    with reserva.open("x", encoding="utf-8") as archivo:
        # deja constancia de la reserva antes de cargar test
        archivo.write('{"estado":"reservado", "politica":"no reabrir ni borrar"}')
    # carga el modelo elegido y verificado
    modelo = joblib.load(MODELO)["pipeline"]
    # abre la particion reservada de test
    filas = cargar("test")
    # separa textos y etiquetas de esta evaluacion
    textos, reales = xy(filas)
    # realiza una sola inferencia sobre todo el lote de test
    probabilidades = modelo.predict_proba(textos)
    # obtiene la clase de mayor puntuacion por mensaje
    predichas = modelo.classes_[np.argmax(probabilidades, axis=1)]
    # comprueba si cada mensaje contiene vocabulario conocido
    conocidos = modelo["vectorizador"].transform(textos).getnnz(axis=1) > 0
    # aplica el umbral y el margen fijados antes de test
    aceptar = mascara_aceptacion(probabilidades, seleccion["umbral"], conocidos)
    # reune medidas predicciones puntuaciones y huellas del resultado
    resultado = dict(
        # identifica el modelo que produjo el resultado
        modelo=seleccion["ganador"],
        # cuenta los mensajes del lote final
        n=len(filas),
        # registra la fecha de la evaluacion final
        fecha_utc=datetime.now(timezone.utc).isoformat(),
        # guarda las medidas de todas las predicciones
        metricas=metricas(reales, predichas),
        # guarda cobertura y aciertos entre los casos aceptados
        selectivas=selectivas(reales, predichas, aceptar),
        # estima variabilidad sin repetir la inferencia del modelo
        intervalo_macro_f1=intervalo_por_grupos(filas, predichas),
        # guarda cada etiqueta predicha y su puntuacion
        predicciones=[
            dict(**f, predicha=str(p), aceptada=bool(a), scores=s.tolist())
            for f, p, a, s in zip(filas, predichas, aceptar, probabilidades)
        ],
        # guarda la huella del modelo entrenado
        modelo_sha256=huella(MODELO),
        # guarda la huella de la seleccion previa al test
        seleccion_sha256=huella(SALIDAS / "seleccion.json"),
    )
    # guarda la evidencia de la evaluacion final
    guardar_json(salida, resultado)
    # marca la apertura como completada y registra la huella final
    guardar_json(
        reserva,
        dict(estado="completado", aperturas=1, test_final_sha256=huella(salida)),
    )
    # devuelve el resultado de la evaluacion
    return resultado
