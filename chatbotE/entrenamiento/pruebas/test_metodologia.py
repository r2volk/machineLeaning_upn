# importa operaciones sobre arreglos y matrices
import numpy as np

# importa las herramientas para comprobar el programa
import pytest

# importa las rutas constantes y utilidades compartidas
from entrenamiento.p00_configuracion import DATOS, SALIDAS, leer_json

# importa la lectura limpieza y verificacion de datos
from entrenamiento.p01_datos import normalizar, verificar_datos

# importa los modelos y las reglas de evaluacion
from entrenamiento.p02_modelos import (
    crear_modelo,
    mascara_aceptacion,
    metricas,
    selectivas,
)

# importa la carga y el uso del modelo entrenado
from entrenamiento.p05_inferencia import cargar_modelo, predecir


# comprueba grupos y proporciones congelados
def test_grupos_y_proporciones_congelados():
    # lee la division registrada del corpus
    manifiesto = leer_json(DATOS / "manifiesto.json")
    # comprueba las huellas de los datos existentes
    verificar_datos()
    # extrae la particion asignada a cada grupo
    grupos = manifiesto["asignacion"]
    # comprueba que el total de grupos coincida con el registro
    assert len(grupos) == manifiesto["n_grupos"]
    # comprueba que existan las tres particiones
    assert set(grupos.values()) == {"train", "validacion", "test"}
    # revisa los conteos de cada particion
    for particion, conteos in manifiesto["conteos"].items():
        # comprueba que cada grupo aporte dos frases
        assert sum(conteos.values()) == 2 * sum(p == particion for p in grupos.values())
        # comprueba que las clases esten balanceadas
        assert len(set(conteos.values())) == 1


# comprueba vectorizador no aprende validacion
def test_vectorizador_no_aprende_validacion():
    # entrena un modelo de juguete con dos frases
    modelo = crear_modelo("logistica_bigramas").fit(
        ["alegre sol", "triste lluvia"], ["alegria", "tristeza"]
    )
    # conserva una copia del vocabulario aprendido
    antes = modelo["vectorizador"].vocabulary_.copy()
    # predice un texto con palabras no vistas durante el ajuste
    modelo.predict(["palabrasecreta validacion"])
    # comprueba que predecir no haya cambiado el vocabulario
    assert modelo["vectorizador"].vocabulary_ == antes
    # comprueba que la palabra nueva no se haya incorporado
    assert "palabrasecreta" not in antes


# comprueba negaciones no se eliminan
def test_negaciones_no_se_eliminan():
    # comprueba que la limpieza conserve la negacion
    assert normalizar("¡NO estoy FELÍZ!") == "no estoy feliz"
    # comprueba otras palabras que pueden cambiar el sentido
    assert normalizar("Sin ganas, nunca alegre.") == "sin ganas nunca alegre"


# comprueba metricas con resultado conocido
def test_metricas_con_resultado_conocido():
    # calcula medidas de un ejemplo con resultados conocidos
    resultado = metricas(
        ["alegria", "alegria", "tristeza", "tristeza"],
        ["alegria", "tristeza", "tristeza", "tristeza"],
    )
    # comprueba que tres de cuatro predicciones sean correctas
    assert resultado["accuracy"] == 0.75
    # comprueba que se recuperen todos los casos de tristeza
    assert resultado["por_clase"]["tristeza"]["recall"] == 1
    # comprueba la proporcion de aciertos al predecir tristeza
    assert resultado["por_clase"]["tristeza"]["precision"] == pytest.approx(2 / 3)


# comprueba abstencion no es una emocion
def test_abstencion_no_es_una_emocion():
    # prepara puntuaciones de tres ejemplos de juguete
    probs = np.array([[0.9, 0.1], [0.51, 0.49], [0.8, 0.2]])
    # aplica las reglas con un caso sin vocabulario conocido
    aceptadas = mascara_aceptacion(probs, 0.5, [True, True, False])
    # comprueba las decisiones de aceptar o rechazar
    assert aceptadas.tolist() == [True, False, False]
    # calcula resultados solo entre los ejemplos aceptados
    r = selectivas(["a", "b", "b"], ["a", "a", "a"], aceptadas)
    # comprueba que se acepte uno de los tres ejemplos
    assert r["cobertura"] == pytest.approx(1 / 3)
    # comprueba que el unico caso aceptado sea correcto
    assert r["accuracy_aceptados"] == 1


# comprueba no hay accuracy si nadie es aceptado
def test_no_hay_accuracy_si_nadie_es_aceptado():
    # comprueba que no se inventen aciertos cuando nadie es aceptado
    assert selectivas(["a"], ["a"], np.array([False]))["accuracy_aceptados"] is None


# comprueba texto desconocido no fuerza neutral
def test_texto_desconocido_no_fuerza_neutral():
    # predice una entrada con texto y simbolos desconocidos
    resultado = predecir(cargar_modelo(), "zzqwwx 🛸")
    # comprueba que no se fuerce una emocion
    assert resultado["etiqueta"] == "incierto"
    # comprueba que se explique la ausencia de vocabulario
    assert resultado["motivo"] == "sin_vocabulario"
    # comprueba que las puntuaciones de clase sumen uno
    assert sum(resultado["scores"].values()) == pytest.approx(1)


# comprueba explicacion tiene aportes del modelo
def test_explicacion_tiene_aportes_del_modelo():
    # carga el modelo de la entrega
    artefacto = cargar_modelo()
    # analiza una frase de desarrollo
    r = predecir(artefacto, "Estoy muy feliz y alegre")
    # recupera el vectorizador del modelo
    vectorizador = artefacto["pipeline"]["vectorizador"]
    # extrae las caracteristicas que el texto puede producir
    tokens = set(vectorizador.build_analyzer()("Estoy muy feliz y alegre"))
    # comprueba que los aportes sean positivos y pertenezcan al texto
    assert all(e["termino"] in tokens and e["aporte"] > 0 for e in r["evidencia"])


# comprueba evaluacion existente no vuelve a cargar modelo
def test_evaluacion_existente_no_vuelve_a_cargar_modelo(monkeypatch, tmp_path):
    # importa la etapa de evaluacion unica del test
    import entrenamiento.p04_evaluacion as modulo

    # crea un resultado simulado para evitar abrir el test real
    (tmp_path / "test_final.json").write_text('{"sentinela":true}', encoding="utf-8")
    # redirige los resultados a una carpeta temporal
    monkeypatch.setattr(modulo, "SALIDAS", tmp_path)

    # detecta una llamada que esta prueba debe impedir
    def prohibido(*args, **kwargs):
        # hace fallar la prueba si intenta cargar datos o el modelo
        raise AssertionError("No debe volver a abrir test ni cargar el modelo")

    # bloquea temporalmente la lectura de particiones
    monkeypatch.setattr(modulo, "cargar", prohibido)
    # bloquea temporalmente la carga del modelo
    monkeypatch.setattr(modulo.joblib, "load", prohibido)
    # comprueba que reutilice el resultado sin hacer otra inferencia
    assert modulo.evaluar_una_vez() == {"sentinela": True}


# comprueba recuperacion sin artefacto solo usa train
def test_recuperacion_sin_artefacto_solo_usa_train(monkeypatch, tmp_path):
    # importa la carga y el uso del modelo entrenado
    import entrenamiento.p05_inferencia as modulo

    # carga el modelo original para compararlo despues
    original = cargar_modelo()
    # conserva la funcion real de lectura de particiones
    real_cargar = modulo.cargar
    # prepara el registro de particiones consultadas
    particiones = []

    # registra las particiones leidas durante la recuperacion
    def registrar(particion):
        # registra cada particion que se solicita
        particiones.append(particion)
        # delega la lectura en la funcion original
        return real_cargar(particion)

    # simula que falta el archivo del modelo
    monkeypatch.setattr(modulo, "MODELO", tmp_path / "ausente.joblib")
    # sustituye la lectura por otra que registra sus llamadas
    monkeypatch.setattr(modulo, "cargar", registrar)
    # reconstruye el modelo por el camino de recuperacion
    recuperado = modulo.cargar_modelo()
    # comprueba que solo se haya leido entrenamiento
    assert particiones == ["train"]
    # define un mensaje de desarrollo para comparar ambos modelos
    mensaje = "Estoy feliz por mi proyecto"
    # obtiene las puntuaciones del modelo original
    esperado = predecir(original, mensaje)["scores"]
    # obtiene las puntuaciones del modelo reconstruido
    obtenido = predecir(recuperado, mensaje)["scores"]
    # comprueba que las puntuaciones sean equivalentes
    assert obtenido == pytest.approx(esperado)
    # comprueba que la recuperacion no escriba un archivo nuevo
    assert not (tmp_path / "ausente.joblib").exists()


# comprueba artefacto alterado no se deserializa
def test_artefacto_alterado_no_se_deserializa(monkeypatch, tmp_path):
    # importa la carga y el uso del modelo entrenado
    import entrenamiento.p05_inferencia as modulo

    # prepara la ruta de un modelo falso
    falso = tmp_path / "modelo.joblib"
    # escribe contenido que no corresponde al modelo original
    falso.write_bytes(b"contenido alterado")
    # hace que la carga apunte al archivo falso
    monkeypatch.setattr(modulo, "MODELO", falso)
    # espera un error al detectar la huella alterada
    with pytest.raises(RuntimeError, match="Artefacto alterado"):
        # intenta cargar el archivo para comprobar el rechazo
        modulo.cargar_modelo()


# comprueba integridad sin exigir coincidencia del codigo ni abrir el test real
@pytest.mark.parametrize(
    "caso",
    ["codigo_distinto", "sin_huellas_codigo", "manifiesto_alterado", "modelo_alterado"],
)
def test_evaluacion_verifica_datos_y_modelo_sin_bloquear_por_codigo(
    monkeypatch, tmp_path, caso
):
    import entrenamiento.p04_evaluacion as modulo
    from entrenamiento.p00_configuracion import guardar_json, huella

    # usa solo archivos temporales y una reserva que impide cualquier inferencia
    manifiesto = tmp_path / "manifiesto.json"
    manifiesto.write_text("{}", encoding="utf-8")
    modelo = tmp_path / "modelo.joblib"
    modelo.write_bytes(b"modelo de prueba sin deserializar")
    seleccion = {
        "manifiesto_sha256": huella(manifiesto),
        "modelo_sha256": huella(modelo),
    }
    if caso != "sin_huellas_codigo":
        seleccion["codigo_sha256"] = {"inexistente.py": "huella diferente"}
    guardar_json(tmp_path / "seleccion.json", seleccion)
    reserva = tmp_path / "test_reservado.json"
    reserva.write_text('{"estado":"reservado"}', encoding="utf-8")
    if caso == "manifiesto_alterado":
        manifiesto.write_text('{"cambio":true}', encoding="utf-8")
    if caso == "modelo_alterado":
        modelo.write_bytes(b"modelo alterado")

    monkeypatch.setattr(modulo, "SALIDAS", tmp_path)
    monkeypatch.setattr(modulo, "DATOS", tmp_path)
    monkeypatch.setattr(modulo, "MODELO", modelo)
    verificaciones = []
    monkeypatch.setattr(modulo, "verificar_datos", lambda: verificaciones.append(True))

    def prohibido(*args, **kwargs):
        raise AssertionError("No debe cargar modelo ni abrir datos de test")

    monkeypatch.setattr(modulo.joblib, "load", prohibido)
    monkeypatch.setattr(modulo, "cargar", prohibido)
    if caso == "manifiesto_alterado":
        with pytest.raises(RuntimeError, match="El manifiesto fue alterado"):
            modulo.evaluar_una_vez()
    elif caso == "modelo_alterado":
        with pytest.raises(RuntimeError, match="El modelo no coincide"):
            modulo.evaluar_una_vez()
    else:
        # alcanza la reserva aunque falten huellas del codigo o sean diferentes
        with pytest.raises(FileExistsError):
            modulo.evaluar_una_vez()
    assert verificaciones == [True]
    assert not (tmp_path / "test_final.json").exists()
    assert reserva.read_text(encoding="utf-8") == '{"estado":"reservado"}'
