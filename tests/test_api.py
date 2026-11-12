# importa la conversion entre texto json y datos de python
import json

# importa el manejo de rutas de archivos y carpetas
from pathlib import Path

# importa las herramientas para comprobar el programa
import pytest

# importa la aplicacion o su modulo para las pruebas
from app import crear_app

# importa las reglas y la respuesta de precaucion
from seguridad import detectar_riesgo

# carga los ejemplos de desarrollo para las reglas de precaucion
CASOS = json.loads(
    (Path(__file__).parent / "casos_seguridad.json").read_text(encoding="utf-8")
)


# prepara una aplicacion para las pruebas
@pytest.fixture
def app():
    # crea una aplicacion independiente para la prueba
    aplicacion = crear_app()
    # activa el modo de pruebas del servidor
    aplicacion.config.update(TESTING=True)
    # devuelve la aplicacion preparada
    return aplicacion


# prepara un cliente simulado para las pruebas
@pytest.fixture
def client(app):
    # crea un cliente que simula peticiones sin abrir un servidor real
    return app.test_client()


# comprueba pagina desde flask y csp
def test_pagina_desde_flask_y_csp(client):
    # solicita la pagina principal
    r = client.get("/")
    # comprueba que la pagina se entregue correctamente
    assert r.status_code == 200
    # comprueba que aparezca el titulo esperado
    assert "Un momento para ti" in r.text
    # comprueba que la interfaz use una ruta relativa para el chat
    assert "fetch('/api/chat'" in r.text
    # comprueba que exista la politica de permisos para los scripts
    assert "nonce-" in r.headers["Content-Security-Policy"]
    # comprueba que la respuesta no se guarde en cache
    assert r.headers["Cache-Control"] == "no-store"
    # comprueba que la pagina no cree cookies
    assert "Set-Cookie" not in r.headers


# comprueba entradas invalidas
@pytest.mark.parametrize(
    "data", [None, [], {}, {"message": None}, {"message": 5}, {"message": "   "}]
)
def test_entradas_invalidas(client, data):
    # comprueba el rechazo de cada estructura de entrada invalida
    assert (
        client.post(
            "/api/chat", data=json.dumps(data), content_type="application/json"
        ).status_code
        == 400
    )


# comprueba json malformado y tipo incorrecto
def test_json_malformado_y_tipo_incorrecto(client):
    # comprueba el rechazo de un cuerpo json incompleto
    assert (
        client.post("/api/chat", data="{", content_type="application/json").status_code
        == 400
    )
    # comprueba el rechazo de texto sin el tipo de contenido requerido
    assert client.post("/api/chat", data="hola").status_code == 415


# comprueba limites y turno
def test_limites_y_turno(client):
    # comprueba el limite de caracteres del chat normal
    assert client.post("/api/chat", json={"message": "x" * 1501}).status_code == 400
    # comprueba que un booleano no se admita como numero de turno
    assert (
        client.post("/api/chat", json={"message": "hola", "turn": True}).status_code
        == 400
    )
    # comprueba el rechazo de un cuerpo demasiado grande
    assert client.post("/api/chat", json={"message": "x" * 40000}).status_code == 413


# comprueba respuesta normal
def test_respuesta_normal(client):
    # envia una frase de desarrollo al chat
    r = client.post("/api/chat", json={"message": "Estoy alegre y feliz", "turn": 0})
    # comprueba que la peticion termine correctamente
    assert r.status_code == 200
    # comprueba que exista una respuesta redactada
    assert r.json["reply"]
    # comprueba que no se active precaucion en este ejemplo
    assert not r.json["safety"]
    # comprueba que se devuelvan puntuaciones para cinco clases
    assert len(r.json["analysis"]["scores"]) == 5


# comprueba casos de precaucion
@pytest.mark.parametrize("caso", CASOS, ids=[str(i) for i in range(len(CASOS))])
def test_casos_de_precaucion(caso):
    # compara la activacion de cada caso con la respuesta esperada
    assert detectar_riesgo(caso["texto"]) is caso["precaucion"]


# comprueba riesgo salta modelo aun no disponible
def test_riesgo_salta_modelo_aun_no_disponible(app, monkeypatch):
    # importa la aplicacion o su modulo para las pruebas
    import app as modulo

    # simula una aplicacion sin modelo disponible
    app.config["ARTEFACTO"] = None

    # detecta una llamada que esta prueba debe impedir
    def prohibido(*args):
        # hace fallar la prueba si la alerta llega al modelo emocional
        raise AssertionError("Riesgo nunca debe consultar ML")

    # sustituye la prediccion por una funcion que impide usarla
    monkeypatch.setattr(modulo, "predecir", prohibido)
    # envia una senal explicita de precaucion
    r = app.test_client().post("/api/chat", json={"message": "Quiero hacerme daño"})
    # comprueba que se responda con apoyo aunque falte el modelo
    assert r.status_code == 200 and r.json["safety"] is True
    # comprueba que no se muestre una clasificacion emocional
    assert r.json["analysis"] is None
    # comprueba que la respuesta incluya el recurso solicitado
    assert "113, opción 5" in r.json["reply"]
    # comprueba que la respuesta declare los limites del prototipo
    assert "no puedo evaluar tu seguridad" in r.json["reply"]


# comprueba riesgo en mensaje largo tiene prioridad
def test_riesgo_en_mensaje_largo_tiene_prioridad(client):
    # envia una senal dentro de un mensaje largo
    r = client.post("/api/chat", json={"message": "hola " * 400 + "quiero suicidarme"})
    # comprueba que la precaucion tenga prioridad sobre ese limite
    assert r.status_code == 200 and r.json["safety"]


# comprueba continuacion de seguridad
def test_continuacion_de_seguridad(client):
    # envia el indicador de una alerta anterior
    r = client.post("/api/chat", json={"message": "Estoy solo", "safety_context": True})
    # comprueba que se mantenga la precaucion sin clasificar emociones
    assert r.json["safety"] and r.json["analysis"] is None
    # comprueba que se rechace un indicador con tipo incorrecto
    assert (
        client.post(
            "/api/chat", json={"message": "hola", "safety_context": "false"}
        ).status_code
        == 400
    )


# comprueba variantes de riesgo
@pytest.mark.parametrize(
    "texto",
    [
        "Me voy a hacer daño",
        "Quisiera dejar de existir",
        "Me he cortado",
        "Estoy pensando en hacerme dao",
    ],
)
def test_variantes_de_riesgo(texto):
    # comprueba que cada variante active precaucion
    assert detectar_riesgo(texto)


# comprueba modelo degradado no inventa respuestas
def test_modelo_degradado_no_inventa_respuestas(app):
    # simula que el modelo no pudo cargarse
    app.config["ARTEFACTO"] = None
    # crea un cliente con esa aplicacion sin modelo
    c = app.test_client()
    # comprueba que la ruta de salud informe el problema
    assert c.get("/api/health").status_code == 503
    # comprueba que el chat normal informe que no puede clasificar
    assert c.post("/api/chat", json={"message": "Estoy feliz"}).status_code == 503


# comprueba no hay acceso a archivos arbitrarios
def test_no_hay_acceso_a_archivos_arbitrarios(client):
    # comprueba que no se pueda descargar el modelo por esta ruta
    assert client.get("/documentos/modelo.joblib").status_code == 404
    # comprueba que la web no exponga directamente el test del corpus
    assert client.get("/entrenamiento/datos/test.csv").status_code == 404
    # comprueba que se rechace un dominio ajeno al servidor local
    assert (
        client.get("/api/health", headers={"Host": "sitio-ajeno.example"}).status_code
        == 400
    )


# comprueba html del usuario permanece dato
def test_html_del_usuario_permanece_dato(client):
    # envia contenido que parece codigo junto con una frase normal
    r = client.post(
        "/api/chat", json={"message": "<script>alert(1)</script> Estoy feliz"}
    )
    # comprueba que la peticion se procese
    assert r.status_code == 200
    # comprueba que la respuesta no incluya el codigo enviado
    assert "<script>" not in r.json["reply"]
