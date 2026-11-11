"""Contrato HTTP, privacidad, validación y prioridad de seguridad con mensajes de desarrollo."""
import json
from pathlib import Path

import pytest
from app import crear_app
from seguridad import detectar_riesgo

CASOS = json.loads((Path(__file__).parent / 'casos_seguridad.json').read_text(encoding='utf-8'))


@pytest.fixture
def app():
    aplicacion = crear_app()
    aplicacion.config.update(TESTING=True)
    return aplicacion


@pytest.fixture
def client(app):
    return app.test_client()


def test_pagina_desde_flask_y_csp(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'Un momento para ti' in r.text
    assert "fetch('/api/chat'" in r.text
    assert 'nonce-' in r.headers['Content-Security-Policy']
    assert r.headers['Cache-Control'] == 'no-store'
    assert 'Set-Cookie' not in r.headers


@pytest.mark.parametrize('data', [None, [], {}, {'message': None}, {'message': 5}, {'message': '   '}])
def test_entradas_invalidas(client, data):
    assert client.post('/api/chat', data=json.dumps(data), content_type='application/json').status_code == 400


def test_json_malformado_y_tipo_incorrecto(client):
    assert client.post('/api/chat', data='{', content_type='application/json').status_code == 400
    assert client.post('/api/chat', data='hola').status_code == 415


def test_limites_y_turno(client):
    assert client.post('/api/chat', json={'message': 'x'*1501}).status_code == 400
    assert client.post('/api/chat', json={'message': 'hola', 'turn': True}).status_code == 400
    assert client.post('/api/chat', json={'message': 'x'*40000}).status_code == 413


def test_respuesta_normal(client):
    r = client.post('/api/chat', json={'message': 'Estoy alegre y feliz', 'turn': 0})
    assert r.status_code == 200
    assert r.json['reply']
    assert not r.json['safety']
    assert len(r.json['analysis']['scores']) == 5


@pytest.mark.parametrize('caso', CASOS, ids=[str(i) for i in range(len(CASOS))])
def test_casos_de_precaucion(caso):
    assert detectar_riesgo(caso['texto']) is caso['precaucion']


def test_riesgo_salta_modelo_aun_no_disponible(app, monkeypatch):
    import app as modulo
    app.config['ARTEFACTO'] = None
    def prohibido(*args):
        raise AssertionError('Riesgo nunca debe consultar ML')
    monkeypatch.setattr(modulo, 'predecir', prohibido)
    r = app.test_client().post('/api/chat', json={'message': 'Quiero hacerme daño'})
    assert r.status_code == 200 and r.json['safety'] is True
    assert r.json['analysis'] is None
    assert '113, opción 5' in r.json['reply']
    assert 'no puedo evaluar tu seguridad' in r.json['reply']


def test_riesgo_en_mensaje_largo_tiene_prioridad(client):
    r = client.post('/api/chat', json={'message': 'hola '*400 + 'quiero suicidarme'})
    assert r.status_code == 200 and r.json['safety']


def test_continuacion_de_seguridad(client):
    r = client.post('/api/chat', json={'message': 'Estoy solo', 'safety_context': True})
    assert r.json['safety'] and r.json['analysis'] is None
    assert client.post('/api/chat', json={'message': 'hola', 'safety_context': 'false'}).status_code == 400


@pytest.mark.parametrize('texto', ['Me voy a hacer daño', 'Quisiera dejar de existir', 'Me he cortado', 'Estoy pensando en hacerme dao'])
def test_variantes_de_riesgo(texto):
    assert detectar_riesgo(texto)


def test_modelo_degradado_no_inventa_respuestas(app):
    app.config['ARTEFACTO'] = None
    c = app.test_client()
    assert c.get('/api/health').status_code == 503
    assert c.post('/api/chat', json={'message': 'Estoy feliz'}).status_code == 503


def test_no_hay_acceso_a_archivos_arbitrarios(client):
    assert client.get('/documentos/modelo.joblib').status_code == 404
    assert client.get('/entrenamiento/datos/test.csv').status_code == 404
    assert client.get('/api/health', headers={'Host': 'sitio-ajeno.example'}).status_code == 400


def test_html_del_usuario_permanece_dato(client):
    r = client.post('/api/chat', json={'message': '<script>alert(1)</script> Estoy feliz'})
    assert r.status_code == 200
    assert '<script>' not in r.json['reply']
