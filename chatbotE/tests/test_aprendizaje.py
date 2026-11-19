# comprueba el aprendizaje voluntario sin usar datos personales
import pytest

from app import crear_app
from entrenamiento import p10_aprendizaje_local as local


@pytest.fixture
# usa una carpeta temporal para no tocar datos del usuario

def cliente(tmp_path, monkeypatch):
    monkeypatch.setattr(local, 'LOCAL', tmp_path / 'privado')
    aplicacion = crear_app()
    aplicacion.config['TESTING'] = True
    return aplicacion.test_client()


# guardar y borrar no agrega conversaciones automaticamente

def test_guardar_y_borrar_ejemplo(cliente):
    assert cliente.get('/api/aprendizaje').json['ejemplos'] == 0
    respuesta = cliente.post('/api/aprendizaje/ejemplos', json={
        'texto': 'mi cometa violeta me alegró durante el paseo', 'etiqueta': 'alegria',
    })
    assert respuesta.status_code == 201
    assert cliente.get('/api/aprendizaje').json['ejemplos'] == 1
    assert cliente.post('/api/aprendizaje/ejemplos', json={
        'texto': 'Mi cometa violeta me alegró durante el paseo!', 'etiqueta': 'tristeza',
    }).status_code == 400
    assert cliente.delete('/api/aprendizaje/ejemplos').json['ejemplos'] == 0


# rechaza etiquetas desconocidas y mensajes de riesgo

def test_no_guardar_riesgo_o_etiquetas_invalidas(cliente):
    assert cliente.post('/api/aprendizaje/ejemplos', json={
        'texto': 'quiero suicidarme', 'etiqueta': 'tristeza',
    }).status_code == 400
    assert cliente.post('/api/aprendizaje/ejemplos', json={
        'texto': 'tengo un día bueno', 'etiqueta': 'felicidad',
    }).status_code == 400
    assert cliente.get('/api/aprendizaje').json['ejemplos'] == 0


# verifica entrenamiento persistencia y restauracion del modelo

def test_entrenar_y_restaurar(cliente):
    cliente.post('/api/aprendizaje/ejemplos', json={
        'texto': 'mi cometa violeta me alegró durante el paseo', 'etiqueta': 'alegria',
    })
    respuesta = cliente.post('/api/aprendizaje/entrenar')
    assert respuesta.status_code == 200
    assert respuesta.json['ejemplos_locales'] == 1
    assert local.cargar_actual()['nombre'] == 'logistica_bigramas_local'
    cliente.delete('/api/aprendizaje/ejemplos')
    assert local.cargar_actual()['nombre'] == 'combinado'
