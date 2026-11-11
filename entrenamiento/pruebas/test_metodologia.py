"""Pruebas del procedimiento con datos de juguete; nunca infieren sobre el test real."""
import numpy as np
import pytest

from entrenamiento.p00_configuracion import DATOS, SALIDAS, leer_json
from entrenamiento.p01_datos import normalizar, verificar_datos
from entrenamiento.p02_modelos import crear_modelo, mascara_aceptacion, metricas, selectivas
from entrenamiento.p05_inferencia import cargar_modelo, predecir


def test_grupos_y_proporciones_congelados():
    manifiesto = leer_json(DATOS / 'manifiesto.json')
    verificar_datos()
    grupos = manifiesto['asignacion']
    assert len(grupos) == manifiesto['n_grupos']
    assert set(grupos.values()) == {'train', 'validacion', 'test'}
    for particion, conteos in manifiesto['conteos'].items():
        assert sum(conteos.values()) == 2 * sum(p == particion for p in grupos.values())
        assert len(set(conteos.values())) == 1


def test_vectorizador_no_aprende_validacion():
    modelo = crear_modelo('logistica_bigramas').fit(['alegre sol', 'triste lluvia'], ['alegria', 'tristeza'])
    antes = modelo['vectorizador'].vocabulary_.copy()
    modelo.predict(['palabrasecreta validacion'])
    assert modelo['vectorizador'].vocabulary_ == antes
    assert 'palabrasecreta' not in antes


def test_negaciones_no_se_eliminan():
    assert normalizar('¡NO estoy FELÍZ!') == 'no estoy feliz'
    assert normalizar('Sin ganas, nunca alegre.') == 'sin ganas nunca alegre'


def test_metricas_con_resultado_conocido():
    resultado = metricas(['alegria', 'alegria', 'tristeza', 'tristeza'],
                         ['alegria', 'tristeza', 'tristeza', 'tristeza'])
    assert resultado['accuracy'] == .75
    assert resultado['por_clase']['tristeza']['recall'] == 1
    assert resultado['por_clase']['tristeza']['precision'] == pytest.approx(2/3)


def test_abstencion_no_es_una_emocion():
    probs = np.array([[.9, .1], [.51, .49], [.8, .2]])
    aceptadas = mascara_aceptacion(probs, .5, [True, True, False])
    assert aceptadas.tolist() == [True, False, False]
    r = selectivas(['a', 'b', 'b'], ['a', 'a', 'a'], aceptadas)
    assert r['cobertura'] == pytest.approx(1/3)
    assert r['accuracy_aceptados'] == 1


def test_no_hay_accuracy_si_nadie_es_aceptado():
    assert selectivas(['a'], ['a'], np.array([False]))['accuracy_aceptados'] is None


def test_texto_desconocido_no_fuerza_neutral():
    resultado = predecir(cargar_modelo(), 'zzqwwx 🛸')
    assert resultado['etiqueta'] == 'incierto'
    assert resultado['motivo'] == 'sin_vocabulario'
    assert sum(resultado['scores'].values()) == pytest.approx(1)


def test_explicacion_tiene_aportes_del_modelo():
    artefacto = cargar_modelo()
    r = predecir(artefacto, 'Estoy muy feliz y alegre')
    vectorizador = artefacto['pipeline']['vectorizador']
    tokens = set(vectorizador.build_analyzer()('Estoy muy feliz y alegre'))
    assert all(e['termino'] in tokens and e['aporte'] > 0 for e in r['evidencia'])


def test_evaluacion_existente_no_vuelve_a_cargar_modelo(monkeypatch, tmp_path):
    import entrenamiento.p04_evaluacion as modulo
    (tmp_path / 'test_final.json').write_text('{"sentinela":true}', encoding='utf-8')
    monkeypatch.setattr(modulo, 'SALIDAS', tmp_path)
    def prohibido(*args, **kwargs):
        raise AssertionError('No debe volver a abrir test ni cargar el modelo')
    monkeypatch.setattr(modulo, 'cargar', prohibido)
    monkeypatch.setattr(modulo.joblib, 'load', prohibido)
    assert modulo.evaluar_una_vez() == {'sentinela': True}


def test_recuperacion_sin_artefacto_solo_usa_train(monkeypatch, tmp_path):
    import entrenamiento.p05_inferencia as modulo
    original = cargar_modelo()
    real_cargar = modulo.cargar
    particiones = []
    def registrar(particion):
        particiones.append(particion)
        return real_cargar(particion)
    monkeypatch.setattr(modulo, 'MODELO', tmp_path / 'ausente.joblib')
    monkeypatch.setattr(modulo, 'cargar', registrar)
    recuperado = modulo.cargar_modelo()
    assert particiones == ['train']
    mensaje = 'Estoy feliz por mi proyecto'
    esperado = predecir(original, mensaje)['scores']
    obtenido = predecir(recuperado, mensaje)['scores']
    assert obtenido == pytest.approx(esperado)
    assert not (tmp_path / 'ausente.joblib').exists()


def test_artefacto_alterado_no_se_deserializa(monkeypatch, tmp_path):
    import entrenamiento.p05_inferencia as modulo
    falso = tmp_path / 'modelo.joblib'
    falso.write_bytes(b'contenido alterado')
    monkeypatch.setattr(modulo, 'MODELO', falso)
    with pytest.raises(RuntimeError, match='Artefacto alterado'):
        modulo.cargar_modelo()
