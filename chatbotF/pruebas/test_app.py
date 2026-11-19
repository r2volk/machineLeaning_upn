# pruebas de la aplicacion flask y del ciclo completo de correcciones
import unittest

from app import crear_app
from entrenamiento.p01_config import CLASES
from entrenamiento.p09_reentrenamiento import limpiar_correcciones, restaurar_modelo_base


class TestAplicacionFlask(unittest.TestCase):
    def setUp(self):
        # asegura un estado limpio sobre el modelo base antes de cada prueba
        limpiar_correcciones()
        restaurar_modelo_base()
        self.app = crear_app()
        self.cliente = self.app.test_client()

    def tearDown(self):
        # restaura el modelo base y limpia las correcciones creadas en la prueba
        limpiar_correcciones()
        restaurar_modelo_base()

    def test_interfaz_sin_degradados_y_con_rutas_relativas(self):
        # comprueba que la raiz sirve index.html limpio y sin degradados css
        resp = self.cliente.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("/api/chat", html)
        self.assertIn("/api/correcciones", html)
        self.assertIn("/api/reentrenar", html)
        self.assertIn("/api/restaurar", html)
        self.assertIn("/api/laboratorio", html)
        self.assertNotIn("linear-gradient", html.lower())
        self.assertNotIn("radial-gradient", html.lower())
        self.assertNotIn("conic-gradient", html.lower())
        self.assertIn("Opción 5", html)

    def test_chat_normal_y_precaucion(self):
        # verifica respuestas emocionales y atencion prioritaria de crisis
        r_ok = self.cliente.post(
            "/api/chat",
            json={"mensaje": "No puedo estar más feliz con esta noticia"},
        )
        self.assertEqual(r_ok.status_code, 200)
        data_ok = r_ok.get_json()
        self.assertFalse(data_ok["es_riesgo"])
        self.assertEqual(data_ok["etiqueta"], "alegria")

        r_crisis = self.cliente.post(
            "/api/chat",
            json={"mensaje": "Ya no quiero vivir, pienso en quitarme la vida"},
        )
        self.assertEqual(r_crisis.status_code, 200)
        data_crisis = r_crisis.get_json()
        self.assertTrue(data_crisis["es_riesgo"])
        self.assertIn("113", data_crisis["respuesta"])
        self.assertIn("Opción 5", data_crisis["respuesta"])

    def test_bloqueo_de_riesgo_y_ciclo_guardar_entrenar_restaurar(self):
        # comprueba rechazo de riesgo y ciclo completo de aprendizaje reversible
        r_riesgo = self.cliente.post(
            "/api/correcciones",
            json={"texto": "Me quiero quitar la vida hoy", "etiqueta": "tristeza"},
        )
        self.assertEqual(r_riesgo.status_code, 400)

        # guarda una frase con modismo peruano que el modelo base marcaba neutral
        frase = "No esperaba aprobar pero saqué veinte"
        r_save = self.cliente.post(
            "/api/correcciones",
            json={"texto": frase, "etiqueta": "alegria"},
        )
        self.assertEqual(r_save.status_code, 201)
        corr_id = r_save.get_json()["correccion"]["id"]

        # entrena nueva version e inspecciona cambio en validacion y prediccion
        r_train = self.cliente.post("/api/reentrenar")
        self.assertEqual(r_train.status_code, 200)
        estado_nuevo = r_train.get_json()["estado_modelo"]
        self.assertTrue(estado_nuevo["version_activa"].startswith("reentrenado_v"))
        self.assertEqual(estado_nuevo["correcciones_incorporadas"], 1)
        self.assertIn("macro_f1_go", estado_nuevo["delta_validacion"])

        r_chat_post = self.cliente.post("/api/chat", json={"mensaje": frase})
        self.assertEqual(r_chat_post.status_code, 200)
        self.assertEqual(r_chat_post.get_json()["etiqueta"], "alegria")

        # restaura al modelo base y elimina la correccion local
        r_rest = self.cliente.post("/api/restaurar")
        self.assertEqual(r_rest.status_code, 200)
        self.assertEqual(r_rest.get_json()["estado_modelo"]["version_activa"], "base")

        r_del = self.cliente.delete(f"/api/correcciones/{corr_id}")
        self.assertEqual(r_del.status_code, 200)
        self.assertEqual(len(r_del.get_json()["correcciones"]), 0)

    def test_laboratorio_expone_metricas_calculadas(self):
        # valida que el laboratorio devuelva los seis modelos y las particiones
        resp = self.cliente.get("/api/laboratorio")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data["experimentos"]["modelos"]), 6)
        self.assertEqual(data["experimentos"]["resumen_datos"]["train"]["total"], 23689)
        for clase in CLASES:
            self.assertIn(clase, data["nombres_clases"])


if __name__ == "__main__":
    unittest.main()
