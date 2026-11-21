# pruebas del pipeline de datos, preprocesamiento, seguridad, inferencia y estilo
import ast
import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

import joblib

from entrenamiento.p01_config import (
    RAIZ_PROYECTO,
    RUTA_MODELO_BASE,
    RUTA_RESULTADOS,
)
from entrenamiento.p02_datos import cargar_particiones
from entrenamiento.p03_preprocesamiento import normalizar_texto, preparar_texto_modelo
from entrenamiento.p04_seguridad import evaluar_riesgo
from entrenamiento.p07_inferencia import analizar_mensaje
from entrenamiento.p10_evaluar_test import evaluar_conjunto_test

TILDES_O_MAYUS = re.compile(r"[ÁÉÍÓÚÜÑáéíóúüñA-Z]")
SIMBOLOS_DECORATIVOS = re.compile(r"(=|\*|--|###|->|<-|~|─|—|•)")


class TestPipelineML(unittest.TestCase):
    def test_conteos_y_cero_fugas_en_particiones(self):
        # verifica los conteos exactos entregados y ausencia de fugas entre particiones
        train, val_go, val_sin, resumen = cargar_particiones()
        self.assertEqual(len(train), 26469)
        self.assertEqual(len(val_go), 1899)
        self.assertEqual(len(val_sin), 1932)
        self.assertEqual(resumen["train"]["por_origen"]["go_emotions_es_mt"], 15381)
        self.assertEqual(resumen["train"]["por_origen"]["sintetico_claude"], 8068)
        self.assertEqual(resumen["train"]["por_origen"]["sintetico_asistido_IA"], 240)
        self.assertEqual(resumen["train"]["por_origen"]["curado_situaciones_implicitas"], 2780)
        for clave, valor in resumen["auditoria_fugas"].items():
            self.assertEqual(valor, 0, f"Fuga detectada en {clave}")

    def test_preprocesamiento_preserva_negaciones_y_enie(self):
        # comprueba minusculas, limpieza de tildes, conservacion de enie y negaciones
        salida = normalizar_texto("¡Nooooo me fue bien mañana en programación 123!")
        self.assertEqual(salida, "no me fue bien mañana en programacion")
        self.assertIn("no", salida.split())
        negada = preparar_texto_modelo("No estoy enojado, solo cansado")
        self.assertEqual(negada, "no neg_estoy neg_enojado solo cansado")
        negada_pron = preparar_texto_modelo("No me siento feliz")
        self.assertEqual(negada_pron, "no me neg_siento neg_feliz")

    def test_seguridad_detecta_riesgo_y_evita_falsos_positivos(self):
        # valida deteccion de crisis y tolerancia a expresiones coloquiales inofensivas
        frases_riesgo = [
            "Ya no quiero vivir mas",
            "Estoy pensando en quitarme la vida",
            "A veces quiero hacerme daño",
            "Me voy a cortar las venas",
        ]
        for frase in frases_riesgo:
            self.assertTrue(evaluar_riesgo(frase)["es_riesgo"], frase)

        frases_seguras = [
            "Me muero de risa con ese video",
            "Me muero de sueño después de estudiar",
            "Me muero de hambre hoy",
            "Solo estaba matando el tiempo esperando el bus",
        ]
        for frase in frases_seguras:
            self.assertFalse(evaluar_riesgo(frase)["es_riesgo"], frase)

    def test_inferencia_prioriza_riesgo_y_entrega_probabilidades(self):
        # confirma prioridad de crisis con linea 113 opcion 5 y suma de probabilidades
        modelo = joblib.load(RUTA_MODELO_BASE)

        crisis = analizar_mensaje(modelo, "No quiero seguir viviendo, quiero hacerme daño")
        self.assertTrue(crisis["es_riesgo"])
        self.assertIsNone(crisis["etiqueta"])
        self.assertIn("113", crisis["respuesta"])
        self.assertIn("Opción 5", crisis["respuesta"])
        self.assertIn("no es una herramienta clínica", crisis["respuesta"].lower())

        alegre = analizar_mensaje(modelo, "No puedo estar más feliz con esta noticia")
        self.assertFalse(alegre["es_riesgo"])
        self.assertEqual(alegre["etiqueta"], "alegria")
        self.assertAlmostEqual(sum(alegre["probabilidades"].values()), 1.0, places=2)

        neutral_no = analizar_mensaje(modelo, "No tengo clases mañana y luego iré al banco")
        self.assertEqual(neutral_no["etiqueta"], "neutral")

        contexto = analizar_mensaje(modelo, "No me fue del todo bien en el examen")
        self.assertTrue(contexto["pide_contexto"])
        self.assertEqual(contexto["estado"], "pedir_contexto")

    def test_resultados_experimentos_coherentes(self):
        # comprueba ventaja global de trigramas y del modelo ganador hibrido
        with RUTA_RESULTADOS.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
        dif = datos["comparacion_bigramas_vs_trigramas"]["diferencia_trigramas_menos_bigramas"]
        self.assertGreater(dif["macro_f1_sintetica"], 0.0)
        self.assertGreater(dif["macro_f1_combinada"], 0.0)
        self.assertGreater(dif["promedio_macro_f1"], 0.0)

        modelos = {m["clave"]: m["metricas"] for m in datos["modelos"]}
        ganador = modelos[datos["modelo_ganador_clave"]]
        trigramas = modelos["tfidf_lr_trigramas"]
        self.assertGreater(ganador["validacion_go"]["macro_f1"], trigramas["validacion_go"]["macro_f1"])
        self.assertGreater(ganador["promedio_macro_f1"], trigramas["promedio_macro_f1"])

        for caso in datos["diagnostico_exploratorio"]["casos"]:
            self.assertFalse(caso["en_entrenamiento"])

    def test_evaluacion_final_una_sola_vez_en_carpeta_temporal(self):
        # prueba el comando de test congelado con un archivo temporal sin tocar modelos reales
        with tempfile.TemporaryDirectory() as tmp:
            ruta_test = Path(tmp) / "test_demo.csv"
            dir_out = Path(tmp) / "salida"
            with ruta_test.open("w", encoding="utf-8", newline="") as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=["texto", "etiqueta", "origen"])
                escritor.writeheader()
                escritor.writerow({"texto": "Estoy muy feliz por el resultado", "etiqueta": "alegria", "origen": "demo"})
                escritor.writerow({"texto": "Me da mucha rabia esta injusticia", "etiqueta": "enojo", "origen": "demo"})
                escritor.writerow({"texto": "La clase empieza a las ocho", "etiqueta": "neutral", "origen": "demo"})
                escritor.writerow({"texto": "Me siento triste y desanimado hoy", "etiqueta": "tristeza", "origen": "demo"})
                escritor.writerow({"texto": "Estoy nervioso por el examen de mañana", "etiqueta": "ansiedad", "origen": "demo"})

            reporte = evaluar_conjunto_test(ruta_test=ruta_test, dir_salida=dir_out)
            self.assertEqual(reporte["metricas_directas"]["total"], 5)
            self.assertTrue((dir_out / "metricas_test.json").exists())
            self.assertTrue((dir_out / "predicciones_test.csv").exists())

            # una segunda ejecucion sin forzar debe bloquearse para proteger el test unico
            with self.assertRaises(RuntimeError):
                evaluar_conjunto_test(ruta_test=ruta_test, dir_salida=dir_out)

    def test_reglas_de_comentarios_en_codigo_python(self):
        # verifica minusculas, sin tildes, sin simbolos decorativos y maximo dos lineas seguidas
        archivos = [RAIZ_PROYECTO / "app.py"]
        archivos.extend(sorted((RAIZ_PROYECTO / "entrenamiento").glob("*.py")))
        archivos.extend(sorted((RAIZ_PROYECTO / "pruebas").glob("*.py")))

        for ruta in archivos:
            codigo = ruta.read_text(encoding="utf-8")
            arbol = ast.parse(codigo)
            for nodo in ast.walk(arbol):
                if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.assertIsNone(
                        ast.get_docstring(nodo),
                        f"Bloque docstring detectado en {ruta.name}",
                    )

            lineas = codigo.splitlines()
            seguidas = 0
            for num, linea in enumerate(lineas, start=1):
                recortada = linea.strip()
                if recortada.startswith("#"):
                    seguidas += 1
                    self.assertLessEqual(
                        seguidas,
                        2,
                        f"Mas de dos lineas de comentario seguidas en {ruta.name}:{num}",
                    )
                    contenido = recortada[1:].strip()
                    self.assertIsNone(
                        TILDES_O_MAYUS.search(contenido),
                        f"Comentario con mayuscula o tilde en {ruta.name}:{num}: {contenido}",
                    )
                    self.assertIsNone(
                        SIMBOLOS_DECORATIVOS.search(contenido),
                        f"Comentario con simbolo decorativo en {ruta.name}:{num}: {contenido}",
                    )
                else:
                    seguidas = 0


if __name__ == "__main__":
    unittest.main()
