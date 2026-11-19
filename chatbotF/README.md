# Chatbot Emocional Académico (Proyecto Final de Machine Learning)

Aplicación web local en Python (**Flask + scikit-learn**) que clasifica señales de **alegría**, **tristeza**, **enojo**, **ansiedad / nerviosismo** y **neutralidad** en mensajes escritos en español, entrenada desde cero sin modelos preentrenados ni APIs externas.

> **Aviso importante:** Este sistema es un proyecto académico universitario y **no es una herramienta clínica ni diagnóstica**. Ante cualquier señal de posible suicidio o autolesión, el sistema prioriza una respuesta empática de orientación hacia apoyo humano y muestra la **Línea 113 Salud del MINSA (Perú), Opción 5 (Salud mental y psicología)**, gratuita y disponible las 24 horas del día, los 365 días del año.

---

## Cómo arrancar

Con el entorno virtual y las dependencias ya instaladas, basta con estos tres pasos:

1. Activa el entorno virtual:
   ```bash
   source .venv/bin/activate
   ```
2. Ejecuta el servidor local:
   ```bash
   python app.py
   ```
3. Abre en tu navegador la dirección exacta impresa en la terminal (`http://127.0.0.1:5000`).

---

## Qué incluye la interfaz

- **Conversación:** chat con texto grande pensado para proyectar en clase, probabilidades por emoción, términos que influyeron en la predicción, solicitud de contexto cuando la señal es débil y respuesta prioritaria ante mensajes de riesgo.
- **Aprendizaje:** formulario explícito para escribir o copiar una frase, elegir su emoción y guardarla localmente, reentrenar una nueva versión del modelo, ver el cambio real en validación, borrar ejemplos y volver al modelo base en un clic.
- **Laboratorio:** panel con los conteos por procedencia, la comparación de los seis modelos en validación, la comparación controlada de bigramas frente a trigramas, las métricas por clase, las matrices de confusión, las métricas con abstención y el diagnóstico exploratorio.

---

## Instalación en un equipo nuevo

Si necesitas recrear el entorno desde cero en otra computadora:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

---

El archivo de versiones fijadas corresponde al entorno de Python 3.14.2 en el que se reprodujo el entrenamiento. `requirements.txt` documenta los rangos de dependencias directas.

## Reproducir el entrenamiento y las pruebas

El modelo base ya está entrenado y guardado en `modelos/modelo_base.joblib`, listo para usar. Conserva ese archivo y `modelos/resultados_experimentos.json` en Git. El entrenamiento lee `train.csv`, `validacion_go.csv` y `validacion_sintetica.csv` de `../datos de entrenamiento/preparados/`. Utiliza una copia de trabajo para regenerar el modelo y las métricas, porque el comando de entrenamiento reemplaza los artefactos de esa copia:

```bash
source .venv/bin/activate
python -m entrenamiento.p08_experimentos
```

Para ejecutar todas las pruebas automatizadas (datos, preprocesamiento, seguridad, inferencia, ciclo de reentrenamiento/restauración y formato de comentarios):

```bash
source .venv/bin/activate
python -m unittest discover -s pruebas -v
```

---

## Datos y archivos locales

`train.csv` contiene 23.689 ejemplos: 15.381 de [GoEmotions traducido al español](https://huggingface.co/datasets/mrm8488/go_emotions-es-mt), 8.068 sintéticos generados con Claude y 240 sintéticos del corpus inicial. La validación de GoEmotions tiene 1.899 ejemplos y la sintética tiene 1.932. Los CSV incluyen la procedencia en la columna `origen`. Los datos derivados de GoEmotions conservan su atribución y [licencia Apache 2.0](../datos%20de%20entrenamiento/fuentes/go_emotions_es/LICENCIA_GOEMOTIONS.txt).

Para respaldar las fuentes completas de la entrega, conserva por separado el CSV original de Claude y los tres Parquet de GoEmotions. El modelo base de este proyecto se puede reproducir con los CSV preparados. Las correcciones personales, el modelo activo, su estado y cualquier salida de test futuro son archivos locales excluidos por `.gitignore`.

## Evaluación final sobre un conjunto test nuevo (una sola vez)

Cuando dispongas del archivo CSV de prueba final (con columnas `texto` y `etiqueta`), evalúa el **modelo base congelado** una sola vez ejecutando:

```bash
source .venv/bin/activate
python -m entrenamiento.p10_evaluar_test --test /ruta/al/test_final.csv
```

El comando conserva en `modelos/evaluacion_final/`:
- `metricas_test.json`: macro-F1, exactitud, precisión, recall, F1 por clase, matriz de confusión, cobertura/exactitud selectiva con abstención y verificación de solape con entrenamiento.
- `predicciones_test.csv`: predicción fila por fila para auditoría posterior.

El script bloquea ejecuciones repetidas por defecto para respetar la regla de usar el test final una sola vez (si necesitas sobrescribirlo de manera explícita, pasa `--forzar`).

---

## Estructura del proyecto

```text
ChatbotGemini/
├── app.py                              # servidor Flask (rutas / y /api/*)
├── templates/
│   └── index.html                      # interfaz completa en un único archivo HTML
├── entrenamiento/
│   ├── p01_config.py                   # rutas, clases, semilla y umbrales
│   ├── p02_datos.py                    # carga, conteos por origen y auditoría de fugas
│   ├── p03_preprocesamiento.py         # limpieza de texto preservando negaciones y ñ
│   ├── p04_seguridad.py                # capa prioritaria de crisis y Línea 113 Opción 5
│   ├── p05_modelos.py                  # referencia simple y candidatos comparados
│   ├── p06_metricas.py                 # macro-F1, matriz de confusión y abstención
│   ├── p07_inferencia.py               # predicción, probabilidades y pedido de contexto
│   ├── p08_experimentos.py             # comparación controlada y guardado del modelo base
│   ├── p09_reentrenamiento.py          # correcciones locales, reentrenamiento y restauración
│   └── p10_evaluar_test.py             # evaluación única del modelo congelado en test
├── datos/                              # correcciones locales que crea la app
├── modelos/                            # modelo base, modelo activo y métricas JSON
├── pruebas/                            # pruebas unitarias y de integración Flask
└── requirements.txt                    # dependencias de Python
```

---

## Documentación complementaria

- **Informe de resultados:** consulta [el informe](../documentacionCF/INFORME_RESULTADOS.md) para ver todas las tablas de validación, matrices de confusión, cobertura con abstención y análisis de errores.
- **Explicación y guía de defensa oral:** consulta [la guía de defensa](../documentacionCF/Explicacion.md) para revisar la relación con los apuntes del curso (semana y sección), cada archivo y función paso a paso, y preguntas probables de la profesora con respuestas listas para responder en voz alta.

## Carpetas de la entrega

Coloca `datos de entrenamiento/` junto a la carpeta del chatbot. Si el corpus está en otro lugar, indica su ruta con `CHATBOT_DATA_DIR`. Usa `CHATBOT_DOCS_DIR` para indicar una ubicación distinta para los informes descargables.
La explicación y el informe de F están en `documentacionCF/`, también junto a este proyecto.
