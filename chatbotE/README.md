# Ánimo

Chatbot académico local que identifica señales de alegría, tristeza, enojo, ansiedad o nerviosismo y neutralidad en mensajes escritos. Puede equivocarse y **no es una herramienta clínica**.

## Ejecutar

1. Activa el entorno: `source .venv/bin/activate`
2. Ejecuta: `python app.py`
3. Abre la dirección exacta que imprime la terminal, por ejemplo **http://127.0.0.1:5000**.

Para preparar el entorno en otro equipo, consulta [instalación](../documentacionCE/REPRODUCIBILIDAD.md).

## Cómo usarlo

- **Conversación**: escribe un mensaje y observa la interpretación. Las reglas de seguridad tienen prioridad sobre el modelo.
- **Laboratorio**: consulta por separado los resultados de la ampliación y del experimento original.
- **Aprendizaje**: guarda voluntariamente frases con su emoción correcta y pulsa «Entrenar nueva versión». El modelo se vuelve a entrenar desde cero con los datos anteriores y los ejemplos locales; no aprende automáticamente de cada mensaje. Puedes eliminarlos y restaurar el modelo ampliado.

Los ejemplos voluntarios y el modelo personalizado quedan en `.animo_local/`, excluido de Git. La conversación normal no se guarda automáticamente. Evita introducir datos sensibles. El aprendizaje personalizado puede empeorar el modelo y no cuenta con un test independiente nuevo.

## Decisiones técnicas

Todo el código Python de aprendizaje automático está en `entrenamiento/`. Se mantiene TF-IDF con unigramas y bigramas, seguido de regresión logística, porque es sencillo de explicar y se relaciona con el curso. Se entrena desde cero, sin modelos preentrenados ni servicios externos durante el funcionamiento. Las respuestas del chat son textos redactados, no las genera el clasificador.

El modelo ampliado combina datos sintéticos con mensajes de GoEmotions traducidos automáticamente al español, con etiquetas adaptadas de forma conservadora. La traducción ya estaba hecha en la fuente con herramientas externas; el proyecto no ejecuta esos modelos de traducción. «Ansiedad» aproxima la etiqueta de nerviosismo, no es un diagnóstico.

Se eligió entre candidatos mediante macro-F1 de validación y se evaluó al ganador una sola vez con el test separado de la fuente. Los resultados son limitados: en el test solo había 12 ejemplos de nerviosismo y el modelo identificó uno. La exactitud global está influida por la abundancia de ejemplos neutrales. Lee [el informe ampliado](../documentacionCE/INFORME_AMPLIACION.md) antes de interpretar las cifras.

Para explicar el procedimiento, las fuentes, el reentrenamiento y cómo reproducir el análisis, consulta [la guía de ampliación](../documentacionCE/APRENDIZAJE.md). Los resultados históricos se conservan en [el informe original](../documentacionCE/INFORME_RESULTADOS.md). Las reglas de seguridad y sus limitaciones se mantienen; Ánimo no sustituye atención profesional ni servicios de emergencia.

## Carpetas de la entrega

Coloca `datos de entrenamiento/` junto a la carpeta del chatbot. Si el corpus está en otro lugar, indica su ruta con `CHATBOT_DATA_DIR`. Usa `CHATBOT_DOCS_DIR` para indicar una ubicación distinta para los informes descargables.
Las guías y los informes del proyecto E están en `documentacionCE/`, también junto a este proyecto.
