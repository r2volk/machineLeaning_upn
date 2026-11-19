# Guía de la ampliación y del aprendizaje local

## Explicación para clase

Ánimo representa un mensaje mediante TF-IDF. Este proceso crea columnas para palabras y parejas de palabras aprendidas únicamente de los textos de entrenamiento. La regresión logística aprende un peso para cada columna y cada emoción. Cuando llega un mensaje, se transforma con el vocabulario ya aprendido y el clasificador calcula puntuaciones para las cinco clases. Una puntuación no es un diagnóstico ni mide la intensidad de una emoción.

El modelo anterior usaba 240 textos sintéticos de entrenamiento. Para la ampliación se agregaron ejemplos sintéticos y textos traducidos de GoEmotions, manteniendo el algoritmo. Cada candidato se entrenó desde cero con su conjunto de entrenamiento; se compararon con macro-F1 en validación. El ganador usó las dos fuentes nuevas sin ponderación de clases. Luego se evaluó una sola vez sobre la partición test de GoEmotions. Los datos y el test originales se conservaron aparte. La explicación completa de las cifras y sus límites está en [el informe de ampliación](INFORME_AMPLIACION.md).

## Archivos nuevos, de arriba abajo

1. `entrenamiento/p07_datos_ampliados.py`: `cargar_go` lee una partición y mantiene solo mensajes con una etiqueta compatible; `dividir_sintetico` reúne grupos de paráfrasis y algunos inicios repetidos; `depurar` retira duplicados, contradicciones y coincidencias con validación; `preparar` guarda los CSV de entrenamiento y validación. No lee el test.
2. `entrenamiento/p08_comparacion_ampliada.py`: `modelo_nuevo` crea el pipeline conocido; `medir` calcula macro-F1, exactitud y F1 por clase; `comparar` entrena candidatos, los compara con ambas validaciones y elige por macro-F1 de Go. Guarda modelo, decisión y huellas.
3. `entrenamiento/p09_test_ampliado.py`: `evaluar` verifica modelo y datos congelados, crea una reserva exclusiva, carga el test una vez, descarta duplicados exactos prefijados y calcula las métricas con un lote de puntuaciones. Si ya existe la reserva, se detiene. No borres la reserva para repetir el test.
4. `entrenamiento/p10_aprendizaje_local.py`: `guardar_ejemplo` comprueba emoción, longitud, reglas de riesgo y duplicados; `entrenar_local` reconstruye el modelo con datos anteriores más correcciones; `cargar_actual` carga la versión local o el modelo ampliado; `borrar_local` borra ejemplos y modelo personalizado; `estado_local` informa del estado sin devolver las frases.
5. `datos de entrenamiento/fuentes/`: CSV sintético de Claude y archivos originales de GoEmotions español. `datos de entrenamiento/preparados/`: particiones procesadas de entrenamiento y validación.
6. `entrenamiento/artefactos/modelo_ampliado.joblib`: modelo elegido. `entrenamiento/resultados/ampliacion/`: selección, evaluación única y reserva del test.
7. `app.py`: rutas HTTP que llaman a los módulos anteriores; la prioridad de las reglas de seguridad se conserva. `index.html`: pestañas y controles para utilizar las funciones.
8. `tests/test_aprendizaje.py`: verifica el guardado voluntario, duplicados, reglas de riesgo, entrenamiento y restauración en una carpeta temporal.

## Cómo enseñar nuevos ejemplos

En la pestaña **Aprendizaje**, escribe una frase o copia expresamente tu último mensaje, selecciona la emoción correcta y pulsa **Guardar ejemplo en este equipo**. Repite con varios ejemplos variados, evitando nombres y datos sensibles. Pulsa **Entrenar nueva versión** cuando quieras incorporar lo guardado. Se vuelven a ajustar TF-IDF y regresión logística **desde cero** usando la combinación original de entrenamiento más tus ejemplos; el entrenamiento incremental no se utiliza. El modelo se activa y permanece en el equipo al reiniciar.

Añadir un ejemplo no garantiza que se aprenda correctamente ni que mejore la clasificación general. El resultado de validación que aparece es orientativo: consultarlo repetidamente y decidir a partir de él puede sesgarlo. Las cifras de test del informe se refieren exclusivamente al modelo ampliado original, no al personalizado. La verificación seria de una nueva versión necesita datos independientes nuevos, suficientes por clase y separados antes de ajustarla.

La conversación normal no se guarda ni se usa para entrenar por sí sola. Los ejemplos elegidos quedan en `.animo_local/ejemplos.json` y el modelo personalizado en la misma carpeta, excluida de Git. Las reglas de riesgo intentan impedir guardar mensajes detectados como sensibles, pero pueden fallar. Pulsa **Eliminar ejemplos y restaurar** para borrar las correcciones y el modelo que aprendió de ellas; se vuelve al modelo ampliado. El historial de conversación visible es independiente y se borra al cerrar o reiniciar la sesión.

## Fuentes y reproducibilidad

La fuente es [GoEmotions en español](https://huggingface.co/datasets/mrm8488/go_emotions-es-mt), traducción automática de GoEmotions; el archivo original incluido declara licencia Apache 2.0. Las etiquetas se adaptaron de forma conservadora y se descartaron mensajes con varias etiquetas. La traducción fue realizada con herramientas de terceros antes del proyecto; no se usa ningún modelo preentrenado en el entrenamiento ni en la aplicación. El otro CSV fue generado sintéticamente con Claude y no contiene anotaciones humanas verificadas.

Para ejecutar pruebas: `python -m pytest`. Para preparar datos originales desde Parquet en una copia de trabajo sin test reservado se necesita instalar las versiones de `requirements-lock.txt` y `requirements-preparacion.txt`; la aplicación y el aprendizaje local usan los CSV preparados y no lo necesitan. Los pasos de preparación, comparación y test son `python -m entrenamiento.p07_datos_ampliados`, `python -m entrenamiento.p08_comparacion_ampliada` y `python -m entrenamiento.p09_test_ampliado`, respectivamente. En esta entrega el test ya está reservado: los scripts se detienen para proteger la evaluación y no deben forzarse. Para un nuevo experimento hace falta una partición de test realmente nueva, un protocolo propio y registrar su evaluación por separado.

## Preguntas frecuentes

**¿Por qué no se sigue entrenando el mismo archivo?** El vocabulario TF-IDF también debe aprender palabras nuevas. La forma sencilla y reproducible con esta regresión logística es reconstruir vectorizador y clasificador desde cero con los datos anteriores y las correcciones.

**¿Por qué no aprende solo al conversar?** La predicción del modelo no es una etiqueta correcta garantizada. Aprender de sus propias respuestas puede reforzar sus errores; además, guardar mensajes sin pedirlo sería inapropiado. Aquí la persona elige la etiqueta y guarda expresamente cada ejemplo.

**¿Por qué eligieron macro-F1?** Porque da el mismo peso a cada emoción, mientras la exactitud puede verse alta si casi todos los mensajes son neutrales. Aun así, con solo ocho ejemplos de nerviosismo en validación, la elección es inestable.

**¿El modelo es más preciso?** La comparación en la validación Go mejoró de 0,232 a 0,608 de macro-F1. Eso es evidencia en esa fuente concreta, no una garantía para personas reales. En el test nuevo, el recall de nerviosismo fue 1 de 12: hay una debilidad importante.

**¿Qué pasa con las frases de riesgo?** Se aplican reglas antes de clasificar y se ofrece apoyo humano. En aprendizaje se rechazan las frases que esas reglas detectan; las reglas pueden fallar. El sistema no es clínico.

## Qué conservar para volver a entrenar

Conserva `datos de entrenamiento/preparados/` con los tres CSV que leen los scripts y `datos de entrenamiento/fuentes/` con el CSV original de 10.000 frases generado con Claude y los Parquet de entrenamiento y validación. El corpus original de E está en `datos de entrenamiento/corpus_original_E/`. Guarda también ambos archivos `.joblib` de `entrenamiento/artefactos/`, los resultados y las marcas de test para conservar la entrega evaluada. Las correcciones personales de `.animo_local/` quedan fuera del repositorio.

Para regenerar la ampliación, trabaja en una **copia nueva** y deja fuera de esa copia `entrenamiento/resultados/ampliacion/` y `entrenamiento/artefactos/modelo_ampliado.joblib`. Conserva el modelo original `modelo.joblib` para la comparación. Copia también la carpeta de datos a un lugar de trabajo separado para que la preparación escriba allí. Exporta `CHATBOT_DATA_DIR` con la ruta absoluta de esa copia. Activa un entorno Python 3.14.2 e instala las dependencias:

```sh
python -m pip install -r requirements-lock.txt -r requirements-preparacion.txt
python -m entrenamiento.p07_datos_ampliados
python -m entrenamiento.p08_comparacion_ampliada
```

Esto regenera datos, selección y modelo. Usa un test nuevo para cualquier evaluación final. El Parquet histórico de test puede quedarse en el respaldo local; estos pasos solo requieren los Parquet de entrenamiento y validación. La [licencia de GoEmotions](../datos%20de%20entrenamiento/fuentes/go_emotions_es/LICENCIA_GOEMOTIONS.txt) acompaña los datos que derivan de esa fuente. Para conservar el archivo exacto de la entrega con su huella SHA-256, guarda y usa el artefacto original junto con su selección.
