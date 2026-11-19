# Informe de resultados: ampliación del modelo Ánimo

## Qué se hizo

Se conservó el experimento original y se realizó uno nuevo, con el mismo tipo de clasificador: **TF-IDF de palabras y bigramas más regresión logística**. La elección se hizo con validación; después se congeló el modelo y se ejecutó una sola evaluación sobre el test separado de GoEmotions. El modelo y los resultados originales permanecen guardados por separado.

## Datos

- Entrenamiento: **23.689** frases tras limpiar duplicados exactos y textos con etiquetas contradictorias. De ellas, 240 son sintéticas del proyecto original, 8.068 son sintéticas generadas con Claude y 15.381 proceden de GoEmotions traducido automáticamente al español.
- Validación GoEmotions: **1.899** frases. Se usó la partición `validation` original de la fuente y solo mensajes con una etiqueta compatible.
- Validación sintética: **1.932** frases generadas por Claude. Se mantuvieron juntas las paráfrasis declaradas y los grupos que comparten las primeras cuatro palabras y la etiqueta. Este agrupamiento reduce algunas similitudes, pero no las elimina: el resultado sintético puede ser optimista.
- Test GoEmotions: **1.908** frases evaluadas. Se partió de mensajes de la partición `test` original con una sola etiqueta compatible y texto utilizable; se retiraron 33 coincidencias o duplicados exactos, incluidas coincidencias con entrenamiento o validación. El test se reservó y evaluó después de elegir el modelo.

El conjunto de GoEmotions contiene mensajes de Reddit traducidos desde el inglés, no una muestra de estudiantes universitarios peruanos. Sus etiquetas son las de la fuente, adaptadas automáticamente: `joy` → alegría, `sadness` → tristeza, `anger` → enojo, `nervousness` → ansiedad y `neutral` → neutral. Nerviosismo es solo una aproximación a la categoría del proyecto, no evidencia de un trastorno. Se descartaron mensajes con varias etiquetas y los de las demás categorías; por tanto, los resultados no aplican a todos los mensajes de GoEmotions.

La traducción fue creada por terceros antes de este proyecto y puede contener errores; los ejemplos sintéticos fueron generados con IA y no equivalen a etiquetas de personas reales. Ambas fuentes pueden tener sesgos, frases artificiales y etiquetas discutibles. La procedencia se conserva en los CSV preparados.

## Comparación en validación

La tabla usa predicciones forzadas para las cinco clases, sin aplicar la abstención del chat. Las cifras proceden de `entrenamiento/resultados/ampliacion/seleccion.json`. Macro-F1 da el mismo peso a cada clase; exactitud es la proporción total de aciertos.

| Entrenamiento del candidato | Macro-F1 Go | Exactitud Go | Macro-F1 sintético |
|---|---:|---:|---:|
| Modelo original | 0,232 | 0,479 | 0,507 |
| Original + sintético Claude | 0,232 | 0,349 | 0,976 |
| Original + GoEmotions | 0,540 | 0,884 | 0,401 |
| **Original + GoEmotions + Claude** | **0,608** | **0,885** | **0,961** |
| Los tres conjuntos + pesos de clase equilibrados | 0,591 | 0,873 | 0,981 |

Se eligió **la combinación sin pesos equilibrados** por tener el mayor macro-F1 en validación Go, no por su resultado en test. La validación Go contiene solo **8** ejemplos de nerviosismo, por lo que diferencias pequeñas de macro-F1 son inestables. La comparación entre el modelo original y el ampliado en esta misma validación sugiere mejor adaptación a esa fuente, pero no demuestra una mejora equivalente en estudiantes reales. Los valores sintéticos altos no deben presentarse como precisión general.

## Test final del modelo elegido

Resultados guardados en `entrenamiento/resultados/ampliacion/test_final.json`, calculados con una única llamada por lote a `predict_proba` y etiquetas sin abstención:

| Medida | Resultado |
|---|---:|
| Macro-F1 | 0,528 |
| Exactitud | 0,873 |
| Ejemplos | 1.908 |

| Clase | Ejemplos | F1 | Recall |
|---|---:|---:|---:|
| Alegría | 92 | 0,639 | 0,500 |
| Ansiedad / nerviosismo | 12 | 0,143 | 0,083 |
| Enojo | 129 | 0,398 | 0,279 |
| Neutral | 1.575 | 0,928 | 0,980 |
| Tristeza | 100 | 0,533 | 0,400 |

**La limitación principal:** 1 de los 12 mensajes etiquetados como nerviosismo fue reconocido como ansiedad; los otros 11 se predijeron como neutrales. Hay pocos ejemplos para estimar esa clase con precisión, y el modelo también confunde con neutral muchos ejemplos de enojo y tristeza. La exactitud de 87,3 % está muy influida por los 1.575 mensajes neutrales. No debe interpretarse como «87,3 % de precisión para cada emoción».

La interfaz puede abstenerse cuando el vocabulario, el score o el margen son insuficientes. Con el umbral heredado de 0,30 y margen 0,10, en este test la cobertura fue 0,974 y la exactitud entre aceptados 0,882. El umbral se heredó del experimento original y **no está calibrado para la nueva fuente**; las puntuaciones tampoco son probabilidades clínicas.

## Qué queda pendiente

Hace falta reunir mensajes reales en español del público objetivo, con consentimiento, sin datos identificables y con etiquetas revisadas por más de una persona. Debe haber suficientes ejemplos de cada emoción, especialmente ansiedad, y particiones independientes por persona o situación. Reservar un test nuevo antes de probar modificaciones y evaluar métricas por clase. El test aquí guardado corresponde solo al modelo ampliado congelado: **no valida las versiones creadas después con correcciones locales**. Las validaciones consultadas repetidamente durante el aprendizaje local son orientativas y pueden dejar de ser independientes de las decisiones del usuario.

Las reglas de riesgo son heurísticas y pueden fallar. El chatbot no diagnostica, no supervisa continuamente ni sustituye ayuda profesional. No se evaluó seguridad clínica ni rendimiento con estudiantes reales.
