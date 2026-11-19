# ¿Ayudan los trigramas a interpretar negaciones?

Probamos si incluir grupos de tres palabras ayuda a Ánimo a interpretar expresiones como «no me fue del todo bien». Los trigramas captan parte del orden de las palabras. En esta frase aparecen «no me fue», «me fue del», «fue del todo» y «del todo bien». El vectorizador solo usa los que aprendió durante el entrenamiento y cada grupo sigue siendo una característica por separado.

## Diseño de la comparación

| Decisión | Cómo se aplicó |
|---|---|
| Un solo cambio experimental | TF-IDF de 1–2 palabras frente a TF-IDF de 1–3. El candidato usa la misma regresión logística multinomial, limpieza, ponderación de clases y regularización que el modelo ampliado guardado. |
| Mismos datos | El script verifica las huellas del modelo y de los CSV. Ajusta el vocabulario, los pesos TF-IDF y la regresión del candidato solo con `train.csv`. |
| Métrica principal | Macro-F1 de la validación de GoEmotions en español. Se reporta el F1 por clase y la validación sintética como información secundaria. |
| Comportamiento del chat | Las dos versiones usan el mismo umbral de puntuación, margen y comprobación de vocabulario. Se mide cuántos mensajes reciben etiqueta y cuántos de los etiquetados aciertan. |
| Diagnóstico escrito para el proyecto | Frases con negación e indirectas, agrupadas en pares. Algunas frases neutrales usan palabras y estructuras similares a las de frases emocionales. Se cuenta un par como resuelto cuando ambas respuestas visibles coinciden con las etiquetas esperadas. |
| Casos ambiguos | Se registran aparte sin una etiqueta oficial. La frase que motivó el experimento está entre ellos. También hay una variante con una señal más clara de desánimo en los pares etiquetados. |
| Test histórico | Los resultados congelados siguen vinculados a sus respectivos modelos. Esta comparación usa únicamente entrenamiento, validación y casos de diagnóstico. |

Antes de mirar las predicciones se fijó este filtro para proponer una evaluación posterior. El candidato debe mantener o mejorar macro-F1 en GoEmotions, acertar más frases del diagnóstico, resolver al menos tantos pares y no asignar más emociones a los controles neutrales. Es un criterio práctico para decidir el siguiente paso. El diagnóstico es pequeño, fue escrito para este problema y tiene una sola anotación por frase. Sus resultados son exploratorios. La comprobación de duplicados exactos no descarta semejanzas entre frases.

Las semanas 9, 11 y 12 del curso cubren NLP clásico, bolsa de palabras con regresión logística, separación de datos y métricas. Los trigramas amplían la representación del texto. La regresión sigue aprendiendo un peso por característica y clase. Hydrogen High revisó el diseño y sugirió usar frases con «no» que expresan alegría o neutralidad, separar lo ambiguo de lo evaluable y revisar métricas por clase.

Ejecuta `python -m entrenamiento.p11_trigramas` desde el entorno del proyecto. El script lee `datos de entrenamiento/diagnostico_E/diagnostico.csv` y guarda resultados y predicciones por frase en `entrenamiento/resultados/contexto/comparacion.json`. El ajuste del candidato se hace en memoria.

## Resultados ejecutados

El candidato se ajustó con **23.689 mensajes**. El vocabulario pasó de **113.298 a 259.200 características**, contando todas las longitudes admitidas. Las cifras de esta sección se redondearon a partir de `comparacion.json`.

| Medida | Hasta bigramas | Hasta trigramas |
|---|---:|---:|
| Macro-F1 en validación GoEmotions (1.899) | 0,608 | 0,600 |
| Macro-F1 en validación sintética (1.932) | 0,961 | 0,978 |
| Frases etiquetadas acertadas por el chat | 18/32 | 18/32 |
| Negaciones acertadas | 14/16 | 13/16 |
| Expresiones indirectas acertadas | 4/16 | 5/16 |
| Pares con ambas frases acertadas | 6/16 | 6/16 |
| Controles neutrales etiquetados como alguna emoción | 0/7 | 0/7 |

| F1 por clase en validación GoEmotions | Hasta bigramas | Hasta trigramas |
|---|---:|---:|
| Alegría | 0,617 | 0,589 |
| Ansiedad / nerviosismo | 0,500 | 0,500 |
| Enojo | 0,440 | 0,427 |
| Neutral | 0,936 | 0,934 |
| Tristeza | 0,545 | 0,552 |

El candidato solo cumplió dos de los cuatro criterios. La subida en el conjunto sintético muestra una diferencia en esa fuente. En el diagnóstico con frases de chat no se tradujo en más aciertos. La diferencia en GoEmotions es descriptiva y la validación incluye solo ocho ejemplos de nerviosismo. Hay poca evidencia para generalizar a conversaciones de estudiantes.

### Tres frases para mostrar en clase

| Frase | Hasta bigramas | Hasta trigramas |
|---|---|---|
| «No me fue del todo bien.» | Neutral | Pide más contexto |
| «No me fue del todo bien y volví a casa con el ánimo por el suelo.» | Tristeza | Tristeza |
| «No me escogieron y se me quitaron las ganas de celebrar.» | Tristeza | Pide más contexto |

En la frase original, los cuatro trigramas posibles estaban fuera del vocabulario aprendido. El candidato quedó entre neutral (0,347) y tristeza (0,256). La diferencia fue 0,091, menor que el margen exigido de 0,10. El aprendizaje de un vocabulario ampliado cambia el ajuste de todo el modelo, así que las puntuaciones pueden variar incluso en frases que carecen de trigramas conocidos. La abstención muestra que sus dos primeras puntuaciones quedaron próximas. En la segunda frase ambos modelos encuentran señales de desánimo.

Con este resultado recomiendo conservar el modelo actual. Para comprobar si el problema principal es el tipo de datos, el siguiente experimento sería fijar primero un conjunto nuevo de frases de estudiantes, con consentimiento y etiquetas revisadas. Después se podría probar si añadir ejemplos conversacionales y variados al entrenamiento mejora las dos versiones con esa misma evaluación. Los casos de diagnóstico que ya miramos seguirían sirviendo para ilustrar errores en la exposición.
