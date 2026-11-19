> **Actualización:** esta guía describe principalmente el experimento original. El modelo ampliado y el aprendizaje local se explican en [APRENDIZAJE.md](APRENDIZAJE.md) y [INFORME_AMPLIACION.md](INFORME_AMPLIACION.md).

# Ánimo explicado paso a paso, sin conocimientos previos

Empieza aquí. Vamos a seguir un ejemplo y explicar cada idea antes de ponerle un nombre técnico. Puedes detenerte al final de cada sección e intentar contarla con tus palabras.

Los ejemplos de frases y números inventados para enseñar están señalados como ejemplos. Cuando decimos «resultado del proyecto», hablamos de las cifras guardadas del trabajo.

## 1. ¿Qué estamos haciendo?

Hicimos una página donde puedes escribir un mensaje. El programa intenta decir si ese mensaje expresa alegría, tristeza, enojo, ansiedad o algo neutral.

Por ejemplo, alguien podría escribir «Estoy feliz porque aprobé». Queremos que el programa reconozca una frase de alegría. Es un ejemplo para explicar lo que buscamos; no significa que hayamos comprobado aquí su respuesta exacta.

**Clasificar** quiere decir colocar algo en una categoría. Aquí las categorías son esas cinco emociones. Una categoría también puede llamarse **clase** o **etiqueta**. Cuando encuentres estas palabras, piensa en «alegría», «tristeza», etc.

Después de elegir una categoría, el programa muestra una respuesta que ya escribimos. Si no se cumplen ciertas condiciones para elegir, pide más información. Veremos esas condiciones más adelante.

El programa analiza lo que alguien escribe. No puede saber todo lo que esa persona está sintiendo.

## 2. ¿Cómo le enseñamos?

Imagina que quieres enseñarle a alguien a distinguir frases de alegría y tristeza. Podrías mostrarle esto:

| Frase de ejemplo | Respuesta que le enseñas |
|---|---|
| Estoy contento por la buena noticia | alegría |
| Me siento triste porque extraño a mi familia | tristeza |

Hacemos algo parecido con el programa: le entregamos frases junto con su categoría. Las frases del proyecto ya tienen esa respuesta asignada antes de empezar a enseñarle.

El programa hace cálculos para encontrar relaciones entre palabras y categorías. Por ejemplo, puede aprender que determinadas palabras aparecen mucho en las frases marcadas como alegría. También puede aprender relaciones equivocadas si los ejemplos son pocos o poco variados.

A enseñarle de esta forma lo llamamos **entrenar**. **Aprendizaje supervisado** significa que aprende usando ejemplos que ya tienen una respuesta asignada. No significa que alguien esté vigilando cada mensaje del chat.

**Machine Learning** es el nombre en inglés de aprendizaje automático: usar datos para ajustar cómo funciona un programa.

## 3. ¿Qué es el modelo?

El programa no guarda el aprendizaje como un resumen escrito que diga «estas son las reglas de las emociones». Lo guarda principalmente como números.

Esos números permiten calcular cuánto favorecen las palabras a una categoría frente a otra. En este proyecto también se conserva la lista de palabras y parejas de palabras que aprendió a utilizar.

A ese conjunto que podemos usar para clasificar mensajes lo llamamos **modelo entrenado**. Cuando digamos «el modelo», piensa en la herramienta que ya aprendió de los ejemplos.

Hay tres momentos diferentes:

- **Entrenar:** enseñarle con frases y sus respuestas.
- **Predecir:** darle una frase y pedirle que elija una categoría usando lo aprendido.
- **Evaluar:** comparar lo que eligió con la respuesta que nosotros conocemos y contar aciertos y errores.

Cuando escribes en el chat, el modelo predice. Sus números no se actualizan con tu mensaje.

## 4. ¿Cómo construimos el proyecto, explicado sin términos difíciles?

Estos son los pasos que hay que entender:

1. Decidimos qué queríamos que hiciera: distinguir cinco categorías en mensajes en español.
2. Preparamos 400 frases y les asignamos una categoría.
3. Separamos esas frases: unas para enseñarle, otras para elegir entre varias alternativas y otras para comprobar el resultado final.
4. Probamos cinco maneras de hacer la clasificación. Cada una aprendió con las frases destinadas a enseñarle.
5. Comparamos sus respuestas con las de las frases destinadas a elegir una alternativa. Así escogimos una.
6. Con la alternativa elegida, comprobamos cuántas frases acertaba en la prueba final.
7. Conservamos el modelo elegido y sus resultados.
8. Conectamos ese modelo con la página, para que pudiera analizar los mensajes escritos por el usuario.
9. Añadimos respuestas redactadas y reglas para mostrar precaución ante ciertas expresiones de crisis.

No tienes que aprenderte nueve pasos de memoria. La idea central es: **preparamos ejemplos, le enseñamos, comprobamos sus respuestas y lo conectamos al chat**.

## 5. ¿Qué archivo hace cada cosa?

No necesitas abrir todos de golpe. Esta tabla te servirá cuando quieras encontrar una parte:

| Archivo | Qué hace |
|---|---|
| `datos de entrenamiento/corpus_original_E/escenarios.txt` | Contiene las frases y sus categorías. |
| `entrenamiento/p01_datos.py` | Organiza y prepara esas frases. |
| `entrenamiento/p02_modelos.py` | Construye las alternativas que vamos a probar. |
| `entrenamiento/p03_comparacion.py` | Les enseña y las compara. |
| `entrenamiento/p04_evaluacion.py` | Comprueba el resultado final del elegido. |
| `entrenamiento/p05_inferencia.py` | Utiliza lo aprendido para clasificar un mensaje. |
| `entrenamiento/p06_informe.py` | Dibuja los gráficos y crea una tabla de errores. |
| `entrenamiento/artefactos/modelo.joblib` | Guarda lo aprendido. |
| `conversacion.py` | Contiene las respuestas escritas del chatbot. |
| `seguridad.py` | Busca expresiones que activan precaución. |
| `app.py` | Recibe mensajes de la página y coordina la respuesta. |
| `index.html` | Es la página que utilizas. |

**Inferencia**, la palabra del nombre `p05_inferencia.py`, significa aquí utilizar un modelo ya entrenado para obtener una predicción. No es otra fase de aprendizaje.

## 6. ¿Qué son esas 400 frases?

Son frases inventadas para este trabajo con ayuda de IA. **Sintético** significa eso aquí: creado para el proyecto, en lugar de recogido de conversaciones de personas.

Están organizadas en 200 situaciones. Cada situación tiene dos frases que expresan algo parecido. Por ejemplo, «Aprobé y estoy feliz» y «Estoy contento porque pasé el examen». A decir algo parecido con otras palabras lo llamamos **parafrasear**; cada frase es una **paráfrasis**.

Hay 40 situaciones por categoría. Como cada situación tiene dos frases, son 80 frases por categoría y 400 en total.

Si lees **corpus**, significa la colección de textos. Si lees **dataset**, significa el conjunto de datos organizado para trabajar. En nuestro caso hablamos de las frases, sus categorías y la información que permite organizarlas.

Las categorías las asignamos al preparar las frases. No son emociones medidas en personas ni etiquetas revisadas por varios especialistas.

## 7. ¿Para qué separamos las frases?

Piensa en un estudiante. Para saber si aprendió, conviene preguntarle algo diferente a lo que acaba de estudiar.

Con el programa hacemos lo siguiente:

| Frases | Uso | Nombre que aparece en el código |
|---|---|---|
| 240 | Se usan para enseñarle. | `train`: entrenamiento en inglés. |
| 80 | Se usan para comparar alternativas y elegir una. | Validación. |
| 80 | Se usan para comprobar al elegido al final. | `test`: prueba en inglés. |

En adelante, cuando veas **train**, piensa «las frases para aprender». Cuando veas **test**, piensa «las frases de la prueba final».

Las dos frases de una misma situación permanecen juntas. Si una sirve para aprender y la otra para examinarlo, podríamos estar haciéndole una prueba demasiado parecida a lo que le enseñamos.

En el código aparece una **semilla**, el número 2026. Sirve para poder repetir la mezcla de las situaciones en las mismas condiciones. No es una nota ni el número de frases.

## 8. ¿Qué hace antes de analizar las palabras?

Primero convierte la frase a una forma más uniforme. Por ejemplo:

`¡NO estoy FELÍZ! 123 → no estoy feliz`

Pasa las mayúsculas a minúsculas y quita tildes, signos, números y emojis. También cambia ñ por n y deja espacios sencillos.

Esto se llama **normalizar**. Lo hace tanto con las frases para aprender como con los mensajes nuevos.

Conserva «no», «sin» y «nunca». Son importantes porque «estoy feliz» y «no estoy feliz» pueden decir cosas distintas. Conservar «no» no asegura que el modelo interprete bien toda la frase.

La función que hace esto se llama `normalizar()` y está en `p01_datos.py`.

## 9. ¿Por qué convierte las palabras en números?

La herramienta que clasifica trabaja haciendo cuentas. Por eso necesita una representación numérica del texto.

Imagina, solo para explicar, que conoce estas tres palabras: feliz, triste y examen. Podríamos representar frases así:

| Frase | Veces que aparece feliz | Veces que aparece triste | Veces que aparece examen |
|---|---|---|---|
| feliz | 1 | 0 | 0 |
| triste examen | 0 | 1 | 1 |

La lista de términos conocidos se llama **vocabulario**. La herramienta que transforma el texto en números se llama **vectorizador**. Un **vector** es aquí la fila de números que representa una frase.

Nuestro ganador no se queda solo con contar. Utiliza **TF-IDF**, un cálculo que considera cuántas veces aparece un término en la frase y qué tan común es entre las frases usadas para aprender.

No necesitas memorizar la fórmula para entender el recorrido: **texto → números → elección de categoría**. Un término poco común puede recibir más peso en esa representación, pero eso no garantiza que sea útil para distinguir emociones.

También se utilizan parejas de palabras consecutivas, como «estoy feliz». Cada pareja se llama **bigrama**. Una palabra individual es un **unigrama**.

El vocabulario se aprende solo con las 240 frases de entrenamiento. Después queda fijo. Las palabras desconocidas se ignoran; escribirlas en el chat no las añade al vocabulario.

## 10. ¿Qué números aprende para distinguir emociones?

Además de representar la frase con números, el programa aprende cuánto debe contar cada término a favor o en contra de cada categoría. Esos valores se llaman **pesos**.

Por ejemplo didáctico, una palabra podría favorecer alegría más que tristeza. No escribimos a mano el valor de cada palabra: el entrenamiento lo calcula usando las frases y sus categorías.

El cálculo busca dar mejores respuestas a los ejemplos. Ajusta los pesos para reducir una medida del error. Esa medida se llama **función de pérdida**. Es una forma matemática de penalizar respuestas poco acordes con la etiqueta enseñada.

La herramienta elegida se llama **regresión logística multinomial**. El nombre es largo, pero su trabajo aquí es elegir entre las cinco categorías.

Combina los números de la frase con los pesos aprendidos y produce una puntuación para cada categoría. Las cinco puntuaciones suman aproximadamente uno. El cálculo que las convierte a esa forma se llama **softmax**.

Una puntuación de 0,60 en tristeza no significa que la persona esté «60% triste». Es un valor que usa el programa para comparar sus categorías.

## 11. ¿Por qué probamos cinco alternativas?

Porque no sabíamos cuál daría mejores respuestas con estos datos. Las cinco aprenden o se definen usando entrenamiento; las comparamos con las 80 frases de validación.

| Nombre en Python | Explicación |
|---|---|
| `logistica_tfidf` | Usa la herramienta logística con palabras individuales y TF-IDF. |
| `logistica_conteos` | Usa logística con conteos de palabras. |
| `logistica_bigramas` | Usa logística con palabras y parejas, representadas con TF-IDF. |
| `naive_bayes` | Usa otra forma matemática de clasificar: Naive Bayes. |
| `referencia` | Siempre responde con la categoría más frecuente en entrenamiento. |

La última es una comparación muy simple: queremos saber si aprender de las palabras aporta más que repetir una categoría siempre. En validación acertó el 20%.

Ganó `logistica_bigramas`. **El chat utiliza solamente ese elegido.** Las otras alternativas sirven para el experimento.

Verás la palabra **Pipeline** en Python. Significa que unimos dos pasos: convertir el texto en números y clasificar esos números. Conservamos juntos los dos pasos aprendidos.

## 12. ¿Qué es un umbral y por qué pide más información?

Un **umbral** es un límite que debe alcanzarse para tomar una decisión. Aquí es una puntuación mínima.

Ejemplo con números inventados: si alegría obtiene 0,42 y tristeza 0,40, alegría queda primera, pero hay poca diferencia. El programa puede pedir más información en lugar de mostrar alegría.

Las condiciones del proyecto son:

1. La mayor puntuación debe llegar a 0,30.
2. Debe superar a la segunda por al menos 0,10.
3. Debe quedar alguna palabra o pareja conocida después de transformar el texto.

Si falla una condición, aparece `incierto`. A no mostrar una categoría lo llamamos **abstenerse**. No es una sexta emoción que le enseñamos.

El límite 0,30 se eligió con validación. Se probaron límites entre 0,30 y 0,70 y se buscó aceptar muchos mensajes manteniendo ciertos resultados en ese conjunto. Esto no garantiza iguales aciertos con mensajes nuevos.

## 13. ¿Cómo contamos los aciertos?

Damos al modelo las 80 frases de la prueba final y comparamos lo que responde con sus categorías conocidas.

Acertó **53 de 80**. Si divides 53 entre 80 obtienes 0,6625: **66,25%**. A la proporción de aciertos la llamamos **accuracy**.

También revisamos cada categoría. Para alegría, por ejemplo:

- **Precision:** entre todas las frases que llamó alegría, ¿cuántas tenían esa etiqueta?
- **Recall:** entre todas las frases etiquetadas como alegría, ¿cuántas reconoció?
- **F1:** una medida que combina las dos preguntas anteriores.
- **Macro-F1:** calcula esa medida para cada categoría y promedia las cinco, con igual importancia.

Una **métrica** es simplemente una medida del resultado. Accuracy y F1 son métricas.

La **matriz de confusión** es una tabla que muestra las categorías conocidas frente a las respuestas del modelo. Permite ver, por ejemplo, cuántas frases de alegría confundió con tristeza.

No necesitas aprender todas las fórmulas de golpe. Primero entiende qué pregunta responde cada medida.

## 14. ¿Por qué unas veces contamos 80 mensajes y otras 60?

El programa acepta mostrar una categoría en 60 de los 80 mensajes. En los otros 20 pide más contexto.

Entre esos 60 aceptados acertó 44: 44/60 es aproximadamente **73,33%**. Es el porcentaje de aciertos solo entre los mensajes aceptados.

La **cobertura** indica qué parte de los mensajes aceptó: 60/80, es decir, **75%**.

Por eso hay dos porcentajes de aciertos: 53/80 para todas las predicciones internas y 44/60 para las aceptadas. No son la misma pregunta ni usan el mismo grupo.

Pedir contexto no elimina todos los errores: 16 de los mensajes aceptados quedaron mal clasificados.

## 15. ¿Dónde queda guardado lo aprendido?

En `entrenamiento/artefactos/modelo.joblib`. Guarda la transformación del texto, los valores aprendidos para clasificar, el nombre elegido y el límite para aceptar respuestas.

Al iniciar la aplicación se carga ese archivo. Así puede utilizar lo aprendido sin repetir las clases cada vez.

Si falta, el programa tiene una forma de reconstruir el modelo en memoria usando las frases de entrenamiento. No necesita abrir la prueba final para hacerlo ni guarda automáticamente otro archivo.

## 16. ¿Qué sucede cuando escribo en la página?

Sigamos un ejemplo de mensaje, sin asegurar qué respuesta dará exactamente: «Estoy feliz por terminar mi proyecto».

1. La página envía el texto a Python.
2. Python revisa que sea un mensaje válido y comprueba las reglas de precaución.
3. Transforma las palabras conocidas en números.
4. El modelo calcula las puntuaciones de las cinco categorías.
5. Se comprueba si cumple las condiciones para mostrar la primera.
6. Se elige una respuesta redactada para esa categoría o para incertidumbre.
7. La página muestra el resultado.

Las palabras destacadas en el análisis son términos que favorecieron a la primera categoría frente a la segunda. Se muestran hasta cinco aportes positivos. No son todos los cálculos de la decisión.

## 17. ¿Las respuestas las inventa el modelo?

Las respuestas están escritas en `conversacion.py`. El programa selecciona una según la categoría y va alternando los textos disponibles.

Por tanto, hay dos tareas: el modelo calcula la categoría del mensaje; otra parte del programa selecciona una respuesta preparada.

El clasificador analiza un mensaje cada vez. Aunque tú veas el diálogo en pantalla, no utiliza toda esa conversación para interpretar el siguiente mensaje. Una respuesta como «sí» puede no darle suficiente información.

## 18. ¿Qué ocurre con una expresión de crisis?

Antes de clasificar, `seguridad.py` busca ciertas expresiones relacionadas con suicidio o autolesión. Si encuentra una, el programa muestra apoyo y omite el análisis de emociones.

Estas son reglas escritas por quienes desarrollan el programa. No forman una emoción adicional aprendida. Pueden activarse cuando alguien cita un texto o habla de un tercero, y pueden fallar ante expresiones no contempladas.

Después de una alerta, la página envía un indicador para continuar con precaución. «Nueva conversación» lo reinicia. Eso no significa que el modelo comprenda todo el diálogo.

## 19. ¿Cómo se conectan la página y Python?

La página es la parte que ves. Python hace los cálculos y decide la respuesta que se va a enviar.

`app.py` los conecta usando **Flask**, una librería: herramientas de programación que podemos utilizar sin escribirlas desde cero.

La página envía mensajes a una dirección llamada `/api/chat`. Python devuelve información para mostrar. A esa forma acordada de intercambiar información la llamamos **API**.

La información viaja como **JSON**: campos con nombre, por ejemplo `message` para el mensaje y `reply` para la respuesta.

JavaScript es el código de la página que envía esos mensajes y dibuja los resultados. El aprendizaje ocurre en Python.

## 20. ¿Para qué sirven las 61 pruebas?

Son comprobaciones automáticas de funcionamiento. Por ejemplo: ¿qué ocurre si el mensaje está vacío?, ¿se pide contexto cuando no hay palabras conocidas?, ¿se atienden las reglas de precaución aunque falte el modelo?

Se ejecutan con una herramienta llamada `pytest`. Usan ejemplos de desarrollo y archivos temporales; no vuelven a examinar al modelo con las 80 frases finales.

Pasar 61 pruebas significa cumplir esas comprobaciones de software. No significa acertar 61 emociones. La evaluación de emociones es la del 53/80.

## 21. ¿Qué resultados conviene recordar?

| Pregunta | Resultado del proyecto |
|---|---|
| ¿Cuántas frases hay? | 400, creadas para el trabajo con ayuda de IA. |
| ¿Cuántas sirvieron para aprender? | 240. |
| ¿Cuántas para elegir? | 80. |
| ¿Cuántas para la prueba final? | 80. |
| ¿Qué alternativa ganó? | Logística con TF-IDF de palabras y parejas. |
| ¿Cuántas acertó en la prueba final? | 53 de 80: 66,25%. |
| ¿Qué categoría fue más difícil? | Alegría: reconoció 4 de sus 16 frases. |
| ¿Cuánto fue su macro-F1 final? | Aproximadamente 0,653. |

Estos resultados muestran límites importantes. Las frases son pocas e inventadas; no permiten asegurar que funcionará igual con conversaciones reales.

Puedes consultar todas las cifras y los 27 errores en [el informe](INFORME_RESULTADOS.md).

## 22. ¿Qué significan otros gráficos y palabras del informe?

**Sobreajuste:** aprende muy bien los ejemplos conocidos, pero falla más con otros. Es parecido a memorizar ejercicios y no poder resolver uno diferente. Las diferencias de resultados también pueden deberse a que los conjuntos son pequeños y distintos.

**Curva de aprendizaje:** un gráfico que muestra qué pasó al enseñarle con diferentes cantidades de frases. Se probaron 60, 120, 180 y 240. Más frases ayudaron en este experimento; no significa que cualquier frase añadida vaya a mejorar el modelo.

**Bootstrap:** se vuelve a calcular la medida con muestras formadas a partir de situaciones ya evaluadas. Usa las respuestas guardadas, sin volver a enseñar ni a pedir predicciones al modelo. Ayuda a explorar cuánto podría variar el resultado dentro de esa muestra.

Estas ideas pueden esperar hasta que entiendas el recorrido principal.

## 23. ¿Cómo lo pongo en marcha?

Con el entorno del proyecto preparado, abre una terminal en la carpeta del proyecto y ejecuta:

```sh
source .venv/bin/activate
python app.py
```

Abre la dirección que aparece en la terminal. Mantén esa terminal abierta durante la demostración. Ctrl+C detiene el servidor.

Para instalarlo en otro equipo, sigue [las instrucciones de instalación](REPRODUCIBILIDAD.md).

## 24. ¿Qué instrucciones de Python debo reconocer?

| Instrucción | Significado aquí |
|---|---|
| `def` | Crea una función: un conjunto de instrucciones para una tarea. |
| `return` | Entrega el resultado de esa tarea. |
| `.fit()` | Aprende a partir de los ejemplos. |
| `.transform()` | Convierte texto en números usando lo aprendido. |
| `.predict_proba()` | Calcula puntuaciones para las categorías. |
| `argmax` | Encuentra qué valor es el mayor. |

**Scikit-learn** es la librería que aporta las herramientas para transformar textos y entrenar clasificadores. Las herramientas vienen programadas, pero nuestro modelo aprende sus valores con nuestros ejemplos.

## 25. ¿Qué no conviene confundir?

- Aprender se hace durante el entrenamiento. Escribir en el chat utiliza lo aprendido.
- La categoría la calcula el modelo. La respuesta escrita se elige de textos preparados.
- Las reglas de precaución están programadas por separado.
- Laboratorio muestra resultados guardados. Abrirlo no entrena otra vez.
- Los mensajes no se almacenan permanentemente en la aplicación. Exportar permite guardar una copia local por decisión del usuario.

## 26. ¿Qué código muestro si me preguntan?

Los archivos `p00` a `p06` están en `entrenamiento/`.

| Si preguntan por… | Archivo y función |
|---|---|
| Configuración y lectura de archivos | `p00_configuracion.py`: constantes, `leer_json()`, `guardar_json()`. |
| Preparar frases | `p01_datos.py`: `normalizar()`, `preparar()`, `xy()`. |
| Construir las alternativas | `p02_modelos.py`: `crear_modelo()`. |
| Enseñarles y elegir | `p03_comparacion.py`: `comparar()` y su llamada a `.fit()`. |
| Medir aciertos | `p02_modelos.py`: `metricas()`; `p04_evaluacion.py`: `evaluar_una_vez()`. |
| Clasificar un mensaje | `p05_inferencia.py`: `predecir()`. |
| Cargar lo aprendido | `p05_inferencia.py`: `cargar_modelo()`. |
| Destacar términos del análisis | `p05_inferencia.py`: `explicar()`. |
| Dibujar resultados | `p06_informe.py`: `generar_informe()`. |
| Elegir la respuesta escrita | `conversacion.py`: `responder()`. |
| Activar precaución | `seguridad.py`: `detectar_riesgo()`. |
| Recibir el mensaje de la página | `app.py`: `chat()`. |

## 27. Comprueba si ya lo puedes contar

Intenta responder sin mirar:

1. ¿Qué hace el programa cuando escribo un mensaje?
2. ¿Cómo le enseñamos y de dónde vienen las frases?
3. ¿Qué significa train?
4. ¿Por qué guardamos frases para comprobarlo al final?
5. ¿Por qué convierte palabras en números?
6. ¿Qué es un umbral?
7. ¿El programa aprende mientras conversamos?
8. ¿De dónde salen sus respuestas escritas?
9. ¿Qué significa que acertó 53 de 80?

Si una respuesta no sale, vuelve solo a esa parte. No hace falta releer todo de una vez.

## 28. Palabras adicionales que podrías encontrar en el código

Esta sección es para consultar, después de entender las anteriores.

| Palabra | Significado sencillo |
|---|---|
| Característica | Una entrada numérica que se utiliza para clasificar; aquí representa una palabra o pareja. |
| Matriz dispersa | Forma de guardar una tabla con muchos ceros sin ocupar espacio innecesario para cada cero. |
| Parámetro | Un valor que aprende el modelo, como un peso. |
| Hiperparámetro | Un ajuste que elegimos antes de enseñarle, como `C` o `max_iter`. |
| Regularización | Una forma de limitar los pesos aprendidos para reducir la adaptación excesiva a los ejemplos. |
| `C=4` | Ajusta la fuerza inversa de esa limitación; un C menor aplica más regularización. |
| `max_iter=2000` | Límite de pasos del cálculo que ajusta los pesos; no obliga a hacer siempre 2000. |
| Fuga de datos | Información de la prueba influye indebidamente en cómo enseñamos o elegimos el modelo. |
| Naive Bayes | Otra herramienta para clasificar a partir de asociaciones entre términos y categorías. |
| Serializar | Guardar lo aprendido en un archivo para cargarlo después. |
| SHA-256 | Código calculado desde el contenido de un archivo para detectar cambios. |

## 29. Una explicación breve que puedes practicar

«Ánimo intenta reconocer qué emoción expresa un mensaje en español. Para enseñarle preparamos 400 frases con categorías. Usamos 240 para aprender, 80 para elegir entre alternativas y 80 para comprobar el resultado final. El programa convierte palabras en números y utiliza valores aprendidos para elegir una categoría. El ganador acertó 53 de 80 frases finales. Después selecciona una respuesta que ya escribimos. Mientras conversamos no aprende de nuevo. Es un trabajo académico con frases sintéticas y tiene errores que todavía debemos mejorar».

Después puedes repasar preguntas con la [guía de defensa](guia-para-defender-animo.md).

## 32. Cómo enseñarle con más frases, paso a paso

Esta sección explica una práctica que puedes hacer después de entender el proyecto. **Leer los comandos no los ejecuta.** Para hacer la práctica tendrías que escribirlos o pegarlos en una terminal.

La idea es sencilla: crear una copia del proyecto, añadir frases para enseñarle más ejemplos y volver a entrenar en esa copia. Al terminar tendremos otro modelo y podremos estudiar sus resultados.

Añadir una frase a un archivo no cambia por sí solo lo que el modelo sabe. Para que aprenda de los ejemplos nuevos hay que repetir el entrenamiento.

### Paso 1. Crear una carpeta para hacer la práctica

Queremos conservar el proyecto que ya funciona. Por eso vamos a trabajar en otra carpeta, llamada `animo-experimento-nuevo`.

Abre la terminal en la carpeta del proyecto. En macOS, puedes pegar este bloque:

```sh
mkdir ../animo-experimento-nuevo && \
rsync -a --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' ./ ../animo-experimento-nuevo/ && \
cd ../animo-experimento-nuevo && \
python3 -m venv .venv && \
source .venv/bin/activate && \
python -m pip install -r requirements.txt
```

Cada línea hace una tarea:

| Comando | Qué hace |
|---|---|
| `mkdir ../animo-experimento-nuevo` | Crea una carpeta al lado de la del proyecto. `..` significa subir a la carpeta que contiene el proyecto. |
| `rsync ...` | Copia los archivos del proyecto a la carpeta nueva. Las opciones `--exclude` dejan fuera Git, el entorno de Python y archivos temporales. |
| `cd ../animo-experimento-nuevo` | Entra en la carpeta nueva. |
| `python3 -m venv .venv` | Crea un espacio propio para instalar las herramientas de Python de esta copia. Ese espacio se llama entorno virtual. |
| `source .venv/bin/activate` | Activa ese espacio para los siguientes comandos. |
| `python -m pip install -r requirements.txt` | Instala las librerías que necesita el proyecto. Puede necesitar internet para descargarlas. |

Los símbolos `&&` indican que el siguiente comando se ejecuta solamente si el anterior termina bien. La barra `\` permite continuar el bloque en la línea siguiente.

Si la carpeta ya existe, el primer comando se detiene. No borres una carpeta sin revisar qué contiene. Para seguir estos ejemplos tal cual, necesitas una carpeta nueva con el nombre indicado. Si eliges otro nombre, también debes cambiarlo en el bloque de comprobación del paso 3.

Abre la carpeta nueva en VS Code. A partir de aquí, los cambios y comandos de esta práctica se hacen **dentro de esa copia**.

### Paso 2. Añadir frases

En la copia, abre:

`datos de entrenamiento/corpus_original_E/escenarios.txt`

Busca una categoría, por ejemplo `[alegria]`. Debajo puedes añadir una línea como esta:

```text
Estoy feliz por haber terminado mi curso | Me alegra mucho completar mis estudios
```

La barra `|` separa dos formas de expresar una misma situación. Las dos frases de esa línea deben corresponder a la categoría bajo la que están escritas.

Añade situaciones diferentes en las cinco categorías. No te limites a cambiar una palabra de la misma frase muchas veces. Intenta mantener cantidades parecidas por categoría y revisa que la emoción elegida tenga sentido.

El programa no admite frases repetidas después de limpiarlas. Por ejemplo, cambiar solo las mayúsculas o las tildes no crea una frase diferente para esta comprobación.

Recuerda cómo se divide el material: unas frases sirven para aprender, otras para elegir la alternativa y otras para medirla. **Solo las palabras y parejas que aparezcan en las frases para aprender pueden entrar al vocabulario.** Una palabra que aparezca únicamente en las frases para examinarlo seguirá siendo desconocida.

El proyecto identifica estos textos como `sintetico_asistido_IA`: frases creadas con ayuda de IA. Si utilizas textos de otra procedencia, debes explicar ese origen en la ficha de datos y adaptar el campo correspondiente en `p01_datos.py`. Para usar mensajes de personas necesitas autorización y cuidar su privacidad.

### Paso 3. Permitir que la copia construya sus propios archivos

El proyecto ya tiene guardados los grupos de frases, el modelo elegido y los resultados. Si ejecutas sus comandos con esos archivos presentes, suele reutilizarlos.

Para repetir todo desde el principio en la copia, vamos a quitar **sus archivos calculados**. Conservaremos el código y `escenarios.txt`, donde están las frases que acabas de editar.

Con la terminal situada en `animo-experimento-nuevo`, puedes ejecutar:

```sh
python - <<'PY'
from pathlib import Path
import shutil

raiz = Path.cwd()
if raiz.name != 'animo-experimento-nuevo':
    raise SystemExit('detenido: entra primero en la copia del experimento')

datos = raiz / 'entrenamiento' / 'datos'
for nombre in ('manifiesto.json', 'train.csv', 'validacion.csv', 'test.csv'):
    (datos / nombre).unlink(missing_ok=True)
shutil.rmtree(raiz / 'entrenamiento' / 'resultados', ignore_errors=True)
(raiz / 'entrenamiento' / 'resultados').mkdir()
(raiz / 'entrenamiento' / 'artefactos' / 'modelo.joblib').unlink(missing_ok=True)
PY
```

Este bloque hace lo siguiente:

1. `Path.cwd()` averigua en qué carpeta está la terminal.
2. La condición `if` comprueba el nombre de esa carpeta. Si no es el indicado, se detiene antes de borrar nada. Revisa igualmente que sea la copia correcta.
3. Quita las tablas que separaban las frases para aprender, elegir y examinar: `train.csv`, `validacion.csv` y `test.csv`.
4. Quita `manifiesto.json`, el archivo que registraba esa separación y los datos usados.
5. Quita la carpeta de resultados y crea una vacía.
6. Quita el modelo guardado en la copia.

`unlink()` elimina un archivo. `shutil.rmtree()` elimina una carpeta y su contenido. Por eso es importante ejecutar el bloque en la copia, no en el proyecto que quieres presentar.

El informe escrito, `INFORME_RESULTADOS.md`, permanece copiado. **Sus cifras todavía describen el modelo que tenías antes de esta práctica.** Habrá que actualizarlo cuando tengamos los resultados nuevos.

### Paso 4. Enseñarle otra vez y comprobarlo

Ejecuta estos comandos en orden. Espera a que termine uno antes de pasar al siguiente y revisa si aparece algún error:

```sh
python -m entrenamiento preparar
python -m entrenamiento comparar
python -m entrenamiento evaluar
python -m entrenamiento informe
```

| Última palabra del comando | Qué hace |
|---|---|
| `preparar` | Lee las frases y vuelve a separarlas en los grupos para aprender, elegir y examinar. Mantiene juntas las dos frases de cada situación. |
| `comparar` | Enseña a los cinco candidatos con las frases para aprender y utiliza las de validación para elegir uno y su límite de aceptación. |
| `evaluar` | Usa las frases de la prueba final para contar los aciertos del elegido y guarda las respuestas. |
| `informe` | Dibuja los gráficos y crea la tabla de errores a partir de los resultados guardados. |

Antes de `evaluar`, deja decididos los datos, los candidatos y la forma de elegir. No cambies después esas decisiones solo porque no te guste la nota.

Los resultados quedan en:

- `entrenamiento/resultados/seleccion.json`: comparación y candidato elegido.
- `entrenamiento/resultados/test_final.json`: aciertos, errores y respuestas de la prueba final.
- `entrenamiento/resultados/figuras/`: gráficos.
- `entrenamiento/resultados/errores_test.csv`: frases que clasificó mal.

**El comando `informe` no reescribe el texto de `INFORME_RESULTADOS.md`.** Debes actualizar ese documento usando las cifras nuevas. Lo mismo se aplica a las cifras del manual, la guía y otros documentos que copies: no se actualizan solas.

### Paso 5. Entender qué podemos concluir

Hemos conseguido enseñarle otra vez con más ejemplos. Eso no asegura que ahora clasifique mejor los mensajes de otras personas.

Hay una diferencia importante: si ya conoces las frases del examen y sus errores, puedes terminar preparando el modelo para esas preguntas. Volver a mezclarlas no hace que dejes de conocerlas.

Para comprobar mejor una mejora, necesitas situaciones realmente nuevas, guardadas para examinar al modelo y no usadas para decidir cómo enseñarle.

Los comandos de esta práctica mezclan automáticamente todas las situaciones. **No tienen una opción que garantice que las frases nuevas queden solo en la prueba final.** Para hacer esa comprobación independiente habría que adaptar la preparación y mantener esas situaciones nuevas fuera de las frases usadas para aprender y elegir.

Por eso, si el porcentaje aumenta al cambiar las frases y su reparto, cuenta exactamente qué cambió. No presentes ese aumento, por sí solo, como prueba de que el modelo mejoró con cualquier mensaje.

### Paso 6. Revisar lo que aprendimos de la práctica

No mires solo el porcentaje total. Revisa también qué categorías confunde, cuántos mensajes acepta etiquetar y en cuántos de esos aceptados se equivoca.

Mantén las cinco categorías para esta práctica. Añadir otra emoción requiere cambiar también la configuración, las respuestas y la página, además de preparar datos y evaluar de nuevo.

La idea que debes poder explicar es esta: **para que aprenda de nuevas frases hay que volver a entrenar; para saber si mejoró hay que comprobarlo con cuidado usando frases que no hayan guiado esa mejora**.
