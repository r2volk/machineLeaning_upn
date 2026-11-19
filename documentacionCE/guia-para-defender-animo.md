> **Actualización:** esta guía describe principalmente el experimento original. El modelo ampliado y el aprendizaje local se explican en [APRENDIZAJE.md](APRENDIZAJE.md) y [INFORME_AMPLIACION.md](INFORME_AMPLIACION.md).

# Guía para explicar Ánimo con tus propias palabras

Esta guía te ayuda a preparar la exposición desde cero. Primero explica las ideas con palabras comunes. Los nombres técnicos aparecen acompañados de su significado.

Puedes practicar respondiendo en voz alta. No hace falta repetir las respuestas exactamente como están escritas.

## 1. Qué puedes decir al presentar el proyecto

> Nuestro proyecto se llama Ánimo. Es una página donde escribes un mensaje y el programa intenta decir si expresa alegría, tristeza, enojo, ansiedad o algo neutral.
>
> Para enseñarle, preparamos frases con una de esas categorías asignada. El programa calculó relaciones entre las palabras y las categorías. Después utiliza lo aprendido para responder sobre mensajes nuevos.
>
> Probamos cinco alternativas y elegimos una. En la prueba final acertó 53 de 80 frases. A veces pide más información en vez de mostrar una categoría. Sus respuestas escritas ya estaban preparadas.
>
> Las frases para enseñarle fueron creadas para el proyecto con ayuda de IA. El programa todavía tiene errores y no sirve para conocer con certeza cómo se siente una persona.

## 2. Entender qué hace y cómo aprende

### ¿Qué quiere decir clasificar?

Colocar algo en una categoría. Aquí queremos colocar mensajes en cinco categorías: alegría, tristeza, enojo, ansiedad o neutral. Si la profesora dice «clase» o «etiqueta», se refiere a una de esas categorías.

Ejemplo para explicar la idea: a «Estoy contento por la buena noticia» le podemos asignar alegría. Eso no es una promesa sobre la respuesta exacta que dará el programa.

### ¿Qué es Machine Learning?

Es aprendizaje automático: usar ejemplos para ajustar cómo responde un programa. Aquí le damos frases y sus categorías para que encuentre relaciones entre ambas.

### ¿Qué significa aprendizaje supervisado?

Que los ejemplos ya tienen una respuesta que le queremos enseñar. Le mostramos una frase junto a «alegría», «tristeza» u otra categoría. No significa que alguien revise cada mensaje mientras usamos el chat.

### ¿Qué es el modelo?

Es la herramienta que queda después de enseñarle. Conserva información aprendida para clasificar mensajes nuevos. Principalmente guarda una lista de términos conocidos y números que ayudan a decidir entre las categorías.

### ¿Qué significa entrenar?

Enseñarle con ejemplos. El programa hace cálculos y ajusta números para que sus respuestas se acerquen a las categorías de las frases que le damos.

### ¿Qué números aprende?

Aprende, entre otras cosas, cuánto debe contar cada palabra o pareja de palabras a favor o en contra de una categoría. Esos valores se llaman **pesos**. Nosotros no escribimos a mano el valor para cada palabra: lo calcula el entrenamiento.

### ¿Qué diferencia hay entre entrenar, predecir y evaluar?

| Palabra | Qué hacemos |
|---|---|
| Entrenar | Enseñarle con frases y sus categorías. |
| Predecir | Darle un mensaje para que elija una categoría usando lo aprendido. |
| Evaluar | Comparar sus respuestas con las que conocemos y contar aciertos y errores. |

### ¿Aprende mientras hablo con él?

Actualmente no. Al conversar utiliza los números que ya aprendió. Tu mensaje no cambia esos números ni añade palabras nuevas a su lista de términos conocidos.

### ¿Entonces por qué decimos que aprende?

Porque esos números se calcularon usando ejemplos durante el entrenamiento. Después se guardan y se utilizan. Enseñarle y utilizar lo aprendido son momentos diferentes.

### ¿Dónde ocurre eso en Python?

En `entrenamiento/p03_comparacion.py` encontrarás `.fit()`: es la llamada que realiza el aprendizaje. En `entrenamiento/p05_inferencia.py`, `predecir()` utiliza lo aprendido para clasificar un mensaje.

## 3. Entender las frases que utilizamos

### ¿De dónde vienen las frases?

Hay 400 frases creadas para este proyecto con ayuda de IA. No son conversaciones recogidas de personas. Por eso las llamamos **sintéticas**: inventadas para preparar el trabajo.

### ¿Qué son corpus y dataset?

**Corpus** significa colección de textos. **Dataset** significa conjunto de datos organizado para trabajar. En este proyecto hablamos de las frases, sus categorías y los datos que permiten ordenarlas.

### ¿Qué son las 200 situaciones?

Cada situación está expresada con dos frases parecidas. Por ejemplo: «Aprobé y estoy feliz» y «Me alegra haber pasado el examen». Son ejemplos para entender la organización.

Decir algo parecido con otras palabras se llama parafrasear. Cada forma es una **paráfrasis**. Hay 200 parejas de frases: 400 frases en total.

### ¿Qué significan train, validación y test?

Separamos las frases según para qué vamos a utilizarlas:

| Nombre | Significado | Cantidad |
|---|---|---|
| Train | Las frases para enseñarle. Train se refiere a entrenamiento en inglés. | 240 |
| Validación | Las frases para comparar alternativas y elegir una. | 80 |
| Test | Las frases de la prueba final para medir al elegido. | 80 |

### ¿Por qué no le enseñamos todas?

Queremos comprobar si puede responder sobre situaciones diferentes a las que usó para aprender. Es parecido a darle a un estudiante preguntas distintas a sus ejercicios de estudio.

### ¿Por qué las dos frases de una situación permanecen juntas?

Porque enseñar una frase y examinarlo con otra casi equivalente haría que la prueba se pareciera demasiado a lo que ya conoce. Mantenemos ambas en el mismo grupo.

### ¿Qué es la fuga de datos?

Que información de la prueba termine ayudando indebidamente a preparar o elegir el modelo. Es parecido a estudiar sabiendo las respuestas del examen. Separar las situaciones y aprender las palabras solo con entrenamiento ayuda a evitarlo.

### ¿Por qué no usamos el identificador de cada frase para clasificar?

El identificador es un nombre para encontrarla. En este proyecto contiene información sobre la categoría y podría revelar la respuesta. Por eso usamos únicamente el texto para decidir y la categoría como respuesta que enseñar.

### ¿Qué problema tiene que las frases sean inventadas?

Pueden ser poco variadas y compartir palabras o estilos. Acertar con ellas no demuestra que funcionará igual con personas reales. Las categorías tampoco fueron comprobadas por varios especialistas.

## 4. Entender cómo llega a una categoría

### ¿Por qué limpia los mensajes?

Para tratar de forma parecida «FELIZ» y «feliz», por ejemplo. Pasa todo a minúsculas y elimina tildes, signos, números y emojis. También convierte ñ en n y ordena los espacios.

Eso se llama **normalizar**. Se hace en `normalizar()`, dentro de `entrenamiento/p01_datos.py`. Conserva palabras como «no», aunque eso no garantiza interpretar bien todas las negaciones.

### ¿Por qué convierte palabras en números?

Porque la herramienta que clasifica hace cuentas. Necesita números que representen las palabras de la frase.

Ejemplo sencillo: si las columnas fueran «feliz» y «triste», podríamos representar «feliz» con `[1, 0]` y «triste» con `[0, 1]`. Es un ejemplo didáctico; la representación del ganador tiene más columnas y utiliza otros valores.

### ¿Qué son vocabulario, vectorizador y características?

- **Vocabulario:** lista de palabras y parejas de palabras que aprendió a utilizar.
- **Vectorizador:** herramienta que convierte el texto en una fila de números usando esa lista.
- **Característica:** cada entrada numérica que utiliza para clasificar. Aquí puede representar una palabra o una pareja.

El vocabulario se aprende con las 240 frases de entrenamiento y después queda fijo.

### ¿Qué es TF-IDF?

Es la forma de calcular los valores que representan las palabras. Considera cuánto aparece un término en una frase y qué tan común es entre las frases de entrenamiento.

Un término poco común puede recibir más peso en esa representación. Eso no significa que toda palabra rara ayude a reconocer emociones. Después el clasificador aprende cómo utilizar esos valores.

### ¿Qué es un bigrama?

Una pareja de palabras consecutivas, como «estoy feliz». El ganador utiliza palabras solas y parejas. Esto conserva algo del orden, pero no hace que entienda cualquier frase.

### ¿Qué es un Pipeline?

Unir dos pasos: transformar las palabras en números y clasificar esos números. Se guardan juntos para volver a utilizar lo aprendido en ambos pasos.

### ¿Por qué se llama regresión logística multinomial?

Es el nombre de la herramienta matemática elegida. Su trabajo aquí es escoger una categoría. «Multinomial» indica que distingue varias categorías: en nuestro caso, cinco.

### ¿Cómo decide?

Combina los números de la frase con los pesos aprendidos y calcula una puntuación para cada categoría. El cálculo llamado **softmax** convierte esos resultados en cinco puntuaciones que suman aproximadamente uno. La mayor es su primera opción.

### ¿Por qué a veces pide más contexto?

Porque no siempre mostramos su primera opción. Exigimos tres condiciones:

1. La mayor puntuación debe llegar a 0,30.
2. Debe superar a la segunda por al menos 0,10.
3. Debe quedar alguna palabra o pareja conocida para analizar.

Si falla una condición, muestra incertidumbre y pide más información.

### ¿Qué son umbral y abstención?

**Umbral** significa un límite que debe alcanzarse. Aquí el límite para la mayor puntuación es 0,30.

**Abstención** significa no mostrar una categoría. No es una sexta emoción aprendida. Por ejemplo, si alegría obtiene 0,42 y tristeza 0,40, hay tan poca diferencia que el programa pide más contexto. Son números inventados para explicar la regla.

### ¿Qué pasa con una palabra desconocida?

Se ignora. Si no queda ninguna palabra o pareja conocida, pide contexto. Si quedan otras conocidas, puede dar una categoría, aunque también puede equivocarse.

### ¿Una puntuación de 0,60 en tristeza significa estar 60% triste?

No. Es un valor que utiliza el modelo para comparar sus opciones. No mide cuánto siente una persona ni es una probabilidad comprobada sobre personas reales.

### ¿Qué significan las palabras destacadas en el análisis?

Son términos presentes que favorecieron la primera categoría frente a la segunda. Muestra hasta cinco aportes positivos. No son todos los cálculos de la decisión ni una explicación psicológica de la persona.

## 5. Entender cómo elegimos y cuánto acertó

### ¿Qué alternativas comparamos?

Tres formas de usar la herramienta logística, otra herramienta llamada Naive Bayes y una comparación sencilla que siempre repite la categoría más frecuente en entrenamiento.

Las tres primeras cambian cómo representan las palabras: conteos, TF-IDF con palabras o TF-IDF con palabras y parejas. Naive Bayes es otra manera matemática de combinar asociaciones entre términos y categorías.

### ¿Hay cinco modelos respondiendo en el chat?

No. Probamos cinco durante la preparación y elegimos uno usando las 80 frases de validación. El chat utiliza al ganador: `logistica_bigramas`, logística con TF-IDF de palabras y parejas.

### ¿Cómo sabemos cuál ganó?

Medimos sus respuestas en validación con una medida llamada macro-F1. Elegimos la alternativa con el valor mayor. La prueba final se utiliza después para medir al elegido.

### ¿Qué significan las medidas que aparecen?

Una **métrica** es una medida del resultado. Estas son las principales:

| Medida | Pregunta que responde |
|---|---|
| Accuracy | De todas las frases, ¿qué proporción acertó? |
| Precision de alegría | De las frases que llamó alegría, ¿cuántas tenían esa categoría? |
| Recall de alegría | De todas las frases con categoría alegría, ¿cuántas reconoció? |
| F1 de una categoría | ¿Qué resultado obtiene al combinar precision y recall? |
| Macro-F1 | ¿Cuál es el promedio de las F1 de las cinco categorías, dando igual importancia a cada una? |

Las preguntas de precision y recall se pueden hacer para cada categoría, no solo alegría.

### ¿Qué es la matriz de confusión?

Una tabla que cruza las categorías conocidas con las que eligió el programa. Sirve para ver, por ejemplo, cuántas frases de alegría confundió con tristeza. Las filas son las categorías conocidas y las columnas son las respuestas del modelo.

### ¿Qué números conviene recordar?

| Resultado del proyecto | Valor |
|---|---|
| Frases utilizadas | 400, creadas con ayuda de IA. |
| Para aprender, elegir y examinar | 240, 80 y 80. |
| Aciertos en la prueba final | 53 de 80: 66,25%. |
| Macro-F1 de validación | Aproximadamente 0,787. |
| Macro-F1 de la prueba final | Aproximadamente 0,653. |
| Mensajes en los que aceptó mostrar categoría | 60 de 80. |
| Aciertos entre esos aceptados | 44 de 60: aproximadamente 73,33%. |
| Categoría más débil | Alegría: reconoció 4 de sus 16 frases. |

### ¿Por qué aparecen dos porcentajes de aciertos?

53/80 cuenta todas las respuestas internas de la prueba final. 44/60 cuenta solo los mensajes donde cumplió las condiciones para mostrar categoría.

Son preguntas diferentes. Aunque pida contexto en 20 mensajes, entre los otros 60 todavía hay 16 errores.

### ¿Qué significa cobertura?

Qué proporción de mensajes aceptó etiquetar. En esta prueba fueron 60 de 80: 75% de cobertura.

### ¿Por qué acertó menos en la prueba final que en validación?

Son otras situaciones y ambos grupos son pequeños. Además, elegimos al que mejor salió en validación, lo cual puede favorecer esa cifra. No tenemos datos suficientes para atribuir toda la diferencia a una sola causa.

### ¿Qué es sobreajuste?

Adaptarse demasiado a los ejemplos conocidos y responder peor a otros. Es parecido a memorizar ejercicios sin poder resolver uno distinto.

### ¿Qué muestra la curva de aprendizaje?

Un gráfico de lo que pasó al enseñarle con 60, 120, 180 y 240 frases. Aquí mejoró la medida de validación al utilizar más frases. Eso no demuestra que cualquier frase añadida vaya a ayudar.

### ¿Qué es bootstrap?

Una comprobación adicional que forma otras muestras tomando situaciones ya evaluadas y vuelve a calcular la medida con sus respuestas guardadas. Sirve para explorar cuánto varía el resultado dentro de ese material. No vuelve a entrenar ni a pedir predicciones al modelo.

### ¿Qué mejoraríamos?

Añadir situaciones más variadas, revisar mejor sus categorías y medir con frases nuevas que no hayan servido para preparar el modelo. Si se utilizan mensajes de personas, hacen falta autorización y protección de su privacidad.

## 6. Entender el funcionamiento de la página

### ¿De dónde sale la respuesta escrita?

De `conversacion.py`. Ya escribimos varias respuestas por categoría. El programa elige una según la categoría del mensaje y alterna los textos disponibles según el turno. No inventa cada respuesta palabra por palabra.

### ¿Cómo se conectan la página y Python?

La página envía el mensaje a una dirección llamada `/api/chat`. El archivo `app.py` lo recibe y coordina la respuesta utilizando Flask, una librería de herramientas para crear servicios web con Python.

A esa forma acordada de pedir y devolver información se la llama **API**. Los datos se envían con campos con nombre, como `message` y `reply`; ese formato se llama **JSON**.

### ¿Recuerda el diálogo?

La página mantiene temporalmente los mensajes para mostrarlos. Sin embargo, el modelo clasifica cada mensaje por separado, sin recibir toda la conversación. Una palabra como «sí» puede darle muy poca información.

### ¿Qué hace ante expresiones de crisis?

`seguridad.py` busca ciertas expresiones antes de clasificar emociones. Si encuentra una, muestra apoyo y omite el análisis emocional. Son reglas escritas por quienes desarrollan el programa; no una categoría que aprendió el modelo.

Las reglas pueden activarse con citas o mensajes sobre terceros y pueden pasar por alto expresiones no contempladas. No pueden asegurar si alguien está a salvo.

### ¿Por qué mantiene la precaución en mensajes siguientes?

La página envía una señal de que ya hubo una alerta. Eso hace que el programa siga respondiendo con precaución. «Nueva conversación» reinicia esa señal. No significa que comprenda el diálogo completo.

### ¿Qué pasa si el modelo no está disponible?

Las reglas de precaución pueden seguir funcionando porque se comprueban antes de usarlo. En mensajes normales, la aplicación informa que no puede clasificar en ese momento.

### ¿Dónde se guarda lo aprendido?

En `entrenamiento/artefactos/modelo.joblib`. Al iniciar se carga ese archivo con lo aprendido y el límite de aceptación. Si falta, el programa puede reconstruir el modelo en memoria usando solo las frases de entrenamiento; no abre la prueba final para hacerlo.

### ¿Laboratorio vuelve a enseñarle?

No. Muestra resultados guardados. Abrir esa pantalla no repite el entrenamiento ni la prueba final.

### ¿Se guardan las conversaciones?

La aplicación no las almacena permanentemente. La página las mantiene temporalmente. Si el usuario exporta, decide guardar una copia local.

### ¿Qué son las 61 pruebas automáticas?

Comprobaciones de funcionamiento, como qué ocurre con un mensaje vacío o una palabra desconocida. Se ejecutan con la herramienta `pytest` y ejemplos de desarrollo.

Pasar esas 61 comprobaciones no significa acertar 61 emociones. La medida de aciertos emocionales es la de 53/80.

## 7. Qué archivo abrir si preguntan por el código

Una **función** es un grupo de instrucciones para una tarea. En Python suele verse como `def nombre(...)`. Los nombres que terminan en `()` en esta tabla son funciones.

Los archivos `p00` a `p06` están en la carpeta `entrenamiento/`.

| Si preguntan… | Abre este archivo | Busca esto y explica su tarea |
|---|---|---|
| ¿Dónde están las frases? | `datos de entrenamiento/corpus_original_E/escenarios.txt` | Una categoría y sus parejas de frases. |
| ¿Cómo las preparan? | `p01_datos.py` | `preparar()`: separa las frases para aprender, elegir y examinar. |
| ¿Cómo separan texto y respuesta? | `p01_datos.py` | `xy()`: entrega las frases y sus categorías en dos listas. |
| ¿Cómo limpian? | `p01_datos.py` | `normalizar()`: convierte el texto a una forma uniforme. |
| ¿Qué herramientas usan? | `p02_modelos.py` | `crear_modelo()`: construye la conversión a números y la herramienta que clasifica. |
| ¿Dónde aprende? | `p03_comparacion.py` | `.fit()`: aprende con las frases de entrenamiento. |
| ¿Dónde eligen? | `p03_comparacion.py` | `comparar()`: compara con las frases de validación. |
| ¿Dónde cuentan los aciertos finales? | `p04_evaluacion.py` | `evaluar_una_vez()`: comprueba al elegido y guarda el resultado. |
| ¿Dónde analiza un mensaje? | `p05_inferencia.py` | `predecir()`: transforma el mensaje, calcula puntuaciones y decide si muestra categoría. |
| ¿De dónde sale la respuesta escrita? | `conversacion.py` | `RESPUESTAS` contiene los textos; `responder()` selecciona uno. |
| ¿Dónde revisan expresiones de crisis? | `seguridad.py` | `detectar_riesgo()`: comprueba las reglas. |
| ¿Dónde conectan todo? | `app.py` | `chat()`: coordina un mensaje recibido desde la página. |

El capítulo 26 del [manual](explicacion-completa-del-proyecto-animo.md) también te ayuda a encontrar estas partes.

## 8. Cómo hacer la demostración

1. Explica qué intenta reconocer y que las frases se crearon con ayuda de IA.
2. Muestra una situación y sus dos frases en `escenarios.txt`.
3. Abre `p03_comparacion.py` y señala `.fit()` como el momento donde aprende.
4. Escribe una frase en el chat. Cuenta lo que realmente aparezca: puede acertar, equivocarse o pedir más información.
5. Prueba `zzqwwx 🛸` para mostrar un mensaje sin palabras conocidas y explicar por qué pide contexto.
6. Abre Laboratorio. Explica el 53/80 y muestra que alegría fue la categoría más difícil.
7. Si muestras una expresión de crisis, aclara que estás comprobando una regla del programa. Después pulsa «Nueva conversación» para reiniciar la señal de precaución.

Para iniciar el proyecto, con las herramientas ya instaladas, abre una terminal en su carpeta y ejecuta:

```sh
source .venv/bin/activate
python app.py
```

La primera línea activa el espacio de Python preparado para el proyecto. La segunda inicia la aplicación. Abre la dirección que imprime la terminal. Mantén la terminal abierta durante la demostración; Ctrl+C detiene el programa.

No necesitas volver a entrenar para mostrar este avance. Si después quieres aprender a hacerlo, el capítulo 32 del manual explica cada paso y comando.

## 9. Cómo practicar antes de exponer

Intenta responder estas preguntas sin mirar:

1. ¿Qué recibe el programa y qué devuelve?
2. ¿Cómo le enseñamos?
3. ¿Por qué no utilizamos las mismas frases para todo?
4. ¿Qué hace cuando escribimos un mensaje nuevo?
5. ¿De dónde salen las respuestas escritas?
6. ¿Qué significa acertar 53 de 80?
7. ¿Qué limitaciones tiene?

Sobre el equipo, explica lo que realmente hizo cada integrante y la ayuda de IA que utilizaron. Conviene que todos puedan contar el recorrido básico, aunque cada uno profundice en una parte.

Si una pregunta supera lo que sabes, puedes decir «Necesito revisar ese detalle para responderlo bien». Abre el archivo correspondiente si sabes dónde está. Entender el proyecto también incluye reconocer qué parte todavía necesitas estudiar.

Para una explicación más pausada y con ejemplos, vuelve al [manual completo](explicacion-completa-del-proyecto-animo.md).
