# Explicación completa del proyecto Ánimo   
> Guía de estudio de principio a fin, basada en el código y la evidencia de esta entrega. Está pensada para entender, reconstruir y defender el proyecto. Los ejemplos didácticos se identifican como tales; las cifras del experimento proceden de sus resultados guardados. Incluye teoría, decisiones, ejecución, lectura del código, resultados, preguntas y referencias al código real.   

## Cómo leer este documento   
Lee primero los capítulos 1 a 14 para entender el aprendizaje automático. Continúa con la aplicación y la seguridad. Después estudia el inventario de funciones y sus fuentes. Al final están el informe completo y las preguntas de defensa, incorporados aquí para que la explicación sea autosuficiente.   
No necesitas memorizar cada línea. Para cada componente debes poder responder: qué recibe, qué devuelve, qué modifica, cuándo se ejecuta, por qué existe y qué podría salir mal. Si entiendes esas seis preguntas, puedes seguir una explicación aunque la profesora cambie el ejemplo.   
El documento describe la versión actual, con los 14 archivos Python comentados y formateados. Los comentarios son breves, en minúsculas y sin adornos; explican instrucciones y campos importantes. El formato cambia la distribución visual del código, no su lógica. No atribuye cuatro autores humanos ni dos meses de trabajo real. Se utilizó asistencia de IA en el desarrollo y en la redacción de datos, y eso se debe declarar según las reglas del curso. El alcance se demuestra con archivos y evidencia, no con una historia ficticia.   
## 1. Qué es Ánimo y qué problema aborda   
Ánimo recibe mensajes breves en español, clasifica señales de alegría, tristeza, enojo, ansiedad o neutralidad y muestra respuestas previamente redactadas. Permite inspeccionar puntuaciones y palabras que favorecen una clase frente a otra.
El objetivo académico es recorrer datos, representación, comparación, selección, evaluación y aplicación. La etiqueta describe el texto; no mide el estado interno de una persona ni diagnostica trastornos. No se evaluó que conversar con el prototipo mejore el bienestar.

| Componente | Qué aprende o hace |
| --- | --- |
| Vocabulario e IDF | Aprende términos y frecuencias documentales de train. |
| Clasificador | Ajusta asociaciones entre características y etiquetas. |
| Selección y abstención | Se deciden con validación y reglas predefinidas. |
| Limpieza, precaución y respuestas | Son programación explícita; no aprenden al conversar. |

## 2. Conceptos básicos para entenderlo desde cero   
**Inteligencia artificial** es el campo general. **Machine Learning** es una forma de construir sistemas ajustando su comportamiento con datos. **NLP o procesamiento de lenguaje natural** estudia tareas sobre lenguaje humano; aquí se usa una representación clásica de palabras. **Aprendizaje supervisado** significa que durante el ajuste se conocen las respuestas deseadas de los ejemplos.   
Una **observación** es una frase del corpus. La **etiqueta** es su categoría deseada. Una **característica** es una variable numérica que el algoritmo puede usar, como el peso TF-IDF de «feliz» o de «estoy feliz». El **modelo** es la transformación aprendida que usa esas características para producir puntuaciones de clase.   
**Clasificación** predice categorías. **Regresión numérica** predice cantidades continuas. Aunque se llame regresión logística, en este proyecto la logística resuelve clasificación. No se intenta calcular una nota emocional continua ni una escala clínica.   
**Entrenar o ajustar** significa modificar parámetros a partir de ejemplos. **Inferir o predecir** significa usar los parámetros ya fijados con una entrada. **Evaluar** significa contrastar predicciones con etiquetas conocidas mediante métricas. Son operaciones distintas: predecir una frase del usuario no reentrena el modelo.   
**Generalizar** significa funcionar con ejemplos distintos de los utilizados para ajustar. Acertar train no demuestra generalización: el modelo pudo memorizar asociaciones particulares. Por eso se necesitan datos separados.   
Un **parámetro** es aprendido, como un coeficiente. Un **hiperparámetro** se fija desde fuera del ajuste, como C de logística, alpha de NB o la elección de bigramas. El **umbral de abstención** es una decisión posterior que usa las salidas del modelo, no otro peso aprendido por logística.   
## 3. Visión completa de los dos recorridos   
Hay dos recorridos distintos:

1. **Experimento:** preparar datos → dividir por grupos → entrenar candidatos con train → elegir con validación → evaluar una vez con test → analizar resultados.
2. **Uso del chatbot:** recibir mensaje → revisar precaución → transformar el texto con el vectorizador aprendido → predecir o pedir contexto → mostrar una respuesta redactada.

Usar el chatbot no vuelve a entrenar el modelo ni ejecuta el test.

## 4. Cómo se construyó la solución desde cero   
Se delimitó la clasificación supervisada y se relacionó con los apuntes. Después se fijó un protocolo con clases, datos, candidatos, división, selección y evaluación antes de conocer el resultado final.
Se redactó y validó el corpus, se dividió por grupos y se entrenaron candidatos usando train. Validación eligió ganador y umbral; se guardaron modelo y huellas antes de abrir test. Una inferencia final produjo predicciones, métricas y análisis de errores.
La aplicación incorporó después el modelo, las respuestas redactadas, la precaución y las pruebas de software. Los cambios de comentarios, formato o interfaz no fueron nuevos ajustes del clasificador. Estas etapas describen dependencias, no un calendario certificado de trabajo humano.

## 5. Organización del proyecto y razón de cada carpeta   
```text
CHATBOT(E)/
├── app.py                      servidor y API
├── index.html                  interfaz con CSS y JavaScript
├── conversacion.py             respuestas redactadas
├── seguridad.py                reglas de precaución
├── README.md                   ejecución y decisiones
├── INFORME_RESULTADOS.md        resultados y limitaciones
├── requirements.txt            dependencias directas
├── requirements-lock.txt       versiones del entorno original
├── entrenamiento/
│   ├── __init__.py              paquete Python
│   ├── __main__.py              etapas desde la terminal
│   ├── p00_configuracion.py     constantes, rutas y huellas
│   ├── p01_datos.py             limpieza, grupos y división
│   ├── p02_modelos.py           modelos, métricas y abstención
│   ├── p03_comparacion.py       entrenamiento y selección
│   ├── p04_evaluacion.py        evaluación final reservada
│   ├── p05_inferencia.py        carga, predicción y explicación
│   ├── p06_informe.py           gráficos y CSV de errores
│   ├── datos/                  fuente, CSV y manifiesto
│   ├── artefactos/              modelo.joblib
│   ├── resultados/              evidencia y figuras
│   └── pruebas/                 tests de metodología
├── tests/                       pruebas de API y precaución
└── docs/                        instalación y documentación
```
Los módulos definen funciones; importarlos no ejecuta todas las etapas. `python -m entrenamiento` inicia `__main__.py`. Los imports con punto buscan dentro del paquete. `.venv` contiene el entorno local; `__pycache__` y `.pytest_cache` son cachés, excluidas de Git. El manual de estudio también se mantiene fuera de Git.

## 6. Los datos: qué son, de dónde salen y qué significan   
El corpus tiene 400 frases en 200 escenarios: dos paráfrasis por escenario y 40 escenarios por clase. Es 100 por ciento sintético, redactado con asistencia de IA; no contiene conversaciones de participantes ni anotaciones de especialistas.

```text
[alegria]
Estoy feliz porque aprobé el examen | Pasé la evaluación y estoy muy contenta
```
El encabezado define la clase; cada línea contiene dos frases del mismo escenario. El programa materializa esas frases, no inventa nuevas al arrancar.
Cada fila contiene `id`, `grupo`, `texto`, `etiqueta`, `origen` y `particion`. Solo `texto` entra como predictor; `etiqueta` es la respuesta deseada. Los ids y grupos contienen la clase: usarlos en X revelaría la respuesta. `xy()` los excluye.
Alegría agrupa entusiasmo, gratitud y satisfacción; tristeza, pena y desánimo; enojo, molestia y frustración; ansiedad, preocupación anticipatoria; neutral, descripciones sin emoción explícita. Se asignó una etiqueta predominante; neutral no significa ausencia de sentimientos.
No hubo doble anotación ni acuerdo entre anotadores. Balancear clases no elimina el sesgo de vocabulario, estilo o situaciones académicas. Tampoco demuestra representatividad ni equidad demográfica.

## 7. Separación train, validación y test   

| Partición | Escenarios por clase | Frases por clase | Frases totales | Función |
| :------------------- | :---------------------------- | :------------------------ | :---------------------- | :----------------------------------------------- |
| Train | 24 | 48 | 240 | Aprender representación y clasificador |
| Validación | 8 | 16 | 80 | Elegir candidato y umbral |
| Test | 8 | 16 | 80 | Medir al ganador después de congelarlo |

La división se hace sobre grupos completos y dentro de cada clase. Primero se ordenan los identificadores para partir de un orden estable. Después se barajan con un generador local `random.Random(2026)`. Se asigna el primer 60% a train, el siguiente 20% a validación y el restante a test.   
La semilla hace reproducible la secuencia pseudoaleatoria dentro de las condiciones del experimento. El valor 2026 no tiene significado emocional ni es un parámetro que se haya elegido para mejorar test. Cambiar semillas hasta encontrar una cifra agradable introduciría selección sobre el resultado final.   
Separar frases individualmente permitiría que una paráfrasis estuviera en train y su casi equivalente en test. Entonces se mediría en parte reconocer una variante ya enseñada. Agrupar evita ese caso declarado. No elimina todas las similitudes entre escenarios distintos: pueden compartir «estoy feliz», contextos de exámenes y estructuras lingüísticas. Compartir palabras es normal en clasificación de texto; compartir duplicados de una misma observación requiere control especial.   
El control de duplicados compara textos normalizados antes de repartir. Detecta identidades después de retirar tildes y signos; no es un detector completo de semejanza semántica. El conocimiento del autor sobre todo el corpus tampoco convierte test en una evaluación externa ciega.   
## 8. Limpieza y normalización: operación por operación   
`normalizar()` recibe y devuelve texto; aplica reglas fijas, sin aprender ni descargar recursos:

1. `NFKC` unifica representaciones Unicode compatibles.
2. `casefold()` convierte a minúsculas.
3. `NFD` descompone letras y marcas de acento.
4. Retira marcas `Mn`: elimina tildes y convierte ñ en n.
5. La expresión `[a-z]+` conserva letras básicas; descarta números, signos y emojis.
6. Une las palabras con espacios simples.

Ejemplo didáctico: `¡NO estoy FELÍZ! 123` produce `no estoy feliz`. Después, el vectorizador admite tokens de al menos dos letras; por eso «a» o «y» no quedan como características aunque estén en el texto normalizado.
No se eliminan stopwords ni se aplica lematización. Se conservan negaciones como «no», pero eso no garantiza comprenderlas. «Contento» y «contenta» pueden ser características distintas. El mismo preprocesador se aplica al entrenar y predecir; `lowercase=False` evita repetir su conversión.

## 9. Convertir palabras a números   
### Representación y vocabulario
Una bolsa de palabras representa presencia o frecuencia de términos. Ejemplo didáctico: con columnas `[feliz, muy, triste]`, «muy feliz» es `[1, 1, 0]`. El vectorizador conserva el mapa término a columna; `transform()` no añade columnas para palabras desconocidas.
Un unigrama es una palabra y un bigrama dos tokens consecutivos. «No estoy feliz» puede producir `no`, `estoy`, `feliz`, `no estoy` y `estoy feliz`; no produce `no feliz`. Solo se conservan características aprendidas en train. Los unigramas del experimento tienen 587 columnas; unigramas y bigramas, 1651. Son características, no emociones.

### TF-IDF y normalización
TF-IDF combina frecuencia de un término con su rareza entre documentos de train. Se utiliza frecuencia sublineal, IDF suavizado y normalización L2:

```text
tf = 1 + ln(conteo) si el termino aparece
idf = ln((1 + documentos_train) / (1 + documentos_con_termino)) + 1
peso = tf por idf
x_j = peso_j / raiz de la suma de los cuadrados de los pesos
```
La ausencia produce cero; no se calcula el logaritmo de cero. Una palabra presente en todos los documentos tiene IDF igual a uno. L2 reduce el efecto de longitud, pero no convierte características en probabilidades. El IDF no conoce emociones; el clasificador aprende las asociaciones con etiquetas.

### Matriz dispersa
Guarda valores no nulos y sus posiciones porque cada frase usa pocas columnas. Cero significa ausencia de una característica. `getnnz(axis=1)` cuenta entradas presentes por frase; `indices` y `data` permiten recorrerlas para explicar una predicción. Si no queda vocabulario conocido, el sistema se abstiene.

## 10. Regresión logística multinomial   
### Predicción y aprendizaje
Cada clase tiene pesos e intercepto. Para la clase k:

```text
z_k = b_k + suma de w_kj por x_j
p_k = exp(z_k) / suma de exp(z_l) para todas las clases
```
El primer cálculo es un logit lineal; el segundo es softmax. Sus cinco salidas suman aproximadamente uno. `argmax` elige la mayor y `classes_` traduce el índice a etiqueta. Una entrada fuera del dominio también recibe puntuaciones: softmax no crea una clase desconocida.
Durante `fit`, scikit-learn ajusta pesos e interceptos para reducir la entropía cruzada multinomial, junto con regularización. La pérdida de datos se puede expresar como la suma negativa de logaritmos de la probabilidad asignada a cada etiqueta verdadera.
`C=4` es un hiperparámetro fijado de antemano: menor C implica más regularización. `max_iter=2000` limita las iteraciones del optimizador; no obliga a realizar todas. Ninguno se aprende durante el ajuste. La comparación convierte `ConvergenceWarning` en error para no aceptar silenciosamente falta de convergencia.
Se usa el solver predeterminado de la versión fijada y `random_state=2026`. La reproducibilidad depende también de datos, orden, versiones y operaciones numéricas; no todo solver utiliza esa semilla.
F1 perfecta en train con peor validación muestra que ajustar los ejemplos conocidos no basta. Puede haber sobreajuste y variación de muestra; no identifica una causa única ni demuestra que otro C lo resolvería.

## 11. Los cinco candidatos y la comparación   

| Nombre interno | Representación | Clasificador | Pregunta que ayuda a estudiar |
| :--------------------------- | :------------------------------------ | :-------------------------------- | :------------------------------------------------------------------- |
| logistica\_tfidf | TF-IDF, unigramas | Logística | Referencia principal explicable |
| logistica\_conteos | Conteos, unigramas | Logística | ¿Cambia el resultado al ponderar términos? |
| logistica\_bigramas | TF-IDF, unigramas y bigramas | Logística | ¿Aporta información el orden local? |
| naive\_bayes | TF-IDF, unigramas | MultinomialNB | ¿Cómo funciona otra familia sencilla? |
| referencia | TF-IDF en el Pipeline | Dummy de clase frecuente | ¿Aprender características supera una decisión constante? |

La referencia no aprovecha las palabras para distinguir clases, aunque comparte la estructura de Pipeline por uniformidad. El tamaño de vocabulario registrado para ella no significa que esté clasificando por contenido. Con las clases balanceadas hay empate en frecuencias y la elección constante sigue el comportamiento determinista de la implementación.   
### 11.1 Cómo razona Naive Bayes   
El teorema de Bayes relaciona la probabilidad de una clase dado un texto con la probabilidad del texto dado la clase y la probabilidad previa de clase. La variante multinomial aproxima el texto usando características consideradas condicionalmente independientes dada la etiqueta.   
En logaritmos puede expresarse la decisión como una previa de clase más una suma de valores de características multiplicados por log-probabilidades. Se suman logaritmos para evitar multiplicar muchas cantidades pequeñas. `alpha=1` suaviza estimaciones para evitar probabilidades nulas de características sin observaciones en una clase.   
La independencia condicional es una simplificación fuerte: las palabras del lenguaje están relacionadas. También hay una precisión importante: aquí se usa NB con valores TF-IDF no negativos, una variante práctica; no hay que presentarlo como un modelo generativo exacto de conteos enteros.   
### 11.2 Qué es un Pipeline y por qué importa   
Un Pipeline encadena `vectorizador` y `clasificador`. En `fit(x_train, y_train)`, aprende vocabulario/IDF y ajusta el clasificador. Al predecir, usa `transform()` y después el estimador ya ajustado.   
Una fuga común sería aprender el vocabulario o IDF usando todo el corpus antes de dividir. Aunque no se usaran las etiquetas de test, se estaría usando información de su distribución para preparar el modelo. Aquí la división existe antes del ajuste y cada candidato aprende su transformación exclusivamente con train.   
Guardar solo el clasificador no bastaría: también hay que conservar el significado de cada columna y sus pesos IDF. Por eso se serializa el Pipeline completo.   
## 12. Selección, curva y abstención   
Los candidatos se ajustan con train y el ganador se elige por macro-F1 de validación sin redondear; un empate conserva el primero de `NOMBRES`. Las medidas de train muestran la brecha, pero no deciden al ganador. No se comparan los candidatos sobre test para seleccionarlos.
La curva ajusta modelos auxiliares con 25, 50, 75 y 100 por ciento de los grupos de train. Usa subconjuntos anidados, grupos completos y la misma validación: 60, 120, 180 y 240 frases. No sustituye el modelo guardado ni constituye validación cruzada.

Para mostrar una etiqueta se exige simultáneamente:
1. Mayor puntuación al menos igual al umbral.
2. Diferencia de al menos 0,10 respecto a la segunda clase.
3. Alguna característica del vocabulario aprendido.

Si falla una condición, la etiqueta visible es `incierto`: una abstención, no una sexta clase entrenada. Ejemplo didáctico: scores 0,42 y 0,40 tienen margen 0,02 y se rechazan incluso con umbral 0,30.
Los umbrales probados fueron 0,30; 0,40; 0,50; 0,60 y 0,70. Se exige accuracy selectiva de al menos 0,80 y cobertura de al menos 0,50 en validación; se maximiza cobertura y se desempata con el menor umbral. Si ninguno cumple, se usa 0,50.
Se eligió 0,30: aceptó 58 mensajes de validación, con 49 aciertos. 0,40 también cumplía, pero cubría menos. Umbrales altos pueden acertar todos los pocos casos aceptados sin ser mejores para conversar. La meta de validación no garantiza el resultado de test; no se cambió el umbral tras observarlo.

## 13. Métricas y cómo calcularlas a mano   
Accuracy es aciertos divididos por total. En test hubo 53 aciertos de 80: 53/80=0,6625. El porcentaje correspondiente es 66,25%. No es el porcentaje de personas ayudadas ni de conversaciones seguras.   
Para una clase se trata esa categoría como positiva y todas las demás como negativas. TP son ejemplos de la clase correctamente asignados; FP son ejemplos de otras clases asignados a ella; FN son ejemplos reales de la clase enviados a otra; TN son los restantes.   
```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
macro-F1  = promedio de las cinco F1

```
Precision y accuracy pueden traducirse informalmente ambas como precisión, por eso conviene usar los nombres exactos. Recall es recuperación o sensibilidad de la clase, sin implicar sensibilidad clínica del sistema de seguridad.   
Para alegría en la matriz real: TP=4; había 16 mensajes reales de alegría, por lo que FN=12. La columna de predicción alegría suma 12, por lo que FP=8. Precision=4/12=1/3; recall=4/16=0,25; F1=2×4/(2×4+8+12)=2/7, aproximadamente 0,286. La clase con la misma cantidad de ejemplos puede funcionar mucho peor que otras.   
El soporte es cuántos ejemplos reales hay por clase: aquí 16 en cada clase de test. `weighted avg` pondera por soportes y `macro avg` da el mismo peso a cada clase. En este test balanceado coinciden sus promedios de F1; eso no los vuelve conceptos idénticos en cualquier dataset.   
`zero_division=0` establece el valor a informar cuando una división no está definida por falta de positivos predichos o reales. Evita warnings o valores indefinidos en las tablas, pero no crea información que faltaba.   
### 13.1 Leer la matriz de confusión   
Las filas son etiquetas reales y las columnas predicciones, en el orden alegría, ansiedad, enojo, neutral y tristeza. La diagonal contiene aciertos. Una celda fuera de la diagonal muestra una confusión concreta. La suma de cada fila es soporte; la suma de cada columna cuenta cuántos mensajes se asignaron a esa categoría.   
La matriz permite observar qué clases se confunden, pero no explica causalmente por qué. Hay que mirar textos y características para formular hipótesis, y evaluarlas después con nuevos datos.   
### 13.2 Cobertura y accuracy selectiva   
En test se aceptaron 60 de 80 mensajes: cobertura 0,75. Entre esos 60 hubo 44 aciertos y 16 errores: accuracy selectiva 44/60≈0,7333. Entre los 20 rechazados había 9 predicciones internas correctas y 11 erróneas.   
Por eso abstenerse tiene dos efectos: evita mostrar algunas etiquetas equivocadas y también oculta algunas que habrían sido correctas. No elimina todos los errores. Comparar 0,7333 selectivo con 0,6625 total sin aclarar denominadores sería engañoso.   
## 14. Test único, congelación e incertidumbre   
### 14.1 Qué se guarda antes de test   
Se guarda el Pipeline ganador junto con su umbral y nombre, los resultados de candidatos y curva, las versiones, los hashes del artefacto y manifiesto y los hashes de los módulos p00–p05. El informe p06 no decide el modelo y puede regenerar presentaciones desde evidencia guardada.   
SHA-256 resume bytes de un archivo. Si cambian, la huella esperada deja de coincidir con enorme probabilidad. No es cifrado ni una firma de autoría. Una persona que manipule deliberadamente datos y hashes puede eludir el mecanismo: aquí sirve para auditar cambios accidentales y fijar la correspondencia experimental.   
### 14.2 Cómo se impide repetir accidentalmente la apertura   
Si existe `test_final.json`, `evaluar_una_vez()` lo devuelve antes de cargar el modelo. Si no existe, verifica los archivos y crea `test_reservado.json` con modo `x`. Ese modo falla si la reserva ya existe. La reserva se crea antes de cargar la partición de test.   
Después se llama una sola vez a `predict_proba(textos)` sobre el lote. Las etiquetas se obtienen de esas mismas salidas con argmax; no hace falta una segunda llamada a `predict()`. Se guardan scores, etiquetas, aceptación y métricas. La reserva se completa con la huella del resultado.   
Si la ejecución se interrumpe entre reservar y guardar, queda una reserva sin resultado. El código impide reabrir silenciosamente; requiere auditar lo ocurrido. Borrar el marcador para probar otra vez escondería esa información.   
Existe un límite adicional: cuando un resultado final ya está presente se reutiliza; ese retorno temprano no vuelve a comprobar todos sus hashes. Los registros permiten verificar integridad, pero no todo comando realiza automáticamente una auditoría completa de cualquier archivo existente. No se debe exagerar lo que garantiza el código.   
**Estado de esta versión:** los hashes de código guardados corresponden al experimento original. Después se sustituyeron cadenas de documentación por comentarios y se formatearon los archivos, por lo que los bytes de p00–p05 cambiaron. Se comprobó que su lógica ejecutable se conserva, y el hash del modelo sigue coincidiendo con el registrado. Esto no fue un nuevo entrenamiento. El código actual con resultado presente lo reutiliza; si falta ese resultado, los cambios de hashes pueden bloquear la evaluación. Conserva la evidencia original y crea otro experimento para entrenar de nuevo, en lugar de cambiar los hashes históricos.

### 14.3 Bootstrap de escenarios   
Una muestra de test pequeña tiene variabilidad. Para describirla se remuestrean con reemplazo los 40 escenarios de test y se incluyen ambas paráfrasis de cada escenario seleccionado. Se recalcula macro-F1 con predicciones ya guardadas en cada remuestreo.   
Se hacen 1000 réplicas con semilla fija y se toman los percentiles 2,5 y 97,5. El intervalo observado está aproximadamente entre 0,543 y 0,727. No se entrena ni se vuelve a llamar al modelo para este procedimiento.   
Remuestrear frases individualmente trataría como independientes paráfrasis que no lo son. Remuestrear grupos respeta esa dependencia declarada. Aun así, el intervalo es exploratorio: no corrige sesgo del corpus sintético, el proceso de selección ni la ausencia de una población real de referencia.   
### 14.4 Por qué no reentrenar con train + validación después   
El modelo servido mantiene exactamente el ajuste realizado con train. Si después de evaluarlo se ajustara con más ejemplos, sería otro artefacto y la cifra de test dejaría de describirlo directamente. En otra metodología podría definirse un reajuste antes del test, pero este protocolo eligió conservar el modelo seleccionado para facilitar trazabilidad.   
## 15. Guardar, cargar y explicar una predicción   
`modelo.joblib` guarda un diccionario con `pipeline`, `umbral` y `nombre`. El Pipeline contiene vocabulario, IDF y clasificador. Conversar no modifica ese archivo.
`cargar_modelo()` verifica el hash del artefacto. Si falta, lo reconstruye en memoria con el ganador fijo y solo train, sin sobrescribir evidencia. Requiere datos y versiones compatibles; no recupera archivos inexistentes por sí solo. Joblib puede ejecutar objetos al deserializar: usa únicamente artefactos de origen confiable.

### Salida de la predicción
- `etiqueta`, `candidata`, `alternativa`: salida visible y las dos clases de mayor score.
- `aceptada`, `score`, `margen`: decisión de abstención, mayor puntuación y diferencia con la segunda.
- `scores`, `evidencia`: puntuaciones por clase y hasta cinco aportes positivos de términos presentes.
- `motivo`, `modelo`: razón de aceptación o rechazo y candidato utilizado. Los valores NumPy se convierten a tipos serializables en JSON.

### Explicación contrastiva
Para logística, la diferencia entre dos logits es la diferencia de interceptos más la suma de `x_j` multiplicado por la diferencia de pesos de ambas clases. `explicar()` calcula esos aportes de términos presentes, los ordena y muestra hasta cinco positivos.
Usa `coef_` en logística y `feature_log_prob_` en Bayes; devuelve una lista vacía si no encuentra pesos compatibles. Omite interceptos, aportes negativos y términos restantes: los chips no reconstruyen toda la decisión. Son asociaciones aprendidas frente a una alternativa, no causas psicológicas. Unigrama y bigrama pueden aparecer juntos por ser columnas distintas.

## 16. Conversación y estado de sesión   
`RESPUESTAS` contiene tres respuestas para cada una de las cinco etiquetas y para incertidumbre. `responder(etiqueta, turno)` busca la lista correspondiente; si la etiqueta es desconocida usa incertidumbre. El índice se calcula como `turno % len(opciones)`.   
El operador módulo obtiene el resto de la división. Para tres respuestas los turnos recorren índices 0, 1, 2, 0, 1, 2. No hay una generación creativa aprendida ni un razonamiento distinto cada vez que cambia la frase de respuesta.   
La pestaña conserva el historial temporal; recargar o reiniciar la conversación lo borra. El servidor no lo persiste.
El clasificador analiza mensajes individuales. No se envía todo el diálogo como contexto semántico a un modelo de lenguaje. Si alguien contesta «sí» a una pregunta previa, esa palabra aislada puede resultar incierta. La bandera de precaución es una excepción explícita de estado, no una memoria lingüística general.   
## 17. Seguridad: funcionamiento, prioridad y límites   
Las expresiones de riesgo no constituyen una clase aprendida del modelo emocional. `seguridad.py` define expresiones regulares y las compila una vez al importar el módulo. `detectar_riesgo()` normaliza el texto y usa `any()` para comprobar si alguna regla encuentra coincidencia.   
La notación `r'...'` conserva barras invertidas en una cadena de Python para la expresión regular. `\b` indica límites de palabra, `\|` alternativas, `(?:...)` agrupa sin capturar, `?` hace opcional una parte y `\w*` permite una continuación de caracteres de palabra. Por ejemplo, una raíz como `suicid` seguida de `\w*` contempla distintas terminaciones. El patrón `da[n]?o` admite tanto «dano» normalizado como «dao» con la letra omitida.   
Además se comprueban dos variantes explícitas en el texto original, `su1cid` y `su!cid`. Si aparecen se añade una señal normalizada para activar las reglas. La variable `destino` de ese bucle se desempaqueta pero no se utiliza; no está implementando una sustitución general carácter por carácter. Explicar eso con precisión demuestra que se leyó el código real.   
Las reglas también contemplan menciones de terceros, negaciones y algunos mensajes en inglés. Eso no convierte todo el sistema en multilingüe. No hay una detección universal de ofuscaciones, errores o eufemismos.   
Cuando se activa precaución, el servidor entrega `safety=true`, `analysis=null` y un texto de apoyo. Ese texto indica acompañamiento humano, el recurso peruano solicitado por el proyecto y atención de emergencias si hay peligro inmediato o daño ya ocurrido; también declara los límites del prototipo. El texto exacto está en `RESPUESTA_RIESGO`, dentro de `seguridad.py`. El recurso fue consultado al desarrollar el proyecto mediante la herramienta de líneas de ayuda; la opción específica fue un requisito del usuario, no una conclusión del clasificador.   
La ausencia de coincidencia no significa que alguien esté a salvo. La presencia de una coincidencia tampoco diagnostica intención suicida. Una frase informativa o negada puede activar el mismo comportamiento conservador. Los casos de desarrollo lo hacen visible en lugar de esconderlo.   
### 17.1 Prioridad incluso sin modelo   
La comprobación de precaución ocurre antes de acceder al artefacto de ML. Por eso puede devolver apoyo aunque el modelo no esté disponible. La entrada debe haber sido admitida por el límite de cuerpo HTTP y debe tener texto válido.   
La seguridad se comprueba antes del límite de conversación de 1500 caracteres, de modo que un texto algo más largo con una señal explícita puede recibir apoyo. Sin embargo, un cuerpo mayor que el límite HTTP se rechaza con 413 antes de poder analizarse. No debe afirmarse que se escanean mensajes arbitrariamente grandes.   
### 17.2 Continuidad y su límite real   
El cliente mantiene `safety_context=true` tras una alerta; Nueva conversación lo reinicia. Este indicador conserva precaución, no memoria semántica.
Esto evita regresar inmediatamente a una etiqueta emocional por un mensaje corto siguiente. Sigue siendo estado aportado por el cliente, no una sesión clínica validada por el servidor. Un cliente independiente que omita o cambie la bandera pierde esa continuidad. La aplicación no contacta automáticamente servicios ni monitoriza seguridad fuera de la pestaña.   
## 18. Servidor y API, con el detalle necesario para defenderlos   
Flask carga el modelo y sirve la aplicación en la dirección local impresa al arrancar.

`/api/chat` clasifica mensajes; `/api/resultados` muestra evidencia guardada; `/api/health` informa disponibilidad. Flask sirve la página y los documentos permitidos.

Sin persistencia: cada mensaje se clasifica por separado. La precaución tiene prioridad incluso si falta el modelo.

## 19. Interfaz funcional, sin profundizar en diseño visual   
JavaScript envía mensajes a `/api/chat` y muestra sus resultados. El entrenamiento y la inferencia se realizan en Python.

## 20. Pruebas y qué evidencia aportan   
El test estadístico mide al modelo sobre la partición reservada. Los tests de pytest verifican el software con ejemplos de desarrollo, datos de juguete y archivos temporales. Ejecutar pytest no vuelve a inferir sobre la partición estadística de test.
Las 57 pruebas comprueban validación HTTP, precaución sin modelo, continuidad, cabeceras, vocabulario, métricas, recuperación del artefacto y reutilización del resultado final. Una prueba bloquea la carga del modelo y los datos al comprobar que una evaluación existente se devuelve sin repetirla.
`fixture` prepara objetos compartidos; `parametrize` multiplica casos; `monkeypatch` sustituye dependencias y `tmp_path` aporta carpetas temporales. La recuperación comprueba que se lea solo train; el rechazo de artefactos detecta contenido alterado antes de cargarlo.
Las comprobaciones de navegador y HTTP están registradas en `docs/VERIFICACION.md`. Las pruebas de software no demuestran seguridad clínica, y la emulación móvil no equivale a probar un teléfono físico.

## 21. Resultados que debes poder decir sin dudar   
El ganador fue logística con TF-IDF de unigramas y bigramas. Su macro-F1 de validación fue aproximadamente 0,787 y su macro-F1 de test aproximadamente 0,653. La accuracy de test fue 66,25%. La cobertura selectiva de test fue 75% y la accuracy entre aceptados aproximadamente 73,33%.   
La debilidad más marcada es alegría: solo 4 de sus 16 mensajes se clasificaron correctamente. Esa cifra se debe explicar aunque una frase de alegría de la demostración funcione bien. Mostrar ejemplos acertados no reemplaza una evaluación completa.   
El resultado inferior a validación puede reflejar variación muestral, selección favorecida por validación y asociaciones que no se transfieren a escenarios nuevos. Los datos no permiten identificar de forma concluyente cuánto aporta cada causa. No se corrigió el corpus para elevar el resultado final.   
Un resultado modesto puede sustentar un buen trabajo académico si se delimitan las conclusiones y se hace una evaluación correcta. No permite afirmar calidad suficiente para uso clínico ni rendimiento poblacional. El informe completo, con las tablas y todos los errores, está incorporado más adelante.   
## 22. Cómo ejecutarlo y reconstruirlo con criterio   
Para usar la entrega ya preparada:   
```
source .venv/bin/activate
python app.py

```
Abre la dirección exacta impresa. Deja esa terminal activa mientras lo usas. Ctrl+C solicita detener el servidor; el bloque finally cierra su socket. Si hay otras instancias, detener una no detiene automáticamente las demás.   
En otro equipo primero se crea el entorno y se instalan dependencias siguiendo `docs/REPRODUCIBILIDAD.md`. El proyecto se verificó con Python 3.14.2; el entorno y los artefactos son importantes para reproducir cifras y deserialización.   
Las etapas originales fueron preparar → comparar → evaluar → informe. La entrega existente reutiliza archivos congelados cuando repites esas etapas. `python -m entrenamiento todo` no es un botón para buscar un modelo nuevo: con los resultados presentes reutiliza la selección y test guardados y regenera los gráficos y el CSV de errores.   
Para estudiar `fit` sin tocar los archivos del experimento, este ejemplo ajusta un modelo en memoria usando únicamente train:

```python
from entrenamiento.p01_datos import cargar, xy
from entrenamiento.p02_modelos import crear_modelo

modelo = crear_modelo("logistica_bigramas").fit(*xy(cargar("train")))
print(modelo.predict(["Estoy feliz por aprender algo nuevo"]))
```

Ejecuta el ejemplo desde la raíz del proyecto con el entorno activo. Es una práctica de entrenamiento; no mide rendimiento. No hace falta borrar el artefacto ni la reserva de test. Si se desea mejorar el modelo usando errores ya vistos, se debe diseñar otro estudio y obtener un test nuevo; el anterior ya influyó en la investigación.   
## 23. Lenguaje Python que aparece repetidamente   
| Construcción | Uso en el proyecto |
| --- | --- |
| `def`, `return` | Define una función y devuelve su resultado. |
| Listas, diccionarios y conjuntos | Guardan secuencias, campos y elementos únicos. |
| Comprensiones, `enumerate`, `zip` | Transforman datos y recorren índices o listas alineadas. |
| `lambda`, `max`, `sorted` | Definen criterios para elegir u ordenar. |
| `with`, `try`, `except`, `finally` | Administran archivos y errores y liberan recursos. |
| `**opciones`, `*xy(filas)` | Desempaquetan argumentos con nombre o por posición. |
| `axis=1`, `[:, -1]` | Operan por fila y seleccionan la última columna. |
| `argsort`, `reshape`, `tolist` | Ordenan índices, adaptan dimensiones y convierten arreglos. |
| Máscaras y `&` | Seleccionan filas y combinan condiciones elemento a elemento. |
| Decoradores y `__name__` | Registran rutas o pruebas y controlan el arranque directo. |

## 24. Qué aprenderías al volver a construirlo   
Primero podrías implementar un clasificador de dos etiquetas con unas frases de juguete para entender X, y, fit y predict. Ese ejercicio sería didáctico, no evidencia del proyecto final. Después incorporarías las cinco categorías, el corpus fijo, el reparto por grupos y Pipeline.   
Una vez comprendido el ajuste, añadirías la referencia constante, NB y las variantes de representación, manteniendo una validación común. Definirías la regla de selección antes de mirar resultados y registrarías decisiones. Solo entonces congelarías y abrirías un test final.   
Después integrarías la función de inferencia con la API y respuestas redactadas. La precaución se colocaría antes del clasificador. Finalmente añadirías evidencia de errores, explicaciones, pruebas y documentación. Esa secuencia hace que cada decisión responda a un problema concreto y no a añadir complejidad por apariencia.   
## 25. Alcance de los anexos siguientes   
Los anexos conservan el informe completo y las preguntas de defensa. La explicación función por función está en el capítulo 26; los detalles adicionales se pueden consultar en los archivos originales del proyecto.

## 26. Lectura minuciosa de funciones del sistema   
Consulta cada archivo junto a sus comentarios breves. Aquí se resume lo que recibe, hace y devuelve cada función.

### entrenamiento/p00_configuracion.py

- `huella(ruta)`: Recibe una ruta, lee sus bytes y devuelve una cadena hexadecimal SHA-256. No modifica el archivo. Trabaja sobre bytes para detectar incluso cambios de formato que no alteran el significado de un CSV.
- `leer_json(ruta)`: Lee texto UTF-8 y lo convierte a listas, diccionarios, strings y números de Python. No valida por sí mismo un esquema ni garantiza que los campos esperados existan; los consumidores asumen archivos propios de la entrega.
- `guardar_json(ruta, contenido)`: Crea directorios padres si hace falta, escribe JSON indentado en un archivo temporal vecino y después lo reemplaza. ensure\_ascii=False conserva letras legibles. El reemplazo reduce la posibilidad de observar un archivo destino a medio escribir; no es una transacción conjunta para todos los archivos del experimento.

### entrenamiento/p01_datos.py

- `normalizar(texto)`: Transformación determinista compartida por representación y reglas. No recibe etiquetas y no ajusta estado. Su pérdida de información es deliberada y debe explicarse junto a sus beneficios de consistencia.
- `leer_escenarios(ruta=DATOS / 'escenarios.txt')`: Mantiene una etiqueta activa mientras recorre la fuente. Omite comentarios y líneas vacías. Counter lleva un número por clase; el formato :03d rellena con ceros. enumerate numera las dos paráfrasis y el grupo es compartido. Antes de devolver filas compara longitudes de lista y set de textos normalizados para detectar duplicados.
- `preparar()`: Puede crear archivos en la primera ejecución o verificar y reutilizar una partición existente. rng es local: no depende de que otro módulo haya consumido números aleatorios globales. int(.6*n) e int(.8*n) determinan fronteras; los conteos son exactos en este corpus, aunque con otros tamaños el redondeo podría alterar proporciones.
- `verificar_datos()`: Compara cada archivo declarado con la huella del manifiesto y se detiene al primer cambio. No corrige ni regenera datos modificados. Leer bytes de test para verificar integridad no se usa para extraer características, seleccionar ni medir al modelo.
- `cargar(particion)`: Solo admite tres nombres de partición y devuelve filas de csv.DictReader. Los valores se leen como strings. No interpreta automáticamente qué fase experimental tiene permiso para llamarla: esa disciplina está en los módulos que la invocan.
- `xy(filas)`: Proyecta cada fila en dos listas alineadas. La posición i del texto y la posición i de y deben corresponder. Excluye ids con nombres de clases para que no revelen la etiqueta.

### entrenamiento/p02_modelos.py

- `crear_modelo(nombre)`: Valida el nombre y construye objetos aún sin ajustar. Opciones compartidas mantienen normalización y tokenización consistentes. El orden de NOMBRES sirve además para desempates. No lee CSV ni llama a fit dentro del constructor.
- `metricas(reales, predichas)`: Recibe etiquetas reales y predichas alineadas. Devuelve datos simples, incluida matriz convertida a listas. Fija CLASES para que filas, columnas y tablas mantengan el mismo orden. Los resultados no incorporan la abstención: esta se informa aparte.
- `mascara_aceptacion(probabilidades, umbral, conocidos=None)`: Ordena scores por fila y usa las dos últimas columnas ordenadas como mayores valores. No altera el orden original de clases del modelo porque ordenadas es una estructura distinta. Devuelve un vector booleano; con conocidos combina las tres condiciones mediante &.
- `selectivas(reales, predichas, aceptar)`: Convierte listas a arrays para aplicar máscaras. La media de booleanos equivale a la fracción verdadera. aceptar.any() evita dividir entre cero cuando nadie es aceptado; por eso la accuracy indefinida se guarda como None, que en JSON aparece como null.

### entrenamiento/p03_comparacion.py

- `comparar()`: Reutiliza selección si existe. En una ejecución inicial prepara datos, ajusta cinco Pipelines y conserva cada objeto en modelos para elegir sin reentrenarlo. perf\_counter mide tiempo de ajuste. Una vez escogido, estima umbrales con scores de validación, calcula curva y serializa exactamente el objeto elegido. Las predicciones de validación se conservan como auditoría.
- `curva_aprendizaje(nombre, train, x_val, y_val)`: Construye una permutación de grupos por clase y toma prefijos de tamaños crecientes. El set de grupos elegidos sirve para filtrar frases completas. max(1, ...) evita un subconjunto vacío por clase. Cada punto crea su propio estimador; ninguno consulta test.

### entrenamiento/p04_evaluacion.py

- `intervalo_por_grupos(filas, predichas, repeticiones=1000)`: Crea un mapa grupo→índices para rescatar ambas frases de cada escenario. choice con replace=True puede seleccionar un grupo varias veces: sus frases cuentan varias veces en esa réplica. Mantiene una lista fija de clases incluso si una réplica omite alguna. El resultado es un diccionario de extremos, número de réplicas y método.
- `evaluar_una_vez()`: Su orden de operaciones es la protección principal: retorno si hay resultado, verificaciones, reserva exclusiva, carga y único lote de inferencia, derivación de métricas, guardado y cierre del registro. No es una base de datos con rollback; la reserva incompleta conserva evidencia de una interrupción.

### entrenamiento/p05_inferencia.py

- `cargar_modelo()`: Devuelve el diccionario esperado por predecir. El camino de recuperación ajusta un nuevo objeto en memoria con train sin evaluar test. El camino habitual verifica el archivo antes de joblib.load. No comprueba por sí solo todos los hashes de código en cada mensaje.
- `explicar(pipeline, vector, indice_clase, indice_alternativa)`: Inspecciona atributos del estimador para soportar logística y NB. Resta los pesos de candidata y alternativa, recorre solo características presentes y multiplica por el valor del vector. Ordena de mayor a menor y filtra positivos entre los primeros cinco. No incluye intercepto ni estima causalidad.
- `predecir(artefacto, texto)`: Transforma una lista con un solo texto para conservar formato de matriz. Llama directamente al clasificador del Pipeline sobre el vector ya obtenido, evitando vectorizarlo dos veces. argsort identifica primera y segunda clase. Calcula aceptación, construye nombres y valores Python y añade explicación. No actualiza vocabulario ni pesos.

### entrenamiento/p06_informe.py

- `graficar(seleccion, final)`: Construye barras de train y validación, imagen de la matriz final y línea de aprendizaje. Todas sus entradas son resultados existentes. El backend Agg permite dibujar sin una ventana gráfica. Colorear una celda no representa otra métrica: los números son conteos.
- `guardar_figura(figura, destino)`: Exporta PNG a resolución fijada y SVG vectorial, luego cierra la figura para liberar recursos. SVG puede incluir metadatos temporales distintos entre regeneraciones aunque los datos graficados sean idénticos.
- `generar_informe()`: Lee selección y resultados de test ya guardados, identifica los errores comparando etiqueta real y predicha y escribe el CSV. No cambia predicciones ni vuelve a ejecutar el test.

### entrenamiento/__main__.py

- `main()`: Hay dos funciones main en archivos distintos: en app.py gestiona el servidor y cierre; en `entrenamiento/__main__.py` analiza la etapa con argparse y llama a los módulos pertinentes. La ubicación decide su responsabilidad; compartir nombre no las convierte en la misma función.

### seguridad.py

- `detectar_riesgo(texto)`: Devuelve un booleano por coincidencia de reglas. La compilación previa evita reconstruir expresiones en cada mensaje. any termina cuando encuentra una coincidencia suficiente. Sus falsos negativos y activaciones por contexto deben considerarse límites de cobertura, no errores del clasificador emocional.

### conversacion.py

- `responder(etiqueta, turno)`: Recibe la etiqueta visible ya decidida y el turno validado por la API. El fallback protege ante una etiqueta no contemplada. La respuesta no depende de pesos de palabras ni genera tokens mediante un modelo.

### app.py

- `crear_app()`: Configura Flask, carga el modelo compartido y registra rutas; informa si falla la carga.
- `crear_app.preparar_peticion()`: Genera el permiso para los scripts y estilos propios.
- `crear_app.proteger_respuesta(respuesta)`: Añade cabeceras de protección a las respuestas.
- `crear_app.inicio()`: Sirve la página desde Flask, sin insertar mensajes del usuario en el HTML.
- `crear_app.salud()`: Comprueba presencia del artefacto en memoria, no ejecuta una batería de calidad ni un ejemplo de inferencia. Un estado ok significa disponible, no precisión garantizada.
- `crear_app.chat()`: Coordina el contrato de entrada y las ramas del flujo. La bandera de precaución se atiende antes del control del turno y del modelo. Normalmente devuelve reply, safety y analysis; los errores devuelven error y un estado HTTP específico.
- `crear_app.resultados()`: Entrega métricas guardadas, sin ejecutar el experimento ni enviar predicciones individuales.
- `crear_app.documento(nombre)`: Descarga únicamente documentos de una lista permitida.
- `crear_app.demasiado_grande(error)`: Rechaza cuerpos excesivos con el estado HTTP 413.
- `main()`: Hay dos funciones main en archivos distintos: en app.py gestiona el servidor y cierre; en `entrenamiento/__main__.py` analiza la etapa con argparse y llama a los módulos pertinentes. La ubicación decide su responsabilidad; compartir nombre no las convierte en la misma función.

## Documentos de consulta

- Curso y ampliaciones: `docs/MAPA_CURSO.md`.
- Protocolo previo: `entrenamiento/PROTOCOLO.md`.
- Fuente y ficha del corpus: `entrenamiento/datos/escenarios.txt` y `FICHA_DATOS.md`.
- Instalación y evidencia: `docs/REPRODUCIBILIDAD.md` y `docs/VERIFICACION.md`.
- Demostración y casos de precaución: `docs/DEMOSTRACION.md` y `tests/casos_seguridad.json`.

## Anexo D. Informe completo del experimento   
### Ánimo · Informe de resultados   
Este informe presenta resultados de archivos ejecutados y congelados; `entrenamiento/p06_informe.py` regenera sus gráficos y el CSV de errores. Fecha de evaluación (UTC): 2026-11-11T06:31:09.725664+00:00. No contiene cifras estimadas a mano. Las cifras redondeadas se presentan con tres decimales; los JSON conservan precisión completa.   
### 1. Pregunta y alcance   
¿Puede un clasificador sencillo distinguir cinco señales emocionales en mensajes breves en español, superando una referencia constante? Se implementó un chatbot académico local con respuestas redactadas, explicación de aportes de palabras, abstención y reglas de precaución independientes. La tarea es clasificar **texto**, no conocer el ánimo real ni identificar trastornos. «Ansiedad» es una etiqueta de preocupación textual.   
### 2. Datos y validez   
Corpus de **400 mensajes**, **200 escenarios**, dos paráfrasis por escenario. **El 100% es sintético y fue redactado con asistencia de IA para este proyecto**; no hay conversaciones reales, consentimiento de participantes ni etiquetas de especialistas. Etiquetas asignadas por intención del texto, sin doble anotación ni acuerdo entre anotadores. Ningún modelo preentrenado forma parte del sistema entregado. La asistencia de IA en la autoría del corpus y código se declara, y no equivale a entrenamiento con datos humanos.   

| Partición | alegria | ansiedad | enojo | neutral | tristeza | Total |
| :------------------ | :--------------- | :---------------- | :------------- | :--------------- | :---------------- | :------------- |
| train | 48 | 48 | 48 | 48 | 48 | 240 |
| validacion | 16 | 16 | 16 | 16 | 16 | 80 |
| test | 16 | 16 | 16 | 16 | 16 | 80 |

Semilla: 2026. La división se hace por escenario y clase. Se rechazan duplicados normalizados antes de dividir y se congelan hashes de los archivos. Los grupos resuelven la fuga de las paráfrasis declaradas; no prueban independencia semántica completa entre distintos escenarios. El mismo proceso de autoría produce todas las particiones: se comparten estilo y vocabulario, por lo que los resultados pueden ser optimistas. El autor conoce el corpus; no es un test externo ciego.   
### 3. Representación y selección   
Se conservan negaciones. Se normalizan mayúsculas, tildes y signos, con tokenización de palabras. TF-IDF es una ampliación práctica de la bolsa de palabras mencionada en clase. El vocabulario y el IDF se aprenden **solo en train**, dentro de Pipeline. No hay embeddings, descargas en ejecución ni APIs externas. El modelo servido se mantiene entrenado solo con train: no se reajusta después de la selección.   

| Candidato | F1 train | F1 validación | Accuracy validación | Variables | Ajuste (s) |
| :--------------------------- | :---------------- | :---------------------- | :---------------------------- | :----------------- | :------------------ |
| logistica\_tfidf | 1.000 | 0.773 | 0.775 | 587 | 0.0226 |
| logistica\_conteos | 1.000 | 0.745 | 0.750 | 587 | 0.0113 |
| logistica\_bigramas | 1.000 | 0.787 | 0.787 | 1651 | 0.0171 |
| naive\_bayes | 0.992 | 0.698 | 0.700 | 587 | 0.0027 |
| referencia | 0.067 | 0.067 | 0.200 | 587 | 0.0023 |

Ganador: **logistica\_bigramas**, por macro-F1 de validación sin redondear. Brecha F1 train-validación: **0.213**. Comparar conteos y TF-IDF observa el efecto de ponderar términos; comparar unigramas y bigramas observa el efecto de incorporar parejas de palabras. Son experimentos con una sola partición, no pruebas estadísticas concluyentes de superioridad. Los tiempos corresponden al equipo de ejecución, no son un benchmark general.   
### 4. Tamaño de entrenamiento   

| Fracción de train | Mensajes | Grupos | F1 validación |
| :-------------------------- | :---------------- | :-------------- | :---------------------- |
| 0.25 | 60 | 30 | 0.457 |
| 0.5 | 120 | 60 | 0.617 |
| 0.75 | 180 | 90 | 0.737 |
| 1.0 | 240 | 120 | 0.787 |

Se usan subconjuntos anidados por grupo y balanceados, con la misma validación. Es un análisis exploratorio; no se vuelve a elegir el modelo por esta curva. No demuestra una ley de escalamiento ni sustituye validación cruzada repetida.   
### 5. Abstención   

| Umbral | Cobertura validación | Accuracy aceptados |
| :-------------- | :----------------------------- | :-------------------------- |
| 0.3 | 0.725 | 0.845 |
| 0.4 | 0.588 | 0.957 |
| 0.5 | 0.412 | 0.939 |
| 0.6 | 0.175 | 1.000 |
| 0.7 | 0.037 | 1.000 |

Umbral elegido: **0.3**. Regla: Mayor cobertura entre umbrales con accuracy selectiva >= .80 y cobertura >= .50. Además, se exige margen >= 0.10 y al menos una característica conocida. Son heurísticas de producto, no calibración probabilística. Palabras conocidas en un mensaje ajeno al dominio aún pueden inducir respuestas equivocadas. No existe un detector universal de fuera de dominio.   
### 6. Test final, utilizado una sola vez   
El modelo, selección y código se congelaron antes de la reserva exclusiva del test. Se ejecutó una sola inferencia por lote sobre **80 mensajes**, guardando scores y predicciones. `test_reservado.json` registra la apertura. Repetir el comando lee el resultado existente. Los gráficos y el bootstrap reutilizan predicciones guardadas; no hay nueva selección ni ajuste tras el test.   
- **Accuracy total: 0.662**.   
- **Macro-F1 total: 0.653**.   
- Intervalo exploratorio percentil 95% de macro-F1 por escenario: **[0.543, 0.727]**, con **1000** remuestreos. Considera la dependencia de las dos paráfrasis de cada grupo. No incorpora incertidumbre de la selección del modelo ni sesgo sintético.   
- Con abstención: **cobertura 0.750**, **60 aceptados de 80** y **accuracy entre aceptados 0.733**. Esta accuracy no debe confundirse con la del test completo.   

| Clase | Precision | Recall | F1 | Soporte |
| :----------------- | :----------------- | :-------------- | :------------- | :--------------- |
| alegria | 0.333 | 0.250 | 0.286 | 16 |
| ansiedad | 0.706 | 0.750 | 0.727 | 16 |
| enojo | 0.700 | 0.875 | 0.778 | 16 |
| neutral | 0.923 | 0.750 | 0.828 | 16 |
| tristeza | 0.611 | 0.688 | 0.647 | 16 |

La debilidad más marcada es **alegria**: recall **0.250**, con **4 aciertos de 16**. La cifra global oculta esta diferencia. Además, **16 errores pasaron la abstención**; pedir contexto no garantiza que las etiquetas mostradas sean correctas. La meta selectiva de validación no se debe anunciar como rendimiento garantizado en test.   
Macro-F1 promedia las cinco F1 con igual peso. Accuracy cuenta aciertos/total. Precision por clase = TP/(TP+FP); recall = TP/(TP+FN); F1 es la media armónica. Una accuracy alta no basta para comprobar que todas las clases funcionen bien.   

| Real / predicha | alegria | ansiedad | enojo | neutral | tristeza |
| :----------------------- | :--------------- | :---------------- | :------------- | :--------------- | :---------------- |
| alegria | 4 | 3 | 4 | 1 | 4 |
| ansiedad | 1 | 12 | 2 | 0 | 1 |
| enojo | 1 | 0 | 14 | 0 | 1 |
| neutral | 3 | 0 | 0 | 12 | 1 |
| tristeza | 3 | 2 | 0 | 0 | 11 |

### 7. Errores observados, sin ocultarlos   
**27 errores** en el test completo. Se muestran todos, incluso los casos que el sistema rechaza por incertidumbre. `aceptada=True` significa que el chatbot habría mostrado la etiqueta errónea. Este análisis es posterior y no se utilizó para cambiar datos, parámetros ni modelo.   

| Texto sintético | Real | Predicha | Aceptada |
| :------------------------------------------------------ | :---------------- | :---------------- | :---------------- |
| Me siento agradecida por el apoyo que recibí | alegria | tristeza | True |
| Tengo mucha gratitud por quienes me acompañan | alegria | ansiedad | False |
| Me dieron la beca y no paro de sonreír | alegria | ansiedad | False |
| Qué bonito día, tengo ganas de celebrar | alegria | tristeza | False |
| Estoy de un humor excelente y quiero festejar | alegria | enojo | False |
| Me encanta salir a caminar por el parque | alegria | ansiedad | True |
| Qué orgullo haber vencido ese reto | alegria | enojo | True |
| Estoy satisfecho de haberlo conseguido | alegria | enojo | False |
| Pensé que saldría mal pero estoy feliz | alegria | enojo | False |
| Me siento alegre aunque esté cansado | alegria | tristeza | True |
| Fue agotador pero disfruté mucho el día | alegria | neutral | True |
| Mi amigo se recuperó y estoy feliz | alegria | tristeza | False |
| El rechazo de la entrevista me desanimó | tristeza | ansiedad | True |
| Me siento desilusionada con lo que pasó | tristeza | alegria | True |
| Fallar otra vez me quitó las ganas | tristeza | ansiedad | True |
| Me siento decepcionado de mí mismo | tristeza | alegria | False |
| Dejar mi barrio me dio mucha nostalgia | tristeza | alegria | True |
| Estoy molesta con mi compañero irresponsable | enojo | alegria | True |
| La pelea me dejó con mucha rabia | enojo | tristeza | False |
| Las cuentas pendientes me tienen ansioso | ansiedad | alegria | True |
| Me asusta quedarme sin dinero | ansiedad | enojo | True |
| Estoy ansioso mirando el correo cada rato | ansiedad | enojo | True |
| Me asusta la idea de quedarme solo | ansiedad | tristeza | True |
| Estoy revisando mi horario | neutral | alegria | True |
| Debemos sintetizar el contenido de la lectura | neutral | tristeza | False |
| Hoy almorcé arroz con pollo | neutral | alegria | True |
| Eso fue lo que comí al mediodía | neutral | alegria | False |

Interpretación: los modelos de palabras aprenden asociaciones léxicas, no el significado completo. Un contexto compartido (exámenes, familia, trabajo) puede aparecer en emociones distintas. Los bigramas solo capturan orden local; no resuelven automáticamente negación, ironía, causas ni emociones mezcladas. Los errores listados permiten formular hipótesis para un futuro corpus nuevo, no justifican corregir este test y volverlo a medir.   
### 8. Seguridad y privacidad   
Una capa de reglas se ejecuta antes de ML y prioriza apoyo humano ante expresiones de suicidio o autolesión. No muestra score ni diagnóstico de riesgo. Incluye Línea 113, opción 5 (Perú), acompañamiento de alguien de confianza y atención de emergencias si hay peligro inmediato o daño ya ocurrido. No llama servicios ni puede determinar si alguien está a salvo.   
Las reglas son conservadoras: también pueden activarse con negaciones, citas académicas, mensajes de terceros y frases ambiguas. Pueden fallar ante errores, eufemismos y lenguaje no contemplado. La batería de desarrollo en `tests/casos_seguridad.json` verifica comportamiento de software, **no sensibilidad ni especificidad clínica**. El recurso Línea 113 se consultó mediante la herramienta de líneas de ayuda; la opción 5 fue requisito explícito del proyecto. Referencia institucional: [MINSA](https://www.gob.pe/555?oai_link_source=model_response_hotline).   
Sin persistencia de conversaciones en servidor ni navegador. No se aprende en línea. Se exporta solo por decisión del usuario. Cada mensaje se analiza por separado; el servidor funciona localmente y sus logs no incluyen cuerpos de mensajes.
### 9. Limitaciones y trabajo siguiente   
La evaluación vale para este corpus pequeño, balanceado y sintético. No estima rendimiento en Perú, otros dialectos, grupos etarios, diagnósticos o conversaciones reales. No hay auditoría demográfica porque no existen esos atributos: no se afirma equidad. La distribución real probablemente será distinta; los scores no están calibrados. La normalización borra matices, signos y emojis y confunde ñ/n. El idioma esperado es español; no hay detector fiable de idioma. No se ha medido calidad terapéutica, satisfacción ni efecto sobre bienestar.   
Una siguiente fase responsable requiere consentimiento, minimización de datos, dos anotadores independientes, revisión profesional de respuestas y evaluación externa con hablantes diversos. Debe crearse **otro test** antes de ajustar con los errores aquí observados. No reutilizar esta prueba como evidencia independiente. Una futura comparación podría incorporar validación cruzada por grupos, calibración y evaluación de fuera de dominio, solo si su beneficio compensa la complejidad.   
### 10. Relación con los apuntes y reproducibilidad   
Semanas 4 (flujo del proyecto), 7 y 9 (reglas y NLP), 11 (datasets, sesgo, logística y métricas), 12 (logística multinomial, validación/test y matriz de confusión). Se amplían TF-IDF, Pipeline, agrupación de paráfrasis, abstención y bootstrap, explicados en este manual y en `docs/MAPA_CURSO.md`. No se usa la lematización preentrenada del laboratorio de semana 9 por la restricción del proyecto.   
Entorno de ejecución: Python 3.14.2, scikit-learn 1.7.2. Dependencias completas en `requirements-lock.txt`. Instrucciones experimentales en `docs/REPRODUCIBILIDAD.md`. Evidencia numérica: `seleccion.json`, `test_final.json`, `errores_test.csv`, figuras y hashes. Los números de este informe se reproducen a partir de esos archivos; volver a presentar el informe no constituye un nuevo test.   
Fuentes técnicas: [extracción de texto de scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html), [prevención de fuga de datos](https://scikit-learn.org/stable/common_pitfalls.html), [Flask](https://flask.palletsprojects.com/en/stable/quickstart/). Los apuntes locales son la base académica principal.   
## Anexo E. Guía de defensa y preguntas   
### Guía para explicar y defender Ánimo   
### La idea en un minuto   
Ánimo recibe un mensaje en español, comprueba reglas de precaución y, si no se activan, convierte palabras en números para clasificar una señal emocional. El clasificador aprende de ejemplos sintéticos etiquetados. La elección del modelo se hace con validación, y su resultado final se mide una sola vez con test. Una puntuación débil produce una petición de contexto. La respuesta conversacional se elige de textos redactados, no de un generador de lenguaje. Todo funciona localmente.   
No digas «detecta con certeza cómo se siente una persona», «es un psicólogo», «aprende de cada conversación», «los datos son reales», «está validado clínicamente» ni «conserva una precisión del 80%». Ninguna de esas afirmaciones describe este proyecto.   
### Preguntas probables y respuestas sugeridas   
**¿Por qué regresión logística si clasificas emociones?** A pesar del nombre, la logística sirve para clasificación. Aquí es multinomial porque hay cinco clases. Es la continuación directa de las semanas 11 y 12 y permite inspeccionar pesos.   
**¿El chatbot realmente aprende? ¿Qué se aprende?** El entrenamiento ajusta vocabulario/IDF y pesos del clasificador a textos etiquetados. En ejecución los parámetros permanecen fijos. Las respuestas de conversación y reglas de seguridad son programación explícita, no aprendidas.   
**¿Por qué no una red neuronal o un modelo de lenguaje?** No eran necesarios para demostrar la metodología con un corpus pequeño y habría sido más difícil defenderlos. Además, se exigió no usar modelos preentrenados ni APIs. La profundidad está en experimentos y análisis.   
**¿De dónde vienen los datos?** Todos son sintéticos, redactados con asistencia de IA. Las etiquetas reflejan la intención de redacción, no una emoción medida. No hubo participantes humanos ni expertos. Esto limita la validez externa y está declarado.   
**¿Usar IA para redactar datos incumple no usar modelos preentrenados?** Ningún modelo preentrenado se carga, consulta ni utiliza como extractor en el sistema o entrenamiento ejecutado. La autoría asistida del corpus se declara expresamente. Si la norma del curso además prohíbe asistencia en la redacción, habría que sustituir el corpus por uno autorizado y hacer un estudio nuevo; no ocultarlo.   
**¿Cómo evitas fuga de datos?** Se dividen escenarios antes de vectorizar. Las dos paráfrasis viajan juntas. Cada Pipeline hace fit solo con train. Selección y umbral miran validación, test se abre al final. Los ids, grupos y etiquetas no forman parte de X. Esto no elimina semejanza de estilo entre escenarios sintéticos.   
**¿Por qué tres conjuntos?** Train ajusta parámetros, validación permite escoger, test mide al seleccionado una vez. Elegir el mejor resultado de test usaría indirectamente sus etiquetas y daría una cifra optimista.   
**¿Cómo sé que el test se usó una vez?** Hay protocolo, hashes previos y un archivo de reserva creado con modo exclusivo antes de abrirlo. Las predicciones están guardadas. Una segunda ejecución devuelve el JSON existente. Es protección contra errores accidentales, no una prueba criptográfica contra manipulación intencional.   
**¿El bootstrap usa test mil veces?** Remuestrea las predicciones ya calculadas por escenario para describir variabilidad. No vuelve a llamar al modelo, ajustar hiperparámetros ni seleccionar. Es un intervalo exploratorio condicionado al modelo elegido y al corpus.   
**¿Por qué el test salió peor que validación?** La selección favorece al modelo con mejor resultado en una muestra pequeña; otra muestra tiene escenarios distintos. Esto muestra incertidumbre y posible sobreajuste al estilo de desarrollo. Se reporta la caída, sin corregir test ni cambiar semilla.   
**¿Son suficientes los resultados para usarlo con personas?** No. Los resultados muestran un prototipo funcional y un procedimiento académico, no eficacia real. La calidad limitada del test y los datos sintéticos impiden recomendarlo como herramienta clínica o de decisiones sobre personas.   
**¿Por qué no consigues un 95%?** No se fijó una cifra deseada para ajustar el experimento. El resultado es el obtenido. Buscar un número alto reusando test sería invalidar la evaluación. Mejorar exige datos más representativos y otro test.   
**¿Qué pasa con sarcasmo o «no estoy feliz»?** Se conserva «no» y hay bigramas, pero eso no garantiza comprensión. Las palabras dominantes pueden conducir a errores. Se declara como limitación y no se ajusta usando el test final.   
**¿Qué ocurre con una palabra nueva o emojis?** El vectorizador ignora términos fuera del vocabulario y nuestra normalización elimina emojis. Si no quedan características, se pide contexto. Si quedan palabras conocidas puede haber una predicción errónea: no es un detector general de idioma o fuera de dominio.   
**¿Un score del 60% significa que estoy 60% triste?** No. Es una salida relativa del modelo para una etiqueta en este espacio de clases. No está calibrada sobre población real y no mide intensidad ni probabilidad clínica.   
**¿Por qué la meta de 80% en validación no se cumple en test?** Era un criterio de selección de umbral en validación, no una garantía. Precisamente por eso se necesita test independiente y se informa el resultado selectivo real, junto a su cobertura.   
**¿Qué hace con riesgo de suicidio?** Primero aplica reglas de precaución. Devuelve apoyo, Línea 113 opción 5 de Perú, sugerencia de acompañamiento y emergencias si hay daño o peligro inmediato. No clasifica una emoción, estima riesgo ni contacta servicios. Puede equivocarse y no determina que alguien está a salvo.   
**¿Y si el mensaje dice «no quiero suicidarme» o habla de un artículo?** La regla es deliberadamente conservadora y activa apoyo contextual. Son falsos positivos de contexto posibles; se muestran en tests. La negación no es una garantía de seguridad y no se intenta hacer una evaluación clínica con regex.   
**¿Qué pasa si falla el modelo?** El endpoint sigue proporcionando la respuesta de precaución ante señales. Para mensajes normales responde 503 de forma explícita. No inventa un análisis ni una cifra.   
**¿Hay memoria de la conversación?** El navegador conserva mensajes y turno solo para la sesión. ML analiza cada mensaje de forma independiente. Una bandera conserva precaución después de una alerta; es un estado explícito de seguridad, no comprensión del contexto.   
**¿Cómo se protege la privacidad?** No hay persistencia, cuentas, cookies ni APIs externas. Los logs HTTP no incluyen el cuerpo de mensajes. Exportar crea un archivo local con contenido sensible que el usuario decide guardar. No se promete anonimato absoluto de todo el equipo.   
**¿Por qué Flask y un HTML?** Para ejecutar página y API en un proceso local. Se abre la dirección impresa por `python app.py`.
**¿Qué pruebas hay?** Se comprueba metodología con fixtures de juguete y manifiesto, y software con casos de seguridad, entradas inválidas, degradación, privacidad y servicio de HTML. La verificación manual usa un navegador real; sus resultados se registran por separado. Pasar tests no demuestra seguridad clínica.   
**¿Qué mejorarías a continuación?** Corpus autorizado con personas, doble anotación, escenarios mixtos, español peruano diverso, revisión profesional de seguridad y un nuevo test externo. Luego evaluar si validación cruzada por grupos y calibración aportan suficiente beneficio.   
**¿Qué hizo cada integrante y cuánto tardaron?** Esta entrega no contiene un registro de trabajo humano ni afirma dos meses de ejecución. Puede dividirse la exposición en datos, modelos, producto y evaluación según el guion. Debe declararse el uso de asistencia de IA de acuerdo con las reglas del curso.   
### Función por función
Cada función de los archivos Python se explica en el capítulo 26. La estructura del capítulo 5 y estas preguntas complementan esa lectura.

## 27. Preguntas de autoevaluación para comprobar comprensión   
Intenta explicar sin mirar: por qué se agrupan paráfrasis; qué aprende `fit`; por qué IDF usa solo train; cómo se eligen modelo y umbral; qué mide macro-F1; por qué hay errores aceptados; qué ocurre con palabras nuevas; y cómo harías otro experimento sin reutilizar test.
Si alguna respuesta falla, vuelve a los capítulos 6 a 15 y 32. Para el código, utiliza el capítulo 26 junto con los comentarios de cada archivo.

## 28. Vocabulario de repaso rápido

- Corpus: colección de textos. Dataset: datos organizados en filas y columnas.
- Paráfrasis: dos formas de expresar el mismo escenario. Grupo: unidad que las mantiene juntas.
- Característica: columna numérica de entrada. Token: palabra extraída del texto.
- Inferencia: uso del modelo ajustado. Generalización: desempeño con ejemplos nuevos.
- Fuga: información reservada utilizada al desarrollar el modelo. Sobreajuste: ajuste que se transfiere mal.
- Cobertura: fracción de etiquetas aceptadas. Calibración: correspondencia entre scores y frecuencias observadas.
- Artefacto: objeto aprendido guardado. Hash: huella para detectar cambios en bytes.
- Fixture: preparación compartida en pruebas. Endpoint: ruta que atiende una operación HTTP.

## 29. Una respuesta final que puedes sostener   
«El proyecto implementa clasificación supervisada de señales emocionales con NLP clásica. La representación y los clasificadores se ajustan exclusivamente con train; el candidato y el umbral se eligen con validación; el test se reserva hasta congelar el modelo. El chatbot añade respuestas redactadas, abstención, explicaciones contrastivas y reglas de precaución. La evaluación muestra limitaciones importantes, especialmente en alegría. Como los datos son completamente sintéticos, las cifras describen este experimento y no validan eficacia clínica o rendimiento con personas reales. Puedo justificar cada decisión, mostrar el código que la implementa y señalar qué evidencia la respalda.»   
Esta explicación acompaña el aprendizaje; entender el proyecto implica poder seguir un ejemplo nuevo y reconocer honestamente cuándo una pregunta exige un estudio que todavía no se ha realizado.   
## 32. Cómo entrenar una versión con más frases

Esta práctica crea **otro experimento**. No modifica la entrega actual ni convierte su test, que ya fue observado, en un test nuevo e independiente. Trabaja con la terminal situada en la carpeta del proyecto. En macOS puedes crear una copia así:

```sh
mkdir ../animo-experimento-nuevo && \
rsync -a --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' ./ ../animo-experimento-nuevo/ && \
cd ../animo-experimento-nuevo && \
python3 -m venv .venv && \
source .venv/bin/activate && \
python -m pip install -r requirements.txt
```

Si `mkdir` indica que la carpeta ya existe, usa otro nombre y adapta el comando y la comprobación siguientes; no mezcles experimentos. Abre esta copia en VS Code. La creación del entorno y la instalación solo son necesarias una vez.

### Paso 1: escribir escenarios nuevos

Edita `entrenamiento/datos/escenarios.txt` **en la copia**. Cada sección tiene una etiqueta entre corchetes. Cada línea es un escenario con **dos frases equivalentes separadas por `|`**. Por ejemplo, debajo de `[alegria]` podrías añadir:

```text
Estoy feliz por haber terminado mi curso | Me alegra mucho completar mis estudios
```

Añade ejemplos variados a las cinco clases, equilibrando su cantidad y revisando que las dos frases expresen la misma emoción. Evita duplicados incluso después de quitar mayúsculas, tildes y signos: `leer_escenarios()` los rechaza. Solo las palabras y parejas de palabras que queden en **train** podrán incorporarse al vocabulario; si aparecen únicamente en validación o test se ignorarán como características nuevas. Una palabra nueva se aprende al aparecer en ejemplos de entrenamiento; no se incorpora al vocabulario del modelo ya entrenado simplemente por añadirla al archivo. Más frases parecidas no garantizan que mejore con personas reales.

La implementación actual etiqueta la procedencia como `sintetico_asistido_IA`. Si usas textos de otro origen, adapta `p01_datos.py` y la ficha para declarar su procedencia correctamente. Usa solo datos que tengas derecho a utilizar; si son mensajes reales, protege la privacidad y considera el consentimiento.

### Paso 2: preparar la copia para un experimento nuevo

La entrega original tiene manifiesto, particiones, selección, modelo y reserva de test congelados. Por eso editar el archivo y ejecutar el entrenamiento directamente no los reemplaza. **Solo dentro de la copia nueva**, con la terminal en su carpeta, ejecuta:

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

Esto elimina únicamente los resultados y particiones **de esa copia**, para que se creen otros. Conserva intacta la carpeta original. El `INFORME_RESULTADOS.md` copiado describe el experimento anterior: no presentes ese informe como si fuera el resultado nuevo.

### Paso 3: dividir, entrenar y evaluar

Desde la raíz de la copia, ejecuta las etapas en este orden:

```sh
python -m entrenamiento preparar
python -m entrenamiento comparar
python -m entrenamiento evaluar
python -m entrenamiento informe
```

`preparar` crea las particiones por escenario; `comparar` ajusta los candidatos con train y selecciona usando validación; `evaluar` reserva el test y obtiene las predicciones una vez; `informe` genera gráficos y el CSV de errores a partir de los resultados guardados. Consulta `entrenamiento/resultados/seleccion.json` y `test_final.json` y redacta un informe nuevo a partir de esa evidencia. **No se regenerará automáticamente el texto de `INFORME_RESULTADOS.md`.**

Antes de ejecutar `evaluar`, congela las decisiones. Como las frases del experimento original y sus errores ya fueron vistos, una nueva división de esas mismas frases **no constituye un test externo ciego**. Para estimar mejor la generalización necesitas además un conjunto nuevo, separado por escenarios y no revisado durante el desarrollo. Comparar las cifras de experimentos con datos o particiones diferentes exige explicarlo; una cifra mayor por sí sola no demuestra una mejora real.

**Alcance de esta práctica:** el código reparte automáticamente todos los escenarios; añadir frases no garantiza que las nuevas queden en test. El resultado es otro experimento de desarrollo. Para una evaluación externa hay que reservar escenarios nuevos antes de inspeccionarlos y adaptar la preparación para mantenerlos exclusivamente en test, sin incluir ninguna paráfrasis en train o validación. Esa reserva externa no está implementada automáticamente en estos comandos.

### Paso 4: interpretar y mejorar

Conserva las cinco etiquetas actuales para esta práctica. Añadir una emoción exige ampliar `CLASES`, el corpus, las respuestas y la interfaz, y evaluar otra vez el sistema completo. Revisa macro-F1, métricas por clase, matriz, errores y cobertura de la abstención, no solo accuracy. Para una mejora futura considera frases más variadas y autorizadas, ejemplos ambiguos, anotación independiente, revisión de seguridad y una evaluación externa. No corrijas el test después de verlo para volver a presentar la nueva cifra como independiente.
