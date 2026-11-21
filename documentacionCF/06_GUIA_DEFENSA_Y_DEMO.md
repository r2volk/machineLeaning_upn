# 06. Guion de demostración en vivo y preguntas frecuentes

Esta guía está pensada para que hagas una presentación segura, natural y sin leer un guion rígido.

---

## 1. Guion rápido para tu demostración en vivo (5–7 minutos)

### Paso 1: Arrancar y mostrar el objetivo (1 minuto)
- Ejecuta `python app.py` y abre `http://127.0.0.1:5000`.
- Di en palabras simples: *"Construimos un chatbot emocional 100 % local y entrenado desde cero en Python con Flask y scikit-learn. Clasifica señales de 5 emociones en español y siempre aclara que es un proyecto académico, no una herramienta clínica."*

### Paso 2: Probar frases normales, implícitas y negaciones (1.5 minutos)
Escribe 3 o 4 frases en vivo en la pestaña **Conversación** y señala el panel derecho (probabilidades + palabras clave):
1. **Emoción explícita con negación:** `"No puedo estar más feliz con esta noticia"` → **Alegría** (~95 %). Comenta que no quitamos la palabra `"no"` porque cambiaría el sentido.
2. **Situación cotidiana implícita:** `"Ayer me robaron el teléfono"` → **Enojo** (~96 %). Comenta que no dice la palabra *"enojado"*, pero el modelo lo detecta gracias al corpus curado y a las subpalabras.
3. **Pedir contexto + memoria de 1 turno:**
   - Escribe primero: `"No me fue del todo bien en el examen"` → el bot **se abstiene y pide contexto** en vez de adivinar.
   - Luego responde: `"reprobé el curso y me siento sin fuerzas"` → el bot combina tu mensaje anterior con la aclaración y marca **Tristeza · con contexto previo** (conecta directo con la Semana 3: secuencia de percepciones).
4. **Capa de seguridad (siempre prioritaria):** escribe `"Ya no quiero vivir"` → muestra que **no clasifica ninguna emoción**, detiene el modelo de ML y muestra ayuda humana con la **Línea 113 Salud del MINSA, Opción 5**.

### Paso 3: Mostrar el «Laboratorio» y comparar los 9 algoritmos (2 minutos)
Abre la pestaña **Laboratorio** y señala solo 3 cosas clave:
1. **Datos y 0 fugas:** 26,469 frases de entrenamiento (GoEmotions + sintético + 2,780 situaciones implícitas) y 0 frases repetidas en validación.
2. **Tabla de los 9 modelos:** señala cómo sube el Macro-F1 en GoEmotions desde `0.3413` (Naive Bayes simple) hasta **`0.6654`** (nuestro ganador: TF-IDF híbrido + Regresión Logística), superando también a SVM (`0.6015`), al Ensamble (`0.6388`) y a la Red Neuronal `MLPClassifier` con reducción de dimensionalidad `TruncatedSVD` (`0.5157`).
3. **Matrices de confusión:** explica por qué reportamos GoEmotions (`0.6654`) y validación sintética (`0.9915`) **por separado** (por honestidad, ya que el texto traducido real es mucho más difícil y 83.8 % neutral).

### Paso 4: Demostrar «Aprendizaje» en vivo (1.5 minutos)
1. En **Conversación**, escribe el modismo peruano: `"No esperaba aprobar pero saqué veinte"` → el modelo base lo marca como **Neutral** (porque `"saqué veinte"` no estaba en los datos).
2. Pasa a **Aprendizaje**, pulsa *"Usar mi último mensaje del chat"*, elige **Alegría**, pulsa **Guardar ejemplo** y luego **Entrenar nueva versión**.
3. Muestra que el entrenamiento tarda unos segundos, muestra la tabla de validación antes/después y al volver al chat con la misma frase ahora responde **Alegría**.
4. Finalmente pulsa **Restaurar modelo base** para demostrar que es 100 % reversible.

---

## 2. Preguntas probables de la profesora (con respuestas naturales)

### 1. «¿Qué algoritmos probaron y por qué eligieron ese?»
> «Probamos 9 modelos de 5 familias que vimos en clase: Naive Bayes, Regresión Logística, SVM Lineal, una Red Neuronal Perceptrón Multicapa y un Ensamble. Elegimos **TF-IDF híbrido (palabras + subpalabras) con Regresión Logística Multinomial balanceada** porque sacó el mejor Macro-F1 en GoEmotions (`0.6654`) y el mejor promedio global (`0.8285`), nos da probabilidades directas para pedir contexto cuando hay duda y nos permite ver exactamente qué palabras influyeron en cada respuesta.»

### 2. «¿Usaron alguna técnica de reducción de dimensionalidad?»
> «Sí. Como nuestro vectorizador de texto genera 84,700 columnas, para probar la **Red Neuronal (`MLPClassifier`)** de la Semana 14 aplicamos **`TruncatedSVD`** (que es la versión de PCA pensada para tablas de texto llenas de ceros) para comprimir esas 84,700 columnas a 150. Al evaluarla en validación vimos que bajó a `0.4938` en GoEmotions porque al comprimir tanto se pierden palabras poco frecuentes pero importantes para las emociones. Por eso demostramos con números que en nuestro caso conviene mantener las características completas con Regresión Logística.»

### 3. «¿Qué librerías usaron en el proyecto?»
> «Cinco librerías externas principales: **scikit-learn** para todo el Machine Learning (TF-IDF, los 9 algoritmos, `TruncatedSVD` y métricas), **Flask** para la página web local, **NumPy** para cálculos con vectores y probabilidades, **SciPy** como soporte de tablas dispersas y **joblib** para guardar y cargar el modelo entrenado sin reentrenar cada vez. Además usamos módulos propios de Python como `csv`, `re`, `unicodedata`, `json` y `unittest` para las pruebas.»

### 4. «¿Cómo cargan el dataset y cómo evitan hacer trampa (fuga de datos)?»
> «En `p02_datos.py` leemos los archivos CSV fila por fila con `csv.DictReader`, validamos que tengan las columnas `texto` y `etiqueta` y que la emoción sea válida. Unimos `train.csv` con nuestro corpus curado de situaciones implícitas (`26,469` filas en total) y ejecutamos `verificar_sin_fuga()`, que compara conjuntos de textos y grupos para asegurar que haya **0 coincidencias** con las particiones de validación. Además, TF-IDF y el modelo solo hacen `.fit()` sobre entrenamiento; sobre validación solo predicen.»

### 5. «¿Por qué usan Macro-F1 si su exactitud ya es 88 %?»
> «Porque en GoEmotions casi el 84 % de los mensajes son neutrales. Un modelo que dijera "neutral" a todo sacaría 84 % de exactitud sin detectar ninguna emoción. Macro-F1 promedia el F1 de las 5 clases por igual, así que obliga al modelo a hacerlo bien también en alegría, tristeza, enojo y ansiedad.»

### 6. «¿Por qué no borraron palabras cortas (stopwords) como hicimos en clase?»
> «Porque en análisis de emociones palabras cortas como *"no"*, *"ni"* o *"sin"* invierten todo el significado. Si quitáramos *"no"* de *"no estoy enojado"*, el modelo solo leería *"enojado"*. En vez de borrarlas, las conservamos y en `p03_preprocesamiento.py` marcamos hasta dos palabras después de una negación con el prefijo `neg_`.»

### 7. «¿Dónde todavía se equivoca su modelo y por qué?»
> «Somos transparentes con sus límites: como está entrenado 100 % desde cero con TF-IDF (sin modelos preentrenados grandes), si alguien cuenta una situación completamente nueva con palabras que nunca aparecieron en los datos, el modelo no tiene conocimiento del mundo para adivinarla. También le cuesta un poco el sarcasmo y algunas negaciones donde las subpalabras de una palabra muy fuerte como *"triste"* siguen pesando. Por eso incluimos la opción de pedir contexto y la pestaña de Aprendizaje para corregir frases locales.»
