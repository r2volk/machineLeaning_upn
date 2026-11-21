# 01. Explicación sencilla del proyecto (de cero y sin tecnicismos)

Esta guía está escrita para que entiendas **de verdad** cómo funciona tu proyecto, sin tener que memorizar palabras complicadas. Primero explicamos la idea en palabras comunes y luego ponemos el nombre técnico que usamos en clase.

---

## 1. ¿Qué hace este proyecto?

Es un **chatbot local** (funciona en tu propia computadora, sin internet ni APIs pagadas) que lee una frase en español y trata de detectar señales de **5 emociones**:

1. **Alegría**
2. **Tristeza**
3. **Enojo**
4. **Ansiedad / nerviosismo**
5. **Neutral** (frases informativas o sin emoción clara)

**Algo muy importante que siempre debes aclarar:**
Este sistema es un **proyecto académico de Machine Learning**. No es un psicólogo ni hace diagnósticos médicos. Si detecta palabras de posible riesgo de suicidio o autolesión, **no intenta adivinar emociones**: corta el proceso y muestra ayuda humana real (**Línea 113 Salud del MINSA en Perú, Opción 5**, gratuita y 24/7).

---

## 2. ¿Qué pasa desde que escribes un mensaje hasta que responde?

Imagina que escribes: **"Ayer me robaron el teléfono"**. El programa sigue **4 pasos muy simples**:

### Paso 1: Revisión de seguridad (reglas fijas)
Antes de pasar la frase al modelo de Machine Learning, el archivo `p04_seguridad.py` revisa si hay señales de peligro (por ejemplo, *"no quiero vivir"* o *"hacerme daño"*). También sabe ignorar frases cotidianas que no son peligro real, como *"me muero de risa"* o *"me muero de hambre"*.
- Como *"Ayer me robaron el teléfono"* no es una señal de autolesión, sigue al paso 2.

### Paso 2: Limpieza del texto y negaciones
La computadora no entiende mayúsculas ni tildes igual que nosotros. Para ella, `"Teléfono"` y `"telefono"` serían dos palabras distintas. Por eso `p03_preprocesamiento.py` hace lo siguiente:
- Pasa todo a minúsculas y quita tildes, **pero cuida de no perder la letra `ñ`** (porque *"año"* no es *"ano"*).
- Quita letras estiradas exageradas (`"nooooo"` → `"no"`).
- **No borra la palabra `"no"`** ni otras negaciones. Al contrario: cuando aparece `"no"`, `"nunca"` o `"tampoco"`, marca las 2 palabras principales siguientes con `neg_` (sin gastar turno en pronombres cortos como `"me"`, `"te"` o `"se"`). Así, `"No me siento feliz"` se vuelve `"no me neg_siento neg_feliz"` y `"No estoy enojado, solo cansado"` se vuelve `"no neg_estoy neg_enojado solo cansado"`. Así el modelo no se confunde solo porque vio la palabra `"feliz"` o `"enojado"`.

### Paso 3: Convertir palabras en números y calcular probabilidades
Un modelo matemático no puede leer letras; solo trabaja con números.
- Primero, convertimos el mensaje en una fila de números usando **TF-IDF** (en palabras simples: le da más importancia a las palabras específicas como `"robaron"` y menos importancia a palabras repetidas en todos lados como `"el"` o `"de"`).
- Además de mirar palabras sueltas y grupos de hasta 3 palabras (por ejemplo, `"me robaron"`), también mira **pedacitos de 3 a 5 letras** dentro de cada palabra (subpalabras). ¿Para qué? Para que si el modelo conoce `"robo"`, también reconozca `"robaron"` o `"robando"` porque comparten el pedacito `"rob"`.
- Luego, nuestro clasificador (**Regresión Logística Multinomial**) suma los puntos que aporta cada palabra o pedacito a las 5 emociones y los convierte en **porcentajes que suman 100 %** (eso se llama función **Softmax**).

### Paso 4: Responder o pedir más contexto
El modelo no adivina a ciegas cuando tiene dudas:
- Si la emoción ganadora tiene **al menos 45 %** de probabilidad **y** le saca **al menos 10 puntos de ventaja** a la segunda emoción, responde con esa emoción y muestra en pantalla qué palabras influyeron.
- Si tiene dudas (menos de 45 % o muy empatadas), **se abstiene y te pide más contexto**. Y lo mejor: **recuerda tu frase anterior durante 1 turno** para combinarla con tu aclaración cuando le respondas.

---

## 3. ¿Qué librerías de Python usamos y para qué sirve cada una?

Si la profesora pregunta *"¿qué librerías usaron?"*, puedes responder esto de memoria y con total naturalidad:

### Librerías externas (las que instalamos en `requirements.txt`)
| Librería | Qué hace en palabras sencillas | Dónde se usa |
|---|---|---|
| **scikit-learn** | Es la caja de herramientas principal de Machine Learning. De aquí salen TF-IDF, Regresión Logística, Naive Bayes, SVM, Red Neuronal (`MLPClassifier`), Reducción de Dimensionalidad (`TruncatedSVD`), Ensamble (`VotingClassifier`) y las métricas (F1, matriz de confusión). | `p05_modelos.py`, `p06_metricas.py`, `p08_experimentos.py` |
| **Flask** | Crea el servidor web local para que puedas abrir el chatbot en tu navegador (`http://127.0.0.1:5000`) y conectar los botones de la página con Python. | `app.py` |
| **NumPy** | Sirve para hacer cálculos rápidos con listas de números y ordenar las probabilidades de mayor a menor. | `p06_metricas.py`, `p07_inferencia.py` |
| **SciPy** | Trabaja por detrás junto con scikit-learn para manejar tablas gigantes llenas de ceros (matrices dispersas de texto) sin gastar mucha memoria RAM. | Soporte interno de `scikit-learn` |
| **joblib** | Guarda el modelo ya entrenado en un archivo (`.joblib`) para que no tengas que reentrenarlo cada vez que abres la página web. | `app.py`, `p08_experimentos.py`, `p09_reentrenamiento.py`, `p10_evaluar_test.py` |

### Módulos que ya vienen incluidos en Python (no se instalan aparte)
- **`csv`**: lee los archivos `.csv` del dataset fila por fila.
- **`re`** (expresiones regulares): limpia signos, números y detecta frases de riesgo en la capa de seguridad.
- **`unicodedata`**: quita tildes manteniendo la letra `ñ`.
- **`json`**: guarda y lee los resultados de experimentos y las correcciones que enseña el usuario.
- **`pathlib`**: maneja las rutas de carpetas de forma ordenada en Windows o Mac.
- **`unittest`**: corre nuestras 11 pruebas automáticas para verificar que nada se rompa.

---

## 4. ¿Qué algoritmos de Machine Learning usamos?

En el proyecto **entrenamos y comparamos 9 modelos distintos** (todos desde cero, sin usar modelos preentrenados de internet), que pertenecen a **5 familias de algoritmos vistas en clase**:

1. **Naive Bayes** (`MultinomialNB` y `ComplementNB`): cuenta frecuencias de palabras y usa probabilidad clásica. Fue nuestra línea base más simple.
2. **Regresión Logística Multinomial** (`LogisticRegression`): calcula una suma ponderada de palabras y entrega probabilidades directas con Softmax. **Fue el algoritmo ganador** al combinarlo con TF-IDF de palabras + subpalabras.
3. **Máquinas de Vectores de Soporte — SVM Lineal** (`LinearSVC`): busca la mejor frontera para separar clases, pero no entrega probabilidades directas sin un paso extra.
4. **Red Neuronal Perceptrón Multicapa** (`MLPClassifier`, Semana 14) junto con **Reducción de Dimensionalidad** (`TruncatedSVD`, Semana 14): comprime las miles de columnas de texto a 150 columnas y las pasa por una red neuronal de 64 neuronas ocultas.
5. **Método Ensamblado** (`VotingClassifier`, Semana 12): junta las probabilidades de Regresión Logística y Complement Naive Bayes para decidir entre los dos.

En el documento **[02_COMPARACION_ALGORITMOS.md](02_COMPARACION_ALGORITMOS.md)** tienes el detalle sencillo y los números exactos de cómo le fue a cada uno.

---

## 5. Mapa rápido de los archivos del proyecto

- **`app.py`**: enciende la web local y recibe los mensajes del navegador.
- **`templates/index.html`**: es toda la página web (las 3 pestañas: *Conversación*, *Laboratorio* y *Aprendizaje*).
- **`entrenamiento/p01_config.py`**: guarda todas las rutas, las 5 emociones, la semilla fija (`42`) y los umbrales.
- **`entrenamiento/p02_datos.py`**: carga los CSV y verifica que no haya frases repetidas entre entrenamiento y validación.
- **`entrenamiento/p02b_generar_corpus_implicito.py`**: crea de forma reproducible los 2,780 ejemplos de situaciones cotidianas implícitas.
- **`entrenamiento/p03_preprocesamiento.py`**: limpia el texto y marca las negaciones (`neg_`).
- **`entrenamiento/p04_seguridad.py`**: revisa frases de riesgo y responde con la Línea 113 Opción 5.
- **`entrenamiento/p05_modelos.py`**: define los 9 modelos que vamos a comparar y el modelo ganador.
- **`entrenamiento/p06_metricas.py`**: calcula Exactitud, Precisión, Recall, Macro-F1, matriz de confusión y métricas cuando el bot pide contexto.
- **`entrenamiento/p07_inferencia.py`**: analiza un mensaje en vivo (seguridad → contexto previo si hubo → probabilidades → respuesta).
- **`entrenamiento/p08_experimentos.py`**: entrena los 9 modelos, compara hiperparámetros y guarda el modelo ganador.
- **`entrenamiento/p09_reentrenamiento.py`**: guarda las correcciones del usuario, reentrena una nueva versión y permite volver al modelo base.
- **`entrenamiento/p10_evaluar_test.py`**: evalúa el modelo congelado **una sola vez** cuando llegue el archivo de prueba final.
