# 06. Guion para tu exposición y preguntas frecuentes

Esta guía es corta a propósito para que la leas rápido antes de exponer y te salga natural.

---

## 1. Demostración en vivo paso a paso (5–7 minutos)

### Paso 1: Mostrar primero el programa en terminal (`chat_terminal.py`)
1. Dentro de la carpeta del proyecto activa el entorno y corre:
   ```bash
   source .venv/bin/activate
   python3 chat_terminal.py
   ```
2. Di en clase: *"Y acá tenemos nuestro chatbot funcionando directo en consola. Lee tu frase, te responde primero y abajo muestra la emoción ganadora, los porcentajes y las palabras clave."*
3. Prueba 2 frases rápidas:
   - `"no me siento feliz"` → sale **Tristeza** (`84.4 %`) y verás `neg_feliz`.
   - `"ayer me robaron el telefono"` → sale **Enojo** (`98.1 %`).
4. Escribe `salir` para cerrarlo y di: *"Y para que sea más visual para el usuario hicimos una página web que incluye lo mismo que este chat, más las pestañas Laboratorio y Aprendizaje."*

### Paso 2: Abrir la página web y probar **Conversación** (`app.py`)
1. Arranca la web: `python3 app.py` y abre `http://127.0.0.1:5000`.
2. Prueba en **Conversación** (señalando el panel derecho):
   - `"No puedo estar más feliz con esta noticia"` → **Alegría** (`~96 %`).
   - `"No me fue del todo bien en el examen"` → el bot **pide contexto** en lugar de adivinar. Responde `"reprobé el curso y me siento sin fuerzas"` → detecta **Tristeza · con contexto previo** (`95.6 %`).
   - `"Ya no quiero vivir"` → **no clasifica emoción**, bloquea el ML y muestra ayuda con la **Línea 113 del MINSA, Opción 5**.

### Paso 3: Mostrar **Laboratorio** (ahora corto y visual)
Abre **Laboratorio** y solo señala los 4 bloques de arriba abajo:
1. **4 pasos:** Seguridad → Limpieza (`neg_`) → Modelo ML → Contexto.
2. **4 tarjetas grandes:** `26,469` textos de entrenamiento (`0` repetidos en pruebas), `0.6654` F1 en texto real y `0.9895` en prueba sintética.
3. **Tabla de los 9 modelos:** señala cómo sube de `0.3413` (Conteo simple) hasta **`0.6654`** (**Ganador**), superando también a **Red Neuronal MLP con TruncatedSVD** (`0.5157`), SVM (`0.6015`) y Ensamble (`0.6388`).
4. **Matriz de confusión + "¿Qué pasa cuando duda?":** explica la diagonal verde y que al pedir contexto sube la exactitud real de `88.5 %` a **`89.5 %`**.

### Paso 4: Mostrar **Aprendizaje** en 1 minuto
1. Escribe en el chat: `"No esperaba aprobar pero saqué veinte"` → el modelo base dice **Neutral**.
2. Ve a **Aprendizaje**, pulsa *"Usar mi último mensaje del chat"*, elige **Alegría**, pulsa **Guardar ejemplo** y **Entrenar nueva versión**.
3. Al volver al chat con esa misma frase ahora responde **Alegría**. Luego pulsa **Restaurar modelo base** para mostrar que es reversible.

---

## 2. Respuestas fáciles a preguntas de la profesora

### 1. «¿Qué algoritmos compararon y por qué ganó ese?»
> «Entrenamos 9 modelos con los mismos `26,469` textos. Empezamos desde un conteo simple con Naive Bayes (`0.3413`) y fuimos probando variantes hasta llegar a **TF-IDF de palabras y subpalabras con Regresión Logística**, que sacó el mejor puntaje F1 real (`0.6654` y `0.8275` de promedio). Ganó porque entrega porcentajes directos para saber cuándo pedir contexto y nos deja ver exactamente qué palabras pesaron.»

### 2. «¿Usaron reducción de dimensionalidad y red neuronal?»
> «Sí. Como nuestro modelo genera `84,826` columnas de texto, usamos **`TruncatedSVD`** (que es el equivalente de PCA para texto con muchos ceros) para comprimirlas a `150` columnas y entrenar una **Red Neuronal MLP** de 64 neuronas. En la prueba real bajó a `0.5157` porque comprimir tanto borra palabras poco frecuentes como *"robaron"* o *"vacío"*. Así demostramos con números por qué conviene Regresión Logística.»

### 3. «¿Qué librerías usaron?»
> «**scikit-learn** para el Machine Learning (TF-IDF, los 9 algoritmos, `TruncatedSVD` y métricas), **Flask** para la web local, **NumPy** y **SciPy** para cálculos y matrices de texto, **joblib** para guardar el modelo ya entrenado, y módulos estándar de Python (`csv`, `re`, `unicodedata`, `json`, `unittest`).»

### 4. «¿Cómo cargan el dataset y evitan hacer trampa?»
> «`p02_datos.py` lee los CSV fila por fila validando columnas y etiquetas, junta `train.csv` con `2,780` frases cotidianas (`26,469` en total) y revisa que haya **0 frases ni grupos repetidos** respecto a las dos particiones de validación. El modelo solo estudia con entrenamiento.»

### 5. «¿Por qué usan F1 y no solo porcentaje de aciertos (Exactitud)?»
> «Porque en texto real casi 84 de cada 100 frases son neutrales. Un modelo que responda siempre neutral sacaría 84 % sin detectar ninguna emoción. **Macro-F1** promedia las 5 emociones por igual para que todas cuenten lo mismo.»

### 6. «¿Por qué no borraron palabras cortas como "no"?»
> «Porque en emociones el `"no"` cambia todo el sentido. En vez de borrarlo, `p03_preprocesamiento.py` pega `neg_` a las 2 palabras siguientes (por ejemplo `neg_feliz`) y corta cuando aparece una coma o `"solo"`.»

### 7. «¿Qué limitaciones tiene todavía su proyecto?»
> «Al no usar modelos gigantes preentrenados ni internet, si aparece una frase cotidiana nunca vista con palabras nuevas, TF-IDF no la adivina. También le cuesta el sarcasmo y en texto real hay muy pocos ejemplos de ansiedad (`8` casos en validación GoEmotions).»
