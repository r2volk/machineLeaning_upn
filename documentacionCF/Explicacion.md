# Explicación Completa del Proyecto y Guía de Defensa Oral

Este documento explica en lenguaje claro cómo se construyó el proyecto desde cero, qué temas del curso aplicamos, qué hace cada archivo y cómo responder las preguntas más probables en clase.

---

## 1. Relación con las semanas del curso

| Semana | Tema visto en clase | Cómo se aplica en el chatbot |
|---|---|---|
| **Sem. 02** | Sistemas autónomos y cuidado humano en situaciones críticas | Opera localmente en tu PC, pero ante riesgo de crisis bloquea el modelo y deriva a la **Línea 113 Salud del MINSA, Opción 5**. |
| **Sem. 03** | Agentes inteligentes (entrada, decisión, respuesta y memoria) | Entrada = mensaje del usuario; decisión = seguridad + modelo; memoria = recuerda 1 turno cuando pide contexto (`contexto_previo`); aprendizaje = pestaña *Aprendizaje*. |
| **Sem. 04** | Desarrollo en etapas | Definición del problema → carga y limpieza (`p02`–`p03`) → comparación de modelos (`p05`–`p08`) → evaluación (`p06`, `p10`) → consola y web (`chat_terminal.py`, `app.py`). |
| **Sem. 05** | Probabilidades que suman 100 % y semilla fija (`42`) | Entrega porcentajes sobre las 5 emociones que suman 100 %, mide duda para pedir contexto y fija `SEMILLA = 42`. |
| **Sem. 07** | Datos sintéticos y reglas fijas | Distingue texto real traducido (`go_emotions_es_mt`), sintéticos (`sintetico_claude`) y curados (`curado_situaciones_implicitas`); la capa de crisis usa reglas fijas. |
| **Sem. 09** | Reglas + ML y limpieza de texto (`unicodedata`, `re`) | Pasa a minúsculas cuidando la `ñ`, pero **no borra `"no"`, `"nunca"` ni `"sin"`**, sino que marca las 2 palabras siguientes con `neg_`. |
| **Sem. 11** | Mitigación de sesgos en 3 etapas, aprendizaje adaptable y F1 | Preprocesamiento (0 repetidos + `neg_`), entrenamiento (`class_weight="balanced"`, `C=3.0`) y postprocesamiento (pedir contexto). Además, pestaña *Aprendizaje* reversible. |
| **Sem. 12** | Clasificadores, división Train/Val/Test y Matriz de confusión | Compara 9 modelos en las mismas particiones, usando Regresión Logística Multinomial (Softmax) y matrices 5×5. |
| **Sem. 14** | Reducción de dimensionalidad y Red Neuronal MLP | Evalúa `TruncatedSVD (150)` + `MLPClassifier (64)` frente a la Regresión Logística completa. |

---

## 2. Ideas clave que amplían lo visto en clase (explicadas fácil)

### 2.1 TF-IDF de palabras `(1, 3)` + pedazos de letras `char_wb (3, 5)`
- **TF-IDF** da más peso a palabras específicas (`"vacío"`, `"robaron"`) y menos peso a palabras comodín (`"hoy"`, `"de"`).
- **Bigramas y trigramas** miran parejas y tríos seguidos (`"da mucha rabia"`): pasar de `(1, 2)` a `(1, 3)` mejoró todas las particiones (`+0.0032` GoEmotions, `+0.0106` sintética, `+0.0099` combinada).
- **Subpalabras (`char_wb` de 3 a 5 letras)** permiten conectar variaciones de la misma palabra (`robaron ↔ robo`) sin diccionarios externos, elevando GoEmotions F1 de `0.6102` a **`0.6654`**.

### 2.2 Macro-F1 y balanceo (`class_weight="balanced"`)
En entrenamiento el **55.1 %** es `neutral` y en validación real el **83.8 %**. Con `class_weight="balanced"` el modelo cuida por igual las emociones menos frecuentes, y con **Macro-F1** cada una de las 5 clases vale el 20 % del puntaje.

### 2.3 Pedir contexto cuando duda (abstención)
Si la emoción principal tiene menos de **`45 %`** o saca menos de **`10` puntos** sobre la segunda, el bot pide más contexto y recuerda 1 turno (`contexto_previo`). En GoEmotions esto sube los aciertos sobre mensajes respondidos de **`88.47 %` a `89.47 %`**.

### 2.4 Por qué reentrenamiento completo en *Aprendizaje*
Al enseñar un modismo nuevo (ej. `"saqué veinte"`), volver a entrenar todo (~6 s) permite agregar esa palabra nueva al vocabulario de TF-IDF sin olvidar las frases originales y manteniendo intacto `modelo_base.joblib`.

---

## 3. Recorrido archivo por archivo

- **`chat_terminal.py`**: demo en consola para clase (muestra primero la respuesta del chatbot, luego la emoción, barras y términos clave).
- **`app.py`**: servidor web Flask (`/`, `/api/health`, `/api/chat`, `/api/correcciones`, `/api/reentrenar`, `/api/restaurar`, `/api/laboratorio`, `/documentos/*`).
- **`templates/index.html`**: interfaz limpia con **Conversación**, **Laboratorio** (4 pasos, 4 KPIs, tabla de los 9 algoritmos, matriz de confusión real y resumen de contexto) y **Aprendizaje**.
- **`entrenamiento/p01_config.py`**: rutas, 5 clases, `SEMILLA = 42` y umbrales (`0.45`, `0.10`, `max_df=0.5`, `C=3.0`).
- **`entrenamiento/p02_datos.py`** y **`p02b_generar_corpus_implicito.py`**: cargan `26,469` ejemplos y verifican **0 fugas** entre entrenamiento y validaciones.
- **`entrenamiento/p03_preprocesamiento.py`**: `normalizar_texto()` y `preparar_texto_modelo()` con alcance `neg_` de 2 palabras.
- **`entrenamiento/p04_seguridad.py`**: filtro prioritario de riesgo (evita chistes como `"me muero de risa"`) y deriva a **Línea 113 Salud, Opción 5**.
- **`entrenamiento/p05_modelos.py`**: define los **9 modelos** (base Naive Bayes, 4 Regresiones Logísticas, Complement NB, LinearSVC, `TruncatedSVD + MLPClassifier` y Ensamble `VotingClassifier`).
- **`entrenamiento/p06_metricas.py`**, **`p07_inferencia.py`**, **`p08_experimentos.py`**: calculan métricas, ejecutan predicción + contexto de 1 turno, comparan los 9 modelos y guardan `modelo_base.joblib`.
- **`entrenamiento/p09_reentrenamiento.py`** y **`p10_evaluar_test.py`**: gestionan aprendizaje reversible y evaluación final única.
- **`pruebas/test_pipeline.py`** y **`test_app.py`**: 11 pruebas automatizadas con `unittest`.

---

## 4. Preguntas probables y respuestas claras para decir en voz alta

1. **«¿Por qué eligieron Regresión Logística y no Red Neuronal?»**
   > «Probamos `TruncatedSVD (150)` con una Red Neuronal `MLPClassifier` de la Semana 14 y sacó `0.5157` en GoEmotions, mientras Regresión Logística híbrida logró **`0.6654`** (`0.8275` de promedio). Comprimir `84,826` columnas a `150` borra palabras emocionales poco frecuentes. Además Regresión Logística da porcentajes directos y muestra exactamente qué palabras pesaron.»
2. **«¿Por qué no borraron palabras cortas como "no"?»**
   > «Porque en emociones `"no"` cambia todo el sentido. Las conservamos y marcamos hasta 2 palabras siguientes con `neg_` (por ejemplo `neg_feliz`), cortando en comas o palabras como `"solo"`.»
3. **«¿Cómo aseguraron que no hubiera trampa (fuga de datos)?»**
   > «El modelo solo ajusta palabras y pesos con entrenamiento; `verificar_sin_fuga()` confirma **0 textos y grupos repetidos** frente a validación, y Aprendizaje bloquea guardar frases de validación o riesgo.»
4. **«¿Por qué en validación sintética tienen 99 % y en GoEmotions 67 %?»**
   > «Porque las frases sintéticas son limpias y claras, mientras GoEmotions son mensajes cortos traducidos de internet donde el 83.8 % es neutral. Los mostramos por separado para ser transparentes.»
5. **«¿Cómo funciona la seguridad ante crisis?»**
   > «Siempre corre antes del modelo de ML: si detecta riesgo real (sin caer en frases figuradas como *"me muero de risa"*), bloquea la clasificación emocional y muestra ayuda real con la **Línea 113 del MINSA, Opción 5**.»
