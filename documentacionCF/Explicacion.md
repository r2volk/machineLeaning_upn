# Explicación Completa del Proyecto y Guía de Defensa Oral

Este documento explica cómo se construyó el proyecto desde cero, qué temas de los apuntes del curso se aplicaron exactamente, qué hace cada archivo y cada función del código, cómo agregar nuevos datos sin contaminar la evaluación y cómo responder preguntas probables de la profesora en persona.

---

## 1. Relación directa con los apuntes del curso

Antes de diseñar el sistema se revisaron todos los archivos de apuntes en `/Users/ricardo/Desktop/ML respaldo/machineLearning`. Cada decisión del proyecto se apoya en temas concretos vistos en clase:

| Semana y sección de los apuntes | Tema visto en clase | Cómo se aplica en este proyecto |
|---|---|---|
| **Semana 02 — Sec. 1.1 y 2.5** | Sistemas autónomos virtuales y seguridad/fiabilidad con supervisión humana en situaciones críticas | El chatbot es un sistema autónomo virtual que opera localmente, pero incorpora una capa prioritaria de seguridad que deriva a ayuda humana (Línea 113 Opción 5) cuando detecta riesgo. |
| **Semana 03 — Sec. 1.8, 2.3 y 4.5** | Relación entre chatbot y agentes inteligentes (sensores, procesador, actuadores, secuencia de percepciones y aprendizaje supervisado) | Sensor = texto del usuario en el chat; procesador = capa de seguridad + pipeline TF-IDF híbrido y Regresión Logística; secuencia de percepciones = memoria de 1 turno (`contexto_previo`) cuando pide contexto; actuador = respuesta empática y probabilidades; aprendizaje = pestaña de correcciones supervisadas. |
| **Semana 04 — Sec. 2 y 3** | Gestión en 6 etapas de un proyecto de IA y ejemplo de Análisis de Sentimiento | El proyecto sigue exactamente las 6 etapas: problema, propuesta, preparación de datos (`p02`–`p03`), selección del algoritmo (`p05`–`p08`), entrenamiento (`p08`) e implementación web (`app.py`). |
| **Semana 05 — Sec. 7.2, 7.3, 7.5 y 7.8** | Incertidumbre probabilística ($\sum p_i = 1$, valores en $[0, 1]$) y semilla (`seed` / `random_state`) | El clasificador produce una distribución de probabilidad sobre las 5 emociones que suma 1, permite medir incertidumbre (abstención) y fija `SEMILLA = 42` para resultados reproducibles. |
| **Semana 07 — Sec. 2.1, 2.2 y 3.4** | Datos sintéticos en ML cuando faltan datos reales y sistemas basados en reglas | Se distinguen explícitamente los datos sintéticos (`sintetico_claude`, `sintetico_asistido_IA`) de los traducidos (`go_emotions_es_mt`), y la capa de crisis funciona como un módulo determinista basado en reglas. |
| **Semana 09 — Sec. 1.9, 2.1, 2.3, 2.4 y 4.1–4.7** | Reglas de producción (Top-Down) + ML (Bottom-Up), NLP clásico, análisis de sentimiento y laboratorio de preprocesamiento (`unicodedata`, `re`, normalización, tokenización, stopwords) | Se combina un enfoque híbrido: reglas Top-Down para seguridad + NLP clásico Bottom-Up para emociones. Se aplica `lower()`, limpieza de tildes con `unicodedata.normalize('NFD', ...)` y limpieza de signos con `re.sub`. **Matiz importante:** a diferencia de la Sec. 4.4, aquí **no** se eliminan stopwords como `no`, `ni`, `sin` o `muy`, porque invertirían el sentido de las negaciones. |
| **Semana 11 — Sec. 1.2 (c), 2.1, 4.3–4.9, 5.3, 5.4, 7 y 8** | Aprendizaje supervisado, adaptación al cambio con etiquetas del usuario (como el filtro de spam), Bolsa de Palabras + Regresión Logística, mitigación de sesgos en 3 etapas, laboratorio con `LogisticRegression` y métricas (`accuracy`, `precision`, `recall`, `F1`, `support`) | Se implementa el mismo ciclo adaptable del filtro de spam (guardar correcciones → reentrenar), se mitigan sesgos en las 3 etapas (preprocesamiento sin fugas, regularización L2 + `class_weight="balanced"`, postprocesamiento con umbral de abstención) y se usa `LogisticRegression` en `scikit-learn`. |
| **Semana 12 — Sec. 2.3, 4, 5.2, 8.1–8.6 y 9.1** | Algoritmos de clasificación (Regresión Logística, Naive Bayes, SVM, métodos ensamblados), partición **Training / Validation / Testing**, matriz de confusión, **Regresión Logística Multinomial (Softmax + Cross Entropy)** y flujo `instanciar → fit → predict → evaluar` | Se comparan 9 modelos (Naive Bayes, Regresión Logística Multinomial, SVM Lineal, Red Neuronal MLP y Ensamble `VotingClassifier`) respetando estrictamente Training / Validation / Testing; se usa `predict_proba` (Softmax) y matrices de confusión 5×5. |
| **Semana 14 — Sec. 3.5, 4 y 6.3** | Reducción de dimensionalidad y Red Neuronal Perceptrón Multicapa (`MLPClassifier`) | Se incorpora el candidato `TF-IDF híbrido + TruncatedSVD (150) + MLPClassifier` en `p05_modelos.py` para compararlo empíricamente en las mismas particiones frente a la Regresión Logística balanceada. |

---

## 2. Técnicas incorporadas que amplían lo visto en clase (con ejemplos)

### 2.1 TF-IDF con escala sublineal (`TfidfVectorizer(..., sublinear_tf=True)`)
En clase vimos la **Bolsa de Palabras** (Semana 11, Sec. 5.3), que cuenta cuántas veces aparece cada palabra. El problema es que palabras muy comunes como `"hoy"` o `"que"` aparecen en todas las emociones y pueden pesar demasiado.

**TF-IDF** multiplica dos factores:
- **TF (frecuencia en el mensaje):** qué tanto aparece el término en la frase (con `sublinear_tf=True` usa $1 + \log(\text{tf})$ para que repetir una palabra 5 veces no valga 5 veces más que 1 vez).
- **IDF (frecuencia inversa en los documentos):** penaliza palabras presentes en casi todos los mensajes y premia palabras específicas.

**Ejemplo:** en `"Hoy me siento vacío"`, la palabra `"hoy"` aparece en miles de mensajes de todas las clases (IDF bajo), mientras `"vacío"` aparece casi solo en `tristeza` (IDF alto). Así el modelo le da mucho más peso a `"vacío"`.

### 2.2 N-gramas de palabras: bigramas `(1, 2)` y trigramas `(1, 3)`
Con palabras sueltas (`unigramas`), las frases `"estoy bien"` y `"no estoy bien"` comparten las palabras `"estoy"` y `"bien"`. Al incluir **bigramas** (pares consecutivos) y **trigramas** (tríos consecutivos), `"no me fue del todo bien"` genera además:
- Bigramas: `"no me"`, `"me fue"`, `"fue del"`, `"del todo"`, `"todo bien"`.
- Trigramas: `"no me fue"`, `"me fue del"`, `"del todo bien"`.

De igual forma, en `"Me da mucha rabia"`, el trigrama `"da mucha rabia"` refuerza `enojo` más allá de las palabras aisladas. Como comprobamos con los demás parámetros fijos, pasar de `(1, 2)` a `(1, 3)` mejoró el macro-F1 en **GoEmotions (`+0.0093`)**, **validación sintética (`+0.0113`)** y **combinada (`+0.0104`)**.

### 2.3 Macro-F1 y balanceo de clases (`class_weight="balanced"`)
En `train.csv`, el 59.25 % de los mensajes es `neutral` y solo el 8.43 % es `ansiedad`. Si entrenamos sin balanceo, el modelo puede acertar mucho en `neutral` y casi ignorar las emociones minoritarias.
- **`class_weight="balanced"`** (mitigación durante el aprendizaje, Semana 11 Sec. 5.4) asigna mayor peso en la función de pérdida a los errores sobre clases con menos ejemplos, multiplicando cada clase por $\frac{N}{K \cdot N_c}$.
- **Macro-F1** promedia el F1 de las 5 clases por igual ($\frac{1}{5}\sum_{c=1}^{5} F1_c$), sin dejar que `neutral` oculte fallos en `ansiedad`, `enojo`, `tristeza` o `alegria`.

### 2.4 Abstención (pedir contexto cuando la señal es débil)
Corresponde a la etapa de **post-procesamiento** (Semana 11, Sec. 5.4) y manejo de incertidumbre (Semana 05, Sec. 7.2). Con 5 clases, una elección al azar tendría probabilidad `0.20`. Si la emoción ganadora tiene probabilidad menor a **`0.45`** o supera a la segunda por menos de **`0.10`**, el chatbot decide **no adivinar** y pide contexto.

**Ejemplo real del sistema:** ante `"No me fue del todo bien en el examen"`, el modelo asigna `45.0 %` a ansiedad y `19.4 %` a tristeza (bajo el umbral de `0.45`), por lo que responde pidiendo contexto en lugar de imponer una etiqueta rígida. En `validacion_go.csv`, esto eleva la exactitud sobre los mensajes etiquetados de `87.05 %` a `90.16 %`.

### 2.5 Por qué reentrenamiento completo controlado en vez de `partial_fit`
Para la pestaña **Aprendizaje**, se eligió **reentrenar el pipeline completo** combinando `train.csv` con `datos/correcciones_usuario.json` (repitiendo cada corrección 3 veces para que expresiones nuevas superen `min_df=2` sin alterar el equilibrio de las 23,689 frases originales):
- Si usáramos actualización incremental (`SGDClassifier.partial_fit`), un vocabulario TF-IDF fijo no podría incorporar palabras nuevas como `"veinte"` o `"saque veinte"`, y actualizar solo con 1–3 ejemplos puede provocar *olvido catastrófico* de los pesos anteriores.
- Con reentrenamiento completo, el proceso tarda ~3–4 segundos, es 100 % determinista (`SEMILLA = 42`), recalcula el vocabulario TF-IDF, muestra de inmediato la variación real en las dos validaciones y permite volver al `modelo_base.joblib` congelado en cualquier instante.

---

## 3. Recorrido del proyecto archivo por archivo y función por función

Todo el código de Machine Learning vive en `entrenamiento/` ordenado de `p01` a `p10` para leerlo de principio a fin, y `app.py` vive en la raíz.

### `entrenamiento/p01_config.py`
Centraliza rutas (`RAIZ_PROYECTO`, `RUTA_TRAIN`, `RUTA_CORPUS_IMPLICITO`, `RUTA_VAL_GO`, `RUTA_VAL_SINTETICA`, `RUTA_MODELO_BASE`, `RUTA_MODELO_ACTIVO`, `RUTA_CORRECCIONES`), el orden fijo de las 5 `CLASES`, sus nombres legibles (`NOMBRES_CLASES`), `SEMILLA = 42`, los umbrales de abstención (`UMBRAL_PROB_MINIMA = 0.45`, `UMBRAL_MARGEN_MINIMO = 0.10`), `PESO_REPETICION_CORRECCION = 3`, y los hiperparámetros seleccionados en validación (`MAX_DF_TFIDF = 0.5`, `C_REGRESION_LOGISTICA = 3.0`, `REJILLA_AJUSTE_LR`).

### `entrenamiento/p02_datos.py` y `entrenamiento/p02b_generar_corpus_implicito.py`
- `cargar_csv_etiquetado(ruta)`: lee un archivo CSV con `csv.DictReader`, valida que existan columnas `texto` y `etiqueta`, descarta vacíos y verifica que cada etiqueta pertenezca a las 5 clases válidas.
- `cargar_entrenamiento_completo(...)`: une `train.csv` (23,689) con `entrenamiento/corpus_situaciones_implicitas.csv` (2,780) formando **26,469** ejemplos de entrenamiento.
- `p02b_generar_corpus_implicito.py`: genera de forma determinista (`SEMILLA = 42`) las **2,780** frases balanceadas (556 por clase) de situaciones cotidianas implícitas y desesgo temporal (`ayer`, `hoy`, `mañana`), excluyendo automáticamente cualquier texto presente en `train.csv`, validaciones o diagnóstico.
- `resumir_particion(filas)`, `verificar_sin_fuga(...)` y `cargar_particiones(...)`: auditan conteos por origen/clase y comprueban **0 fugas** entre entrenamiento y validaciones.

### `entrenamiento/p03_preprocesamiento.py`
- `normalizar_texto(texto)`: convierte a minúsculas, protege la letra `ñ` con un marcador temporal, quita tildes usando `unicodedata.normalize("NFD", ...)`, devuelve la `ñ`, reduce alargamientos de 3 o más letras repetidas (`"nooooo" → "no"`), sustituye signos y números por espacios y colapsa espacios múltiples. Se usa intacta para `p04_seguridad.py` y auditoría de fugas.
- `preparar_texto_modelo(texto, ventana=2)` y `preparar_lote_modelo(textos)`: aplica limpieza por cláusulas y marca hasta 2 palabras principales después de una negación (`no`, `nunca`, `jamas`, `tampoco`, `ni`) con el prefijo `neg_` sin gastar la ventana en pronombres cortos (`me`, `te`, `se`, etc.), cortando en signos de puntuación o conjunciones como `pero`, `solo`, `aunque`.

### `entrenamiento/p04_seguridad.py`
- `evaluar_riesgo(texto)`: normaliza el mensaje y primero verifica expresiones coloquiales inofensivas (`PATRONES_FALSO_POSITIVO`, como `"me muero de risa"`, `"me muero de sueño"`, `"matar el tiempo"`) para evitar falsos positivos; luego revisa `PATRONES_RIESGO` (ideación suicida o autolesión) y retorna `{"es_riesgo": bool, "motivo": ...}`.
- `construir_respuesta_crisis()`: construye el mensaje empático con orientación hacia apoyo humano, la información oficial verificada de la **Línea 113 Salud del MINSA (Perú), Opción 5 (Salud mental y psicología)**, los canales alternativos de WhatsApp/Telegram/correo del MINSA, teléfonos de emergencia (`106` SAMU / `105` Policía) y la aclaración explícita de que es un proyecto académico y no una herramienta clínica.

### `entrenamiento/p05_modelos.py`
- `construir_vectorizador_hibrido(max_df)`: usa `FeatureUnion` para combinar `TfidfVectorizer` de palabras `(1, 3)` (`min_df=2`, `max_df=0.5`, `sublinear_tf=True`, `token_pattern=PATRON_TOKEN_MODELO`) con `TfidfVectorizer` de subpalabras `char_wb (3, 5)` (`min_df=3`, `sublinear_tf=True`).
- `construir_modelo_ganador(c_reg, max_df)`: une el vectorizador híbrido con `LogisticRegression(C=3.0, class_weight="balanced", max_iter=1000, random_state=42)`.
- `construir_catalogo_modelos()`: devuelve los **9 modelos** comparados (referencia BoW + MultinomialNB, LR unigramas/bigramas/trigramas, ganador híbrido LR, ComplementNB, LinearSVC, Red Neuronal `TruncatedSVD + MLPClassifier` de Semana 14 y Ensamble suave `VotingClassifier` de Semana 12).

### `entrenamiento/p06_metricas.py`
- `redondear(valor)`: redondea valores flotantes a 4 decimales para consistencia en tablas y JSON.
- `evaluar_predicciones(y_real, y_pred, etiquetas)`: calcula exactitud, macro precisión, macro recall, macro-F1, métricas por clase con soporte y matriz de confusión 5×5.
- `evaluar_abstencion(modelo, textos_limpios, y_real, umbral_prob, umbral_margen)`: simula la regla de pedir contexto sobre una partición y calcula cobertura (`etiquetados / total`), abstenidos, aciertos sobre etiquetados, exactitud selectiva y macro-F1 selectivo.
- `evaluar_modelo_en_particiones(modelo, x_go, y_go, x_sin, y_sin)`: ejecuta las métricas en validación GoEmotions, validación sintética, combinada, promedio macro-F1 y abstención (si el modelo tiene `predict_proba`).

### `entrenamiento/p07_inferencia.py`
- `extraer_terminos_influyentes(modelo, texto_limpio, clase_objetivo, limite)`: multiplica cada valor TF-IDF activo del mensaje por su coeficiente en `clasificador.coef_` para la clase ganadora, priorizando palabras y mostrando subpalabras con prefijo `sub:`.
- `analizar_mensaje(modelo, texto_original, umbral_prob, umbral_margen, contexto_previo=None)`: flujo completo en tiempo real: (1) ejecuta `evaluar_riesgo()` sobre `texto_original` antes de tocar el modelo ML; (2) si viene `contexto_previo` seguro del turno anterior cuando el bot pidió contexto, combina ambos textos para clasificar (`uso_contexto = True`); (3) obtiene `predict_proba` y ordena las 5 emociones; (4) si `p1 < 0.45` o `p1 - p2 < 0.10` devuelve `estado = "pedir_contexto"`; (5) de lo contrario devuelve la emoción clasificada, probabilidades, términos influyentes y respuesta empática.

### `entrenamiento/p08_experimentos.py`
- `ejecutar_diagnostico_exploratorio(...)`, `construir_comparacion_ngramas(...)` y `evaluar_ajuste_hiperparametros(...)`: evalúan las 16 frases exploratorias sin solape con entrenamiento, comparan `(1, 2)` vs `(1, 3)` y ejecutan la rejilla de hiperparámetros `(max_df, C)` en validación.
- `ejecutar_experimentos()`: entrena los **9 modelos** solo con entrenamiento, guarda el ganador (`tfidf_hibrido_lr`) en `modelos/modelo_base.joblib` y `modelos/modelo_activo.joblib`, escribe `modelos/resultados_experimentos.json` e inicializa `modelos/estado_modelo.json`.
- `imprimir_resumen(reporte)`: imprime una tabla legible en consola al ejecutar el módulo.

### `entrenamiento/p09_reentrenamiento.py`
- `_textos_validacion_normalizados()`: obtiene el conjunto normalizado de ambas validaciones para impedir que un usuario guarde por error frases de validación como entrenamiento.
- `cargar_correcciones(ruta)` y `guardar_lista_correcciones(correcciones, ruta)`: lectura y escritura explícita del archivo local `datos/correcciones_usuario.json`.
- `agregar_correccion(texto, etiqueta)`: valida texto y emoción, **bloquea mensajes de riesgo con `evaluar_riesgo()`**, bloquea fugas con validación, evita duplicados y guarda el ejemplo con identificador y fecha UTC.
- `eliminar_correccion(corr_id)` y `limpiar_correcciones()`: permiten borrar una corrección individual o todas.
- `cargar_estado_modelo()`, `_metricas_base_referencia()` y `_calcular_delta(...)`: comparan el desempeño actual frente al modelo base.
- `reentrenar_con_correcciones(...)`: filtra nuevamente cualquier mensaje de riesgo, combina `train.csv` con las correcciones locales seguras, entrena desde cero el pipeline ganador, evalúa en validación sin tocar el modelo base, guarda `modelo_activo.joblib` y actualiza `estado_modelo.json`.
- `restaurar_modelo_base(...)`: copia `modelo_base.joblib` sobre `modelo_activo.joblib` y restablece el estado a `"base"`.

### `entrenamiento/p10_evaluar_test.py`
- `evaluar_conjunto_test(ruta_test, ruta_modelo, dir_salida, forzar)`: evalúa el modelo congelado sobre un archivo CSV externo de test, protege contra sobreescrituras accidentales (para respetar la evaluación de una sola vez), mide solape con `train.csv` y guarda `metricas_test.json` y `predicciones_test.csv`.
- `main()`: interfaz CLI (`--test`, `--modelo`, `--salida`, `--forzar`).

### `app.py`
- `cargar_modelo_disponible()`: carga `modelo_activo.joblib` (o `modelo_base.joblib` como respaldo).
- `crear_app()`: configura la app Flask y expone:
  - `GET /`: sirve `templates/index.html`.
  - `GET /api/health`: informa si el modelo está cargado en memoria.
  - `GET /documentos/<nombre>`: permite descargar de forma segura `INFORME_RESULTADOS.md` o `Explicacion.md` desde el Laboratorio.
  - `POST /api/chat`: clasifica o activa el protocolo de crisis.
  - `GET / POST / DELETE /api/correcciones`: administra ejemplos locales.
  - `POST /api/reentrenar`: entrena nueva versión y recarga el modelo en memoria.
  - `POST /api/restaurar`: vuelve al modelo base en disco y memoria.
  - `GET /api/laboratorio`: entrega métricas calculadas para las tablas del Laboratorio.
- `main()`: arranca el servidor en `127.0.0.1:5000` e imprime la URL exacta en la terminal.

### `templates/index.html`
Interfaz única («Ánimo») en HTML, CSS y JavaScript puro (superficies sólidas sin degradados, navegación inferior flotante y texto del chat de `18px` legible para proyección). Usa únicamente rutas relativas (`/api/health`, `/api/chat`, `/api/correcciones`, `/api/reentrenar`, `/api/restaurar`, `/api/laboratorio`, `/documentos/*`) y ofrece las vistas **Conversación**, **Laboratorio** y **Aprendizaje**.

### `pruebas/test_pipeline.py` y `pruebas/test_app.py`
Pruebas automatizadas con `unittest` que verifican conteos y cero fugas, normalización, seguridad y falsos positivos, abstención, comparación de trigramas, evaluación final en carpeta temporal, estilo de comentarios en todo el código Python, interfaz sin degradados y el ciclo completo de guardar → reentrenar → restaurar.

---

## 4. Cómo agregar más ejemplos sin mezclarlos con validación ni test

1. **Mantén las particiones congeladas:** nunca agregues ni edites filas dentro de `datos de entrenamiento/preparados/validacion_go.csv`, `datos de entrenamiento/preparados/validacion_sintetica.csv` ni del archivo de test futuro. Esos archivos son el «metro» fijo con el que medimos si el modelo realmente mejora.
2. **Usa la pestaña Aprendizaje o agrega filas solo a entrenamiento:** los ejemplos cotidianos nuevos deben ir a `datos/correcciones_usuario.json` (a través de la interfaz, que bloquea automáticamente textos que ya estén en validación y mensajes de riesgo) o a un nuevo archivo de entrenamiento auditable.
3. **Cuida los grupos de paráfrasis (`grupo`):** si escribes varias variantes de una misma situación (por ejemplo `"No aprobé el parcial"` y `"Jalé el examen parcial"`), dales el mismo identificador `grupo` y mantén **todo el grupo en entrenamiento**; nunca pongas una variante en train y otra en validación o test, porque eso inflaría artificialmente las métricas.
4. **Registra la procedencia (`origen`):** etiqueta claramente si cada nuevo ejemplo es `usuario_local`, `sintetico_...` o real para poder analizar si mejora en mensajes reales y no solo en frases parecidas a las tuyas.
5. **Usa el test final una sola vez:** no ajustes hiperparámetros ni agregues ejemplos mirando errores del test reservado; evalúa el modelo congelado una única vez con `p10_evaluar_test.py`.

---

## 5. Preguntas probables de la profesora y respuestas para decir en voz alta

### 1. «¿Por qué elegiste Regresión Logística y no una Red Neuronal o un LLM?»
> «Probamos una Red Neuronal Multicapa (`MLPClassifier` con `TruncatedSVD`, Semana 14) en las mismas particiones y obtuvo `0.4938` en GoEmotions frente a **`0.6654`** de nuestra Regresión Logística híbrida, porque en texto disperso de 84,700 características comprimir a 150 dimensiones pierde n-gramas poco frecuentes. Además, queríamos un modelo 100 % entrenado desde cero, con probabilidades nativas por Softmax, explicabilidad término por término y reentrenamiento en pocos segundos.»

### 2. «En el laboratorio de la semana 9 quitamos stopwords; ¿por qué aquí no las eliminaste?»
> «Porque en análisis de emociones las palabras cortas cambian por completo el sentido. Si quitáramos stopwords como *no*, *ni*, *sin* o *muy*, una frase como *"no me fue del todo bien"* quedaría reducida casi solo a *"bien"*. Además, en `p03_preprocesamiento.py` agregamos marcado de alcance de negación (`neg_` hasta 2 palabras tras `no/nunca/jamas/tampoco/ni`), lo que permitió corregir frases como *"No me preocupa en absoluto, todo salió genial"* y *"No estoy enojado, solo cansado"*.»

### 3. «¿Qué aportaron los trigramas y las subpalabras (`char_wb`)?»
> «Con los demás parámetros fijos, pasar de bigramas `(1, 2)` a trigramas `(1, 3)` mejoró las tres particiones (`+0.0020` en GoEmotions, `+0.0105` en sintética y `+0.0101` en combinada). Al unir en `FeatureUnion` las subpalabras de caracteres `char_wb (3, 5)` y ajustar `max_df=0.5` y `C=3.0`, el macro-F1 en GoEmotions saltó de `0.6121` a **`0.6654`** (`+0.0533`), porque las subpalabras conectan conjugaciones y raíces como `robaron ↔ robo` sin usar diccionarios externos.»

### 4. «¿Por qué no elegiste el SVM Lineal o el Ensamble `VotingClassifier`?»
> «En nuestro benchmark de 9 modelos, el ganador `tfidf_hibrido_lr` supera a ambos en GoEmotions (`0.6654` frente a `0.6049` de SVM trigramas y `0.6363` del Ensamble) y en Promedio Macro-F1 (`0.8285`). Además, `LinearSVC` no entrega probabilidades nativas (y al calibrarlo cae en GoEmotions) y el Ensamble pierde la explicabilidad directa de coeficientes por término.»

### 5. «¿Por qué en validación sintética tienes `0.99` y en GoEmotions `0.67`? ¿Tu modelo está sobreajustado?»
> «Esa diferencia muestra el cambio de dominio entre ambos conjuntos y por eso los reportamos por separado. Los datos sintéticos generados con Claude tienen redacción limpia y señales emocionales muy claras. En cambio, GoEmotions viene de comentarios reales de Reddit traducidos automáticamente al español: son más cortos, ambiguos, con modismos y un 83.8 % son neutrales. Reportar ambos números por separado evita dar una impresión engañosa.»

### 6. «¿Por qué `ansiedad` tiene F1 más bajo (`0.5000`) en GoEmotions?»
> «Por dos motivos documentados en los datos: primero, en GoEmotions no existe la etiqueta ansiedad sino *nerviosismo* (`nervousness`), que usamos como aproximación; segundo, en todo `validacion_go.csv` solo hay **8 ejemplos** reales de esa clase (y 84 en entrenamiento de GoEmotions). Con 8 casos, acertar 4 de 8 da `50.0 %` de recall y un solo caso cambia 12.5 puntos porcentuales.»

### 7. «¿Cómo aseguraste que no hubiera fuga de datos (*data leakage*)?»
> «Primero, `TfidfVectorizer.fit` y `LogisticRegression.fit` se ejecutan **únicamente** con `train.csv`; sobre validación solo se llama `transform` y `predict`. Segundo, en `p02_datos.py` programé `verificar_sin_fuga()`, que audita intersecciones de textos y de `grupo` entre train y las dos validaciones, dando 0 solapes. Tercero, al guardar correcciones del usuario, `p09_reentrenamiento.py` también bloquea cualquier frase que ya esté en validación.»

### 8. «¿Para qué sirve `class_weight="balanced"` y cómo se conecta con la semana 11?»
> «En la semana 11 vimos la mitigación de sesgos en tres etapas. Como el 59 % de nuestro entrenamiento es `neutral`, sin balanceo el modelo tiende a predecir casi todo como neutral. Con `class_weight="balanced"`, que corresponde a regulación durante el aprendizaje, penalizamos más los errores en clases minoritarias como `ansiedad` o `tristeza`. Eso subió nuestro macro-F1 en GoEmotions de `0.48`–`0.54` hasta `0.6101`.»

### 9. «¿Cómo funciona exactamente el mecanismo de pedir contexto?»
> «Es un paso de post-procesamiento sobre las probabilidades de Softmax (y en agentes inteligentes, Semana 3, aprovecha la secuencia de percepciones). Si la mayor probabilidad es menor a `0.45` o la diferencia entre la primera y la segunda es menor a `0.10`, el chatbot no fuerza una etiqueta y pide más contexto; además, recuerda ese mensaje durante 1 turno (`contexto_previo`) para combinarlo con la aclaración del usuario. En validación GoEmotions, abstenerse ante dudas eleva la exactitud selectiva de `88.47 %` a `89.68 %` y el macro-F1 selectivo a `0.6779`.»

### 10. «¿Dónde falla todavía tu modelo con frases reales?»
> «Con transparencia, existen dos límites claros: primero, al ser un modelo TF-IDF entrenado 100 % desde cero sin modelos preentrenados (BERT/LLMs), no tiene conocimiento del mundo para inferir situaciones implícitas cuyo vocabulario nunca apareció en entrenamiento; segundo, en negaciones directas de palabras muy fuertes como *"No estoy triste..."* las subpalabras de `triste` todavía pueden inclinar la predicción. Y para modismos locales como *"saqué veinte"*, el usuario puede enseñarlos de forma reversible en la pestaña Aprendizaje.»

### 11. «¿Cómo funciona la protección ante mensajes de suicidio o autolesión?»
> «En `p07_inferencia.py`, antes de normalizar para el modelo de Machine Learning, siempre se llama primero a `evaluar_riesgo()` en `p04_seguridad.py`. Es una capa basada en reglas (Semana 09, enfoque Top-Down) que detecta señales de suicidio o autolesión evitando falsos positivos cotidianos como *"me muero de risa"*. Si detecta riesgo, no clasifica emociones, responde con empatía, deriva a la **Línea 113 Salud del MINSA, Opción 5** (verificada oficialmente para salud mental en Perú, 24/7) y aclara que esto no es una herramienta clínica. Además, `p09_reentrenamiento.py` bloquea estos mensajes para que jamás se guarden ni se usen en entrenamiento.»

### 12. «¿Qué pasa si guardo varias correcciones y el modelo empeora en validación?»
> «Cada vez que pulso *Entrenar nueva versión*, el sistema entrena sobre `train.csv` más las correcciones seguras, mide inmediatamente el cambio frente al modelo base en ambas validaciones y guarda el resultado en `modelo_activo.joblib` sin tocar nunca `modelo_base.joblib`. Si el cambio en validación no conviene, basta con pulsar *Restaurar modelo base* para volver exactamente al estado inicial.»
