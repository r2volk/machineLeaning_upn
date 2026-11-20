# 03. El dataset y cómo lo cargamos paso a paso

Si la profesora pregunta *"¿cómo cargan el dataset?"* o *"¿de dónde salieron los datos?"*, aquí tienes toda la explicación en palabras simples y en el orden exacto en que trabaja el código.

---

## 1. ¿Cómo es nuestro dataset?

Nuestro dataset está guardado en archivos **CSV** (tablas de texto separadas por comas, como una hoja de Excel). Cada fila es un mensaje en español y tiene **4 columnas**:

| Columna | Qué guarda | Ejemplo |
|---|---|---|
| `texto` | La frase escrita en español | `"Hoy me siento vacío y sin ganas de salir"` |
| `etiqueta` | La emoción real (`alegria`, `ansiedad`, `enojo`, `neutral`, `tristeza`) | `"tristeza"` |
| `grupo` | Un identificador para agrupar frases parecidas y evitar que una quede en train y su copia en validación | `"imp_tristeza_0012"` |
| `origen` | De dónde vino esa frase (para no mezclar procedencias a ciegas) | `"go_emotions_es_mt"` |

---

## 2. ¿De dónde salen los datos y cuántos tenemos?

Dividimos los datos en **tres particiones completamente separadas** (como vimos en la Semana 12):

1. **Entrenamiento (`26,469` frases en total):** es lo **único** que ve el modelo para aprender. Se arma juntando dos archivos:
   - `datos de entrenamiento/preparados/train.csv` (**23,689** frases):
     - `15,381` de **GoEmotions traducido al español** (mensajes reales de Reddit, licencia Apache 2.0).
     - `8,068` **sintéticas** generadas con Claude.
     - `240` **sintéticas** del corpus inicial.
   - `entrenamiento/corpus_situaciones_implicitas.csv` (**2,780** frases curadas, **556 por cada emoción**): generadas de forma reproducible con `p02b_generar_corpus_implicito.py` para cubrir situaciones cotidianas donde no se dice la emoción explícitamente (por ejemplo, *"ayer me robaron el teléfono"* o *"ojalá mañana sea un mejor día"*) y para quitar sesgos de palabras comunes como *"ayer"* (que en versiones anteriores aparecía casi solo en alegría).

2. **Validación (2 archivos separados, `3,831` frases en total):** el modelo **nunca** entrena con estas frases; solo sirven para medir y comparar los 9 algoritmos:
   - `validacion_go.csv`: **1,899** frases reales traducidas de GoEmotions.
   - `validacion_sintetica.csv`: **1,932** frases sintéticas.

3. **Prueba final / Test (reservado):** se guarda aparte y se usa **una sola vez** al final con `p10_evaluar_test.py` cuando la profesora entregue el archivo de evaluación.

### Tabla resumen de conteos

| Partición | Alegría | Ansiedad | Enojo | Neutral | Tristeza | Total |
|---|---:|---:|---:|---:|---:|---:|
| **Entrenamiento (total)** | 3,008 | 2,552 | 3,333 | 14,592 | 2,984 | **26,469** |
| **Validación GoEmotions** | 106 | 8 | 109 | 1,592 | 84 | **1,899** |
| **Validación sintética** | 422 | 136 | 266 | 678 | 430 | **1,932** |

---

## 3. ¿Cómo cargamos el dataset en el código? (paso a paso)

Todo ocurre en el archivo **`entrenamiento/p02_datos.py`**, sin trucos ocultos:

1. **Leer cada CSV con `cargar_csv_etiquetado(ruta)`:**
   - Abre el archivo en UTF-8 (para leer bien tildes y la `ñ`) usando `csv.DictReader`, que lee cada fila como un diccionario usando los nombres de las columnas.
   - Revisa que existan las columnas obligatorias `texto` y `etiqueta`. Si falta alguna, lanza un error claro.
   - Recorre cada fila, quita espacios sobrantes, ignora filas vacías y comprueba que la etiqueta sea una de las 5 clases válidas (si hay una etiqueta mal escrita, detiene la carga y avisa en qué línea está el error).
2. **Unir el entrenamiento con `cargar_entrenamiento_completo()`:**
   - Primero carga `train.csv` (23,689 filas) y luego le suma `corpus_situaciones_implicitas.csv` (2,780 filas), llegando a 26,469 filas de entrenamiento.
3. **Auditar que no haya fugas de datos con `verificar_sin_fuga()`:**
   - Convierte los textos y los `grupo` de entrenamiento y de ambas validaciones en conjuntos (`set`).
   - Revisa la intersección entre ellos. En todas las pruebas el resultado es **0 solapes** (ninguna frase ni ningún grupo de validación se coló en entrenamiento).
4. **Limpiar el texto antes de entrenar (`entrenamiento/p03_preprocesamiento.py`):**
   - `normalizar_texto()` pasa a minúsculas, quita tildes cuidando la `ñ`, acorta letras repetidas (`"hooolaaa"` → `"hola"`) y limpia signos/números.
   - `preparar_lote_modelo()` aplica esa limpieza a toda la lista y marca con `neg_` hasta 2 palabras después de una negación (`no`, `nunca`, `jamas`, `tampoco`, `ni`), cortando en comas o conjunciones como `pero` o `solo`.
5. **Convertir a números solo con entrenamiento (`fit` vs. `transform`):**
   - En `p08_experimentos.py`, el vectorizador TF-IDF aprende su vocabulario **únicamente** con `x_train` (`pipeline.fit(x_train, y_train)`).
   - Sobre las validaciones solo se transforma y predice (`pipeline.predict(x_go)`), nunca se ajusta nada con ellas.

---

## 4. Dos detalles honestos sobre nuestros datos que conviene saber explicar

1. **Desbalanceo en `neutral`:** en entrenamiento, `neutral` es el **55.1 %** de las frases, y en GoEmotions de validación es el **83.8 %**. Por eso usamos `class_weight="balanced"` (para que el modelo preste igual atención a las 4 emociones) y medimos con **Macro-F1**.
2. **Pocos ejemplos de `ansiedad` en GoEmotions:** GoEmotions no tiene una etiqueta clínica de ansiedad, sino *nerviosismo* (`nervousness`), que solo aporta **84 casos en entrenamiento** y **8 casos en `validacion_go.csv`**. Por eso en GoEmotions cada frase de ansiedad que acierta o falla mueve su puntaje en 12.5 puntos porcentuales.
