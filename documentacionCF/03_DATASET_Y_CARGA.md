# 03. El dataset y cómo lo cargamos paso a paso

Si la profesora pregunta *"¿cómo cargan los datos?"* o *"¿cómo saben que no hay trampa?"*, aquí está explicado de forma directa.

---

## 1. ¿Cómo son los archivos de datos?

Son archivos **CSV** (tablas de texto separadas por comas, como una hoja de Excel). Cada fila es un mensaje en español con **4 columnas**:

| Columna | Qué guarda | Ejemplo |
|---|---|---|
| `texto` | La frase escrita en español | `"Hoy me siento vacío y sin ganas de salir"` |
| `etiqueta` | Una de las 5 emociones | `"tristeza"` |
| `grupo` | Código para agrupar frases parecidas y no mezclar copias entre entrenar y probar | `"imp_tristeza_0012"` |
| `origen` | De dónde salió la frase | `"go_emotions_es_mt"` |

---

## 2. Cuántos datos tenemos y cómo se dividen

Separamos los datos en **tres grupos que nunca se mezclan**:

1. **Entrenamiento (`26,469` frases):** es lo único con lo que estudia el modelo. Se junta de dos partes:
   - `train.csv` (`23,689` frases): `15,381` de **GoEmotions traducido al español** + `8,068` sintéticas (Claude) + `240` sintéticas iniciales.
   - `corpus_situaciones_implicitas.csv` (`2,780` frases, **556 por emoción**): situaciones cotidianas donde no se dice la emoción tal cual (por ejemplo *"ayer me robaron el teléfono"*), creadas para equilibrar palabras como `"ayer"`.
2. **Validación (`3,831` frases en 2 archivos):** el modelo **nunca estudia** con ellas; solo sirven como examen para comparar los 9 algoritmos:
   - `validacion_go.csv`: **1,899** frases reales traducidas de GoEmotions.
   - `validacion_sintetica.csv`: **1,932** frases sintéticas limpias.
3. **Prueba final / Test (reservado):** guardado aparte para evaluarlo **una sola vez** al final con `p10_evaluar_test.py`.

| Partición | Alegría | Ansiedad | Enojo | Neutral | Tristeza | Total |
|---|---:|---:|---:|---:|---:|---:|
| **Entrenamiento** | 3,008 | 2,552 | 3,333 | 14,592 | 2,984 | **26,469** |
| **Validación GoEmotions** | 106 | 8 | 109 | 1,592 | 84 | **1,899** |
| **Validación sintética** | 422 | 136 | 266 | 678 | 430 | **1,932** |

---

## 3. ¿Cómo los carga el código? (`entrenamiento/p02_datos.py`)

1. **Leer y revisar columnas (`cargar_csv_etiquetado`):**
   - Abre el archivo respetando tildes y la `ñ`.
   - Revisa que existan las columnas `texto` y `etiqueta`, borra filas vacías y comprueba que la etiqueta sea una de las 5 válidas (si hay una mal escrita, avisa en qué fila está).
2. **Unir el entrenamiento (`cargar_entrenamiento_completo`):**
   - Junta los `23,689` ejemplos de `train.csv` con los `2,780` cotidianos → **`26,469` en total**.
3. **Comprobar que no haya trampa (`verificar_sin_fuga`):**
   - Compara los textos y los códigos `grupo` entre entrenamiento y las dos pruebas de validación.
   - Resultado verificado: **0 textos repetidos** y **0 grupos repetidos** (cero fuga de datos).
4. **Limpiar y entrenar solo con estudio:**
   - `p03_preprocesamiento.py` pasa a minúsculas, cuida la `ñ` y marca negaciones con `neg_`.
   - El modelo solo aprende palabras y pesos con entrenamiento (`.fit()`), y sobre validación únicamente predice (`.predict()`).

---

## 4. Dos detalles honestos sobre los datos

1. **Neutral es mayoría:** representa el **55.1 %** en entrenamiento y el **83.8 %** en GoEmotions real. Por eso balanceamos las clases (`class_weight="balanced"`) y medimos con **Macro-F1**.
2. **Poca Ansiedad en GoEmotions:** GoEmotions solo trae la etiqueta *nerviosismo* (`nervousness`) con **84 frases en entrenamiento** y **8 en validación**, por eso cada frase que acierta o falla ahí mueve mucho su porcentaje.
