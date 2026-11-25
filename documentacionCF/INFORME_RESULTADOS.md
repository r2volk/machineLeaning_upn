# Informe de Resultados — Chatbot Emocional Académico

Todas las cifras de este informe provienen directamente de la ejecución real de `python3 -m entrenamiento.p08_experimentos` y están guardadas en `modelos/resultados_experimentos.json`.

> **Alcance:** aquí se reportan **únicamente validación y diagnóstico exploratorio**. El conjunto de prueba final (*test*) permanece reservado para evaluarse una sola vez al final con `p10_evaluar_test.py`.

---

## 1. Datos utilizados y auditoría de fugas

El modelo aprendió **solo con entrenamiento** (`train.csv` + `corpus_situaciones_implicitas.csv`). Las dos particiones de validación quedaron completamente separadas.

| Partición | Procedencia (`origen`) | Alegría | Ansiedad | Enojo | Neutral | Tristeza | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| **Entrenamiento (`train.csv`)** | `go_emotions_es_mt` | 826 | 84 | 995 | 12,666 | 810 | **15,381** |
| **Entrenamiento (`train.csv`)** | `sintetico_claude` | 1,578 | 1,864 | 1,734 | 1,322 | 1,570 | **8,068** |
| **Entrenamiento (`train.csv`)** | `sintetico_asistido_IA` | 48 | 48 | 48 | 48 | 48 | **240** |
| **Entrenamiento (curado local)** | `curado_situaciones_implicitas` | 556 | 556 | 556 | 556 | 556 | **2,780** |
| **Total entrenamiento** | *Todas* | **3,008** | **2,552** | **3,333** | **14,592** | **2,984** | **26,469** |
| **Validación GoEmotions** | `go_emotions_es_mt` | 106 | 8 | 109 | 1,592 | 84 | **1,899** |
| **Validación sintética** | `sintetico_claude` | 422 | 136 | 266 | 678 | 430 | **1,932** |

### Puntos clave del dataset
1. **0 fugas verificadas:** hay **0 textos repetidos** y **0 grupos repetidos** entre entrenamiento (`26,469`) y las validaciones.
2. **Desbalanceo hacia Neutral:** representa el **55.13 %** en entrenamiento y el **83.83 %** en `validacion_go.csv`. Por eso comparamos con **Macro-F1** (donde las 5 emociones valen igual).
3. **Corpus cotidiano (`2,780` frases):** cubre situaciones implícitas (robos, pérdidas, exámenes, logros) y equilibra palabras temporales (`ayer`, `hoy`, `mañana`).
4. **Poca Ansiedad en GoEmotions:** proviene de *nerviosismo* (`nervousness`) con **84 casos en entrenamiento** y **8 en `validacion_go.csv`**.

---

## 2. Comparación de los 9 modelos en validación

Todos entrenados desde cero con `SEMILLA = 42` y el mismo preprocesamiento (`neg_` hasta 2 palabras):

| Modelo | N-gramas | Vocab. | Go Macro-F1 | Go Exact. | Sint. Macro-F1 | Sint. Exact. | Comb. Macro-F1 | Prom. Macro-F1 | ¿Da % nativo? |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Referencia simple: BoW + Naive Bayes | `(1, 1)` | 9,610 | 0.3413 | 83.78 % | 0.8833 | 90.79 % | 0.8129 | 0.6123 | Sí |
| TF-IDF + Regresión Logística balanceada | `(1, 1)` | 9,610 | 0.5924 | 85.57 % | 0.9552 | 96.43 % | 0.8831 | 0.7738 | Sí |
| TF-IDF + Regresión Logística balanceada | `(1, 2)` | 31,472 | 0.6070 | 86.99 % | 0.9691 | 97.46 % | 0.9034 | 0.7880 | Sí |
| TF-IDF + Regresión Logística balanceada | `(1, 3)` | 50,508 | 0.6102 | 87.36 % | 0.9797 | 98.29 % | 0.9133 | 0.7950 | Sí |
| **TF-IDF palabras (1,3) + letras (3,5) + LR ★** | **`(1,3) + char (3,5)`** | **84,826** | **0.6654** | **88.47 %** | **0.9895** | **99.02 %** | **0.9272** | **0.8275** | **Sí** |
| TF-IDF + Complement Naive Bayes | `(1, 3)` | 50,508 | 0.4305 | 84.52 % | 0.9809 | 98.45 % | 0.8961 | 0.7057 | Sí |
| TF-IDF + SVM Lineal balanceado | `(1, 3)` | 50,508 | 0.6015 | 88.31 % | 0.9902 | 99.02 % | 0.9257 | 0.7958 | No |
| TF-IDF híbrido + SVD (150) + Red Neuronal MLP | `(1,3) + char (3,5)` | 84,826 | 0.5157 | 86.15 % | 0.8821 | 89.23 % | 0.8252 | 0.6989 | Sí |
| TF-IDF híbrido + Ensamble suave (LR + CNB) | `(1,3) + char (3,5)` | 84,826 | 0.6388 | 88.89 % | 0.9915 | 99.33 % | 0.9305 | 0.8152 | Sí |

---

## 3. Comparación bigramas vs. trigramas y ajuste de parámetros

### 3.1 Bigramas `(1, 2)` vs. Trigramas `(1, 3)` (parámetros fijos `C=1.0`, `min_df=2`)

| Indicador | Hasta bigramas `(1, 2)` | Hasta trigramas `(1, 3)` | Aporte de trigramas |
|---|---:|---:|---:|
| Tamaño de vocabulario | 31,472 | 50,508 | +19,036 términos |
| Validación GoEmotions — Macro-F1 | 0.6070 | 0.6102 | **+0.0032** |
| Validación GoEmotions — Exactitud | 86.99 % | 87.36 % | +0.37 pp |
| Validación sintética — Macro-F1 | 0.9691 | 0.9797 | **+0.0106** |
| Validación sintética — Exactitud | 97.46 % | 98.29 % | +0.83 pp |
| Validación combinada — Macro-F1 | 0.9034 | 0.9133 | **+0.0099** |
| Promedio Macro-F1 (Go + Sintética) | 0.7880 | 0.7950 | **+0.0070** |

Al agregar **subpalabras de 3 a 5 letras (`char_wb`)** y ajustar `max_df=0.5`, `C=3.0`, GoEmotions Macro-F1 sube hasta **`0.6654`**.

### 3.2 Rejilla de hiperparámetros (`max_df`, `C`) en validación

| `max_df` | `C` | Go Macro-F1 | Go Exact. | Sint. Macro-F1 | Comb. Macro-F1 | Prom. Macro-F1 | Go F1 selectivo | Elegido |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1.0 | 1.0 | 0.6585 | 88.41 % | 0.9871 | 0.9243 | 0.8228 | 0.7037 | |
| 0.5 | 1.0 | 0.6570 | 88.26 % | 0.9871 | 0.9236 | 0.8220 | 0.7025 | |
| 0.5 | 2.0 | 0.6636 | 88.36 % | 0.9899 | 0.9267 | 0.8267 | 0.6809 | |
| **0.5** | **3.0** | **0.6654** | **88.47 %** | **0.9895** | **0.9272** | **0.8275** | **0.6672** | **Sí** |
| 0.5 | 5.0 | 0.6601 | 88.31 % | 0.9897 | 0.9266 | 0.8249 | 0.6764 | |

---

## 4. Por qué se elige al ganador

1. **Mayor Macro-F1 en texto real (`0.6654`) y mayor promedio (`0.8275`)** entre los 9 modelos.
2. **Supera a `TruncatedSVD (150) + MLPClassifier` (`0.5157`)**, ya que comprimir `84,826` rasgos a `150` pierde palabras emocionales poco frecuentes.
3. **Supera a SVM Lineal (`0.6015`) y Ensamble (`0.6388`)**, entrega probabilidades directas (Softmax) y permite ver exactamente qué palabras influyeron en cada predicción.

---

## 5. Detalle del modelo ganador por partición

### 5.1 Validación GoEmotions (`validacion_go.csv`, n = 1,899)
- **Exactitud:** `88.47 %` · **Macro Precisión:** `0.6783` · **Macro Recall:** `0.6538` · **Macro-F1:** `0.6654`

| Emoción | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| `alegria` | 0.7604 | 0.6887 | 0.7228 | 106 |
| `ansiedad` | 0.5000 | 0.5000 | 0.5000 | 8 |
| `enojo` | 0.5278 | 0.5229 | 0.5253 | 109 |
| `neutral` | 0.9280 | 0.9384 | 0.9332 | 1,592 |
| `tristeza` | 0.6753 | 0.6190 | 0.6460 | 84 |

**Matriz de confusión GoEmotions (filas = real, columnas = predicho):**

| Real \ Predicho | Alegría | Ansiedad | Enojo | Neutral | Tristeza |
|---|---:|---:|---:|---:|---:|
| **Alegría** | **73** | 0 | 1 | 32 | 0 |
| **Ansiedad** | 0 | **4** | 0 | 4 | 0 |
| **Enojo** | 0 | 0 | **57** | 51 | 1 |
| **Neutral** | 21 | 3 | 50 | **1,494** | 24 |
| **Tristeza** | 2 | 1 | 0 | 29 | **52** |

### 5.2 Validación sintética (`validacion_sintetica.csv`, n = 1,932)
- **Exactitud:** `99.02 %` · **Macro Precisión:** `0.9909` · **Macro Recall:** `0.9882` · **Macro-F1:** `0.9895`

| Emoción | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| `alegria` | 0.9976 | 0.9692 | 0.9832 | 422 |
| `ansiedad` | 0.9925 | 0.9779 | 0.9852 | 136 |
| `enojo` | 0.9963 | 1.0000 | 0.9981 | 266 |
| `neutral` | 0.9912 | 0.9985 | 0.9949 | 678 |
| `tristeza` | 0.9772 | 0.9953 | 0.9862 | 430 |

**Matriz de confusión Sintética:**

| Real \ Predicho | Alegría | Ansiedad | Enojo | Neutral | Tristeza |
|---|---:|---:|---:|---:|---:|
| **Alegría** | **409** | 0 | 1 | 3 | 9 |
| **Ansiedad** | 0 | **133** | 0 | 2 | 1 |
| **Enojo** | 0 | 0 | **266** | 0 | 0 |
| **Neutral** | 0 | 1 | 0 | **677** | 0 |
| **Tristeza** | 1 | 0 | 0 | 1 | **428** |

---

## 6. Cuando pide contexto (abstención + memoria de 1 turno)

El chatbot clasifica solo si `prob >= 0.45` y `margen >= 0.10`; si no, pide más contexto y recuerda tu mensaje durante 1 turno:

| Partición | Total | Etiquetados | Pide contexto | Cobertura | Aciertos | Exactitud selectiva | Macro-F1 selectivo |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validación GoEmotions | 1,899 | 1,824 | 75 | 96.05 % | 1,632 | **89.47 %** (+1.00 pp) | **0.6672** |
| Validación sintética | 1,932 | 1,897 | 35 | 98.19 % | 1,891 | **99.68 %** (+0.66 pp) | **0.9964** |
| Validación combinada | 3,831 | 3,721 | 110 | 97.13 % | 3,523 | **94.68 %** (+0.89 pp) | **0.9390** |

---

## 7. Pruebas exploratorias sobre 16 frases cotidianas (fuera de entrenamiento)

| Frase probada | Esperada | Salida del modelo | Confianza | Margen |
|---|---|---|---:|---:|
| No me fue del todo bien | tristeza o pedir contexto | `pedir_contexto` | 32.1 % | 1.4 % |
| No me fue del todo bien en el examen | tristeza, ansiedad o contexto | `pedir_contexto` | 28.8 % | 3.4 % |
| No puedo estar más feliz con esta noticia | alegria | `alegria` | 95.8 % | 94.0 % |
| No me lo puedo creer, al fin aprobé el curso | alegria | `pedir_contexto` | 35.8 % | 7.5 % |
| No tengo clases mañana y luego iré al banco | neutral | `neutral` | 88.9 % | 84.1 % |
| No sé si la reunión empieza a las cuatro o a las cinco | neutral | `neutral` | 83.1 % | 69.1 % |
| No me preocupa en absoluto, todo salió genial | alegria | `alegria` | 61.8 % | 40.5 % |
| No estoy triste, solo estoy pensando en mis planes | neutral | `tristeza` | 63.4 % | 34.3 % |
| No estoy enojado, solo cansado | tristeza / neutral / contexto | `neutral` | 68.7 % | 42.4 % |
| Bueno, supongo que así son las cosas | neutral | `neutral` | 90.8 % | 86.0 % |
| No sé qué va a pasar mañana, tengo el pecho apretado | ansiedad | `pedir_contexto` | 40.0 % | 0.6 % |
| No dormí nada pensando en la presentación de mañana | ansiedad | `ansiedad` | 92.6 % | 88.0 % |
| Me da mucha rabia que no cumplan lo acordado | enojo | `enojo` | 94.4 % | 91.4 % |
| Hoy me siento vacío y sin ganas de salir | tristeza | `tristeza` | 99.6 % | 99.4 % |
| Qué bueno que pudimos terminar el proyecto a tiempo | alegria | `alegria` | 87.6 % | 78.2 % |
| No esperaba aprobar pero saqué veinte | alegria (para probar Aprendizaje) | `neutral` | 86.4 % | 80.8 % |

### Limitaciones transparentes
1. Sin modelos preentrenados de internet, TF-IDF no deduce situaciones implícitas cuyo vocabulario nunca vio en entrenamiento.
2. En frases con negación directa sobre palabras muy fuertes (como *"No estoy triste..."*), las subpalabras de `triste` aún pueden pesar.
3. Solo hay `8` casos de `ansiedad` en `validacion_go.csv` y la memoria conversacional cubre 1 turno tras pedir contexto.
