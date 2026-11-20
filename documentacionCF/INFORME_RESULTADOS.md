# Informe de Resultados — Chatbot Emocional Académico

Todas las cifras de este informe provienen directamente de la ejecución real de `python -m entrenamiento.p08_experimentos` y están respaldadas en `modelos/resultados_experimentos.json`.

> **Alcance de esta evaluación:** Aquí se reportan **únicamente validación y diagnóstico exploratorio**. El conjunto de prueba final (*test*) permanece reservado para evaluarse una sola vez cuando sea entregado, usando `python -m entrenamiento.p10_evaluar_test --test <ruta_test.csv>`.

---

## 1. Composición de los datos y auditoría de fugas

El vocabulario, los pesos TF-IDF y los coeficientes de los clasificadores se ajustaron **exclusivamente con los datos de entrenamiento** (`datos de entrenamiento/preparados/train.csv` + el corpus curado local `entrenamiento/corpus_situaciones_implicitas.csv`). Las dos particiones de validación se mantuvieron completamente separadas.

### Conteos por partición y procedencia

| Partición | Procedencia (`origen`) | Alegría | Ansiedad | Enojo | Neutral | Tristeza | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| **Entrenamiento (`train.csv`)** | `go_emotions_es_mt` | 826 | 84 | 995 | 12,666 | 810 | **15,381** |
| **Entrenamiento (`train.csv`)** | `sintetico_claude` | 1,578 | 1,864 | 1,734 | 1,322 | 1,570 | **8,068** |
| **Entrenamiento (`train.csv`)** | `sintetico_asistido_IA` | 48 | 48 | 48 | 48 | 48 | **240** |
| **Entrenamiento (curado local)** | `curado_situaciones_implicitas` | 556 | 556 | 556 | 556 | 556 | **2,780** |
| **Total entrenamiento** | *Todas* | **3,008** | **2,552** | **3,333** | **14,592** | **2,984** | **26,469** |
| **Validación GoEmotions** | `go_emotions_es_mt` | 106 | 8 | 109 | 1,592 | 84 | **1,899** |
| **Validación sintética** | `sintetico_claude` | 422 | 136 | 266 | 678 | 430 | **1,932** |

### Hallazgos clave del dataset

1. **Cero fugas verificadas:** el solape exacto de textos y de `grupo` entre entrenamiento (26,469 filas), validación GoEmotions y validación sintética es **0** en todos los pares.
2. **Desbalanceo marcado:** en entrenamiento, `neutral` representa el **55.13 %** de las frases (`14,592 / 26,469`) y en `validacion_go.csv` llega al **83.83 %** (`1,592 / 1,899`). Por eso la exactitud (*accuracy*) sola es engañosa y la métrica principal de comparación es **macro-F1**.
3. **Mitigación de sesgos y situaciones implícitas:** el corpus local `curado_situaciones_implicitas` (2,780 frases reproducibles con `p02b_generar_corpus_implicito.py`, 556 por clase) cubre situaciones cotidianas implícitas (robos, pérdidas, exámenes, sustentaciones, logros) y balancea palabras temporales (`ayer`, `hoy`, `mañana`) que anteriormente estaban sesgadas hacia `alegria`.
4. **Aproximación de nerviosismo a ansiedad:** en GoEmotions no existe una categoría clínica de ansiedad; la etiqueta `nervousness` (*nerviosismo*) se usó como aproximación a `ansiedad` y aporta únicamente **84 casos en entrenamiento** y **8 casos en `validacion_go.csv`**. En `validacion_go.csv`, cada acierto o fallo individual sobre esos 8 casos mueve el recall de `ansiedad` en 12.5 puntos porcentuales.

---

## 2. Comparación de los 9 modelos en validación

Se entrenaron **nueve modelos desde cero** únicamente sobre entrenamiento (`SEMILLA = 42`), aplicando limpieza en español con alcance de negación (`neg_` hasta 2 palabras tras `no/nunca/jamas/tampoco/ni`): una referencia simple, cuatro variantes de Regresión Logística (unigramas, bigramas, trigramas y el modelo híbrido ganador con subpalabras `char_wb`), Complement Naive Bayes, SVM Lineal, una Red Neuronal Multicapa (`MLPClassifier` con `TruncatedSVD`, Semana 14) y un Ensamble suave (`VotingClassifier`, Semana 12).

| Modelo | N-gramas | Vocab. | Go Macro-F1 | Go Exact. | Sint. Macro-F1 | Sint. Exact. | Comb. Macro-F1 | Prom. Macro-F1 | Prob. nativas |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Referencia simple: BoW + Naive Bayes | `(1, 1)` | 9,591 | 0.3433 | 0.8383 | 0.8852 | 0.9094 | 0.8152 | 0.6142 | Sí |
| TF-IDF + Regresión Logística balanceada | `(1, 1)` | 9,591 | 0.5694 | 0.8494 | 0.9590 | 0.9674 | 0.8826 | 0.7642 | Sí |
| TF-IDF + Regresión Logística balanceada | `(1, 2)` | 31,425 | 0.6101 | 0.8710 | 0.9709 | 0.9752 | 0.9060 | 0.7905 | Sí |
| TF-IDF + Regresión Logística balanceada | `(1, 3)` | 50,466 | 0.6121 | 0.8752 | 0.9814 | 0.9834 | 0.9161 | 0.7968 | Sí |
| **TF-IDF palabras (1,3) + subpalabras (3,5) + LR (Ganador)** | **`(1,3) + char (3,5)`** | **84,700** | **0.6654** | **0.8847** | **0.9915** | **0.9922** | **0.9290** | **0.8285** | **Sí** |
| TF-IDF + Complement Naive Bayes | `(1, 3)` | 50,466 | 0.4310 | 0.8452 | 0.9809 | 0.9845 | 0.8964 | 0.7059 | Sí |
| TF-IDF + SVM Lineal balanceado | `(1, 3)` | 50,466 | 0.6049 | 0.8826 | 0.9833 | 0.9814 | 0.9221 | 0.7941 | No |
| TF-IDF híbrido + SVD (150) + Red Neuronal MLP | `(1,3) + char (3,5)` | 84,700 | 0.4938 | 0.8547 | 0.8816 | 0.8929 | 0.8205 | 0.6877 | Sí |
| TF-IDF híbrido + Ensamble suave (LR + ComplementNB) | `(1,3) + char (3,5)` | 84,700 | 0.6363 | 0.8878 | 0.9920 | 0.9938 | 0.9299 | 0.8141 | Sí |

*(Nota: "Sint." corresponde exclusivamente a `validacion_sintetica.csv` generada con Claude; "Go" corresponde a `validacion_go.csv` de GoEmotions traducido; "Comb." evalúa las 3,831 frases juntas; "Prom. Macro-F1" es el promedio simple entre el macro-F1 de GoEmotions y el macro-F1 sintético para no dejar que una sola partición domine.)*

---

## 3. Comparación controlada: bigramas (1,2) frente a trigramas (1,3) y ajuste de hiperparámetros

### 3.1 Bigramas `(1, 2)` vs. trigramas `(1, 3)` (parámetros fijos)

| Indicador | Hasta bigramas `(1, 2)` | Hasta trigramas `(1, 3)` | Aporte de trigramas |
|---|---:|---:|---:|
| Tamaño de vocabulario | 31,425 | 50,466 | +19,041 términos |
| Validación GoEmotions — Macro-F1 | 0.6101 | 0.6121 | **+0.0020** |
| Validación GoEmotions — Exactitud | 0.8710 | 0.8752 | +0.0042 |
| Validación sintética — Macro-F1 | 0.9709 | 0.9814 | **+0.0105** |
| Validación sintética — Exactitud | 0.9752 | 0.9834 | +0.0082 |
| Validación combinada — Macro-F1 | 0.9060 | 0.9161 | **+0.0101** |
| Promedio Macro-F1 (Go + Sintética) | 0.7905 | 0.7968 | **+0.0063** |

Al sumar **subpalabras de caracteres `char_wb (3, 5)`** en `FeatureUnion` (`tfidf_hibrido_lr`), GoEmotions Macro-F1 da un salto adicional de `0.6121` a **`0.6654` (`+0.0533`)**, conectando raíces y conjugaciones en español (`robaron ↔ robo`, `preocupado ↔ preocupa`, diminutivos) sin diccionarios ni modelos preentrenados.

### 3.2 Rejilla reproducible de hiperparámetros en validación (`p08_experimentos.py`)

| `max_df` (palabras) | Regularización `C` | Go Macro-F1 | Go Exact. | Sint. Macro-F1 | Comb. Macro-F1 | Prom. Macro-F1 | Go F1 selectivo | Seleccionado |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1.0 | 1.0 | 0.6604 | 0.8852 | 0.9890 | 0.9263 | 0.8247 | 0.7014 | |
| 0.5 | 1.0 | 0.6640 | 0.8836 | 0.9900 | 0.9267 | 0.8270 | 0.7050 | |
| 0.5 | 2.0 | 0.6671 | 0.8847 | 0.9908 | 0.9282 | 0.8290 | 0.6692 | |
| **0.5** | **3.0** | **0.6654** | **0.8847** | **0.9915** | **0.9290** | **0.8285** | **0.6779** | **Sí** |
| 0.5 | 5.0 | 0.6635 | 0.8836 | 0.9902 | 0.9274 | 0.8268 | 0.6832 | |

---

## 4. Por qué se elige al ganador

Se selecciona **`TF-IDF híbrido (palabras 1–3 con max_df=0.5 + subpalabras char_wb 3–5) + Regresión Logística Multinomial balanceada (C=3.0)`** como modelo ganador y modelo base del chatbot:

1. **Supera a los otros 8 modelos en GoEmotions y Promedio:** frente a la referencia BoW + Naive Bayes, sube el macro-F1 en GoEmotions de `0.3433` a **`0.6654`** (`+0.3221`) y en validación sintética de `0.8852` a **`0.9915`** (`+0.1063`).
2. **Supera a la Red Neuronal (`MLPClassifier` + `TruncatedSVD`, Semana 14) y al Ensamble (`VotingClassifier`, Semana 12):** la red neuronal cae en GoEmotions (`0.4938`) porque comprimir 84,700 n-gramas a 150 componentes pierde términos poco frecuentes y no balancea clases minoritarias; el ensamble logra `0.6363` en GoEmotions (por debajo de `0.6654`) y pierde la explicabilidad lineal directa.
3. **Probabilidades nativas e interpretabilidad directa:** entrega probabilidades por Softmax (Semana 12, Sec. 8.5) para pedir contexto y permite inspeccionar exactamente qué palabras y subpalabras sumaron peso (`valor_tfidf × coef_`).

---

## 5. Desempeño detallado del ganador (separando datos traducidos vs. sintéticos)

Es fundamental **no mezclar** en la interpretación el desempeño sobre `validacion_sintetica.csv` (`macro-F1 = 0.9915`) con el desempeño sobre `validacion_go.csv` (`macro-F1 = 0.6654`). La validación sintética tiene estilo limpio y explícito; `validacion_go.csv` refleja texto ruidoso, corto e irónico traducido desde Reddit.

### 5.1 Validación GoEmotions (`validacion_go.csv`, n = 1,899, origen `go_emotions_es_mt`)

- **Exactitud:** `0.8847` · **Macro Precisión:** `0.6759` · **Macro Recall:** `0.6556` · **Macro-F1:** `0.6654`

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| `alegria` | 0.7551 | 0.6981 | 0.7255 | 106 |
| `ansiedad` | 0.5000 | 0.5000 | 0.5000 | 8 |
| `enojo` | 0.5377 | 0.5229 | 0.5302 | 109 |
| `neutral` | 0.9285 | 0.9378 | 0.9331 | 1,592 |
| `tristeza` | 0.6582 | 0.6190 | 0.6380 | 84 |

**Matriz de confusión — Validación GoEmotions (filas = real, columnas = predicho):**

| Real \ Predicho | alegria | ansiedad | enojo | neutral | tristeza |
|---|---:|---:|---:|---:|---:|
| **alegria** | **74** | 0 | 1 | 31 | 0 |
| **ansiedad** | 0 | **4** | 0 | 4 | 0 |
| **enojo** | 0 | 0 | **57** | 51 | 1 |
| **neutral** | 22 | 3 | 48 | **1,493** | 26 |
| **tristeza** | 2 | 1 | 0 | 29 | **52** |

### 5.2 Validación sintética (`validacion_sintetica.csv`, n = 1,932, origen `sintetico_claude`)

- **Exactitud:** `0.9922` · **Macro Precisión:** `0.9930` · **Macro Recall:** `0.9901` · **Macro-F1:** `0.9915`

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| `alegria` | 0.9976 | 0.9787 | 0.9880 | 422 |
| `ansiedad` | 0.9925 | 0.9779 | 0.9852 | 136 |
| `enojo` | 1.0000 | 1.0000 | 1.0000 | 266 |
| `neutral` | 0.9912 | 0.9985 | 0.9949 | 678 |
| `tristeza` | 0.9839 | 0.9953 | 0.9896 | 430 |

**Matriz de confusión — Validación sintética (filas = real, columnas = predicho):**

| Real \ Predicho | alegria | ansiedad | enojo | neutral | tristeza |
|---|---:|---:|---:|---:|---:|
| **alegria** | **413** | 0 | 0 | 3 | 6 |
| **ansiedad** | 0 | **133** | 0 | 2 | 1 |
| **enojo** | 0 | 0 | **266** | 0 | 0 |
| **neutral** | 0 | 1 | 0 | **677** | 0 |
| **tristeza** | 1 | 0 | 0 | 1 | **428** |

---

## 6. Evaluación de la abstención (pedir contexto + memoria de 1 turno)

El chatbot etiqueta una emoción solo cuando la probabilidad máxima es `>= 0.45` **y** el margen entre la primera y la segunda emoción es `>= 0.10`. Si no se cumple alguna de las dos condiciones, se abstiene, pide más contexto al usuario y **conserva ese mensaje durante 1 turno** (`contexto_previo`) para reclasificar combinando el mensaje previo con la aclaración.

| Partición | Total | Etiquetados | Pide contexto | Cobertura | Aciertos en etiquetados | Exactitud selectiva | Macro-F1 selectivo |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validación GoEmotions | 1,899 | 1,822 | 77 | 95.95 % | 1,634 | **89.68 %** (+1.21 pp) | **0.6779** (+0.0125) |
| Validación sintética | 1,932 | 1,904 | 28 | 98.55 % | 1,898 | **99.68 %** (+0.46 pp) | **0.9965** (+0.0050) |
| Validación combinada | 3,831 | 3,726 | 105 | 97.26 % | 3,532 | **94.79 %** (+1.20 pp) | **0.9407** (+0.0117) |

---

## 7. Comprobaciones diagnósticas exploratorias (distintas de una evaluación final)

Se probaron **16 frases cotidianas escritas a mano** (todas verificadas con `en_entrenamiento = false`) para observar fortalezas y debilidades ante negaciones, emociones indirectas y modismos locales:

| Frase diagnóstica | Tipo | Salida del modelo | Confianza | Margen | Observación honesta |
|---|---|---|---:|---:|---|
| No me fue del todo bien | Negación indirecta | `neutral` | 45.3 % | 14.6 % | Señal leve en el límite del umbral |
| No me fue del todo bien en el examen | Negación indirecta | `pedir_contexto` | 28.8 % | 0.4 % | Pide contexto y activa memoria de 1 turno |
| No puedo estar más feliz con esta noticia | Negación alegre | `alegria` | 95.8 % | 93.9 % | Resuelve con claridad el «no» alegre |
| No me lo puedo creer, al fin aprobé el curso | Negación alegre | `pedir_contexto` | 40.7 % | 21.0 % | Evita adivinar mal y pide más contexto |
| No tengo clases mañana y luego iré al banco | Negación neutral | `neutral` | 89.3 % | 84.6 % | Resuelve correctamente el «no» neutral |
| No sé si la reunión empieza a las cuatro o a las cinco | Negación neutral | `neutral` | 79.7 % | 62.9 % | Resuelve correctamente el «no» informativo |
| No me preocupa en absoluto, todo salió genial | Negación difícil | `alegria` | 55.0 % | 29.5 % | **Mejorado:** antes fallaba en ansiedad; ahora detecta alegría |
| No estoy triste, solo estoy pensando en mis planes | Negación difícil | `tristeza` | 63.2 % | 33.0 % | **Limitación persistente:** las subpalabras de `triste` aún pesan |
| No estoy enojado, solo cansado | Indirecta ambigua | `neutral` | 72.3 % | 49.9 % | **Mejorado:** antes fallaba en enojo; ahora clasifica neutral |
| Bueno, supongo que así son las cosas | Señal débil | `neutral` | 90.4 % | 85.5 % | Clasifica como neutral (lectura plausible) |
| No sé qué va a pasar mañana, tengo el pecho apretado | Ansiedad cotidiana | `ansiedad` | 52.0 % | 21.3 % | Detecta correctamente ansiedad |
| No dormí nada pensando en la presentación de mañana | Ansiedad cotidiana | `ansiedad` | 91.2 % | 85.4 % | Detecta correctamente ansiedad |
| Me da mucha rabia que no cumplan lo acordado | Enojo cotidiano | `enojo` | 96.1 % | 94.5 % | Detecta enojo con alta confianza |
| Hoy me siento vacío y sin ganas de salir | Tristeza cotidiana | `tristeza` | 99.6 % | 99.4 % | Detecta tristeza con alta confianza |
| Qué bueno que pudimos terminar el proyecto a tiempo | Alegría cotidiana | `alegria` | 87.6 % | 78.1 % | Detecta alegría con alta confianza |
| No esperaba aprobar pero saqué veinte | Modismo local | `neutral` | 86.5 % | 80.9 % | **Caso ideal de Aprendizaje:** modismo peruano ausente en train |

### Síntesis de limitaciones reales (transparencia científica)

1. **Límite teórico de TF-IDF ante situaciones implícitas nunca vistas:** al ser un modelo entrenado 100 % desde cero sin modelos preentrenados (BERT/LLMs), si un usuario describe una situación nueva con vocabulario que no apareció en entrenamiento, TF-IDF no posee conocimiento del mundo para inferirla automáticamente.
2. **Negación directa de palabras emocionales fuertes con subpalabras:** el marcado `neg_` resolvió casos como *"No me preocupa en absoluto, todo salió genial"* y *"No estoy enojado, solo cansado"*, pero aún puede fallar en *"No estoy triste..."* porque `char_wb` comparte fragmentos de la raíz `trist`.
3. **Soporte muy bajo de `ansiedad` en GoEmotions:** con solo 8 mensajes en `validacion_go.csv`, la estimación de F1 en esa clase tiene alta varianza.
4. **Sarcasmo y conversaciones largas:** la memoria conversacional cubre únicamente 1 turno tras pedir contexto; no existe comprensión semántica profunda ni detección de ironía compleja.

---

## 8. Comando documentado para la evaluación final en test

Cuando entregues el archivo de test reservado, ejecuta una sola vez:

```bash
source .venv/bin/activate
python -m entrenamiento.p10_evaluar_test --test /ruta/al/archivo_test.csv
```

El script evaluará el modelo base congelado (`modelos/modelo_base.joblib`), verificará si existe solape con entrenamiento y guardará permanentemente `modelos/evaluacion_final/metricas_test.json` y `modelos/evaluacion_final/predicciones_test.csv`.
