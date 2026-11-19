# Informe de Resultados — Chatbot Emocional Académico

Todas las cifras de este informe provienen directamente de la ejecución real de `python -m entrenamiento.p08_experimentos` y están respaldadas en `modelos/resultados_experimentos.json`.

> **Alcance de esta evaluación:** Aquí se reportan **únicamente validación y diagnóstico exploratorio**. El conjunto de prueba final (*test*) permanece reservado para evaluarse una sola vez cuando sea entregado, usando `python -m entrenamiento.p10_evaluar_test --test <ruta_test.csv>`.

---

## 1. Composición de los datos y auditoría de fugas

El vocabulario, los pesos TF-IDF y los coeficientes de los clasificadores se ajustaron **exclusivamente con `datos de entrenamiento/preparados/train.csv`**. Las dos particiones de validación se mantuvieron completamente separadas.

### Conteos por partición y procedencia

| Partición | Procedencia (`origen`) | Alegría | Ansiedad | Enojo | Neutral | Tristeza | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| **Entrenamiento (`train.csv`)** | `go_emotions_es_mt` | 826 | 84 | 995 | 12,666 | 810 | **15,381** |
| **Entrenamiento (`train.csv`)** | `sintetico_claude` | 1,578 | 1,864 | 1,734 | 1,322 | 1,570 | **8,068** |
| **Entrenamiento (`train.csv`)** | `sintetico_asistido_IA` | 48 | 48 | 48 | 48 | 48 | **240** |
| **Total entrenamiento** | *Todas* | **2,452** | **1,996** | **2,777** | **14,036** | **2,428** | **23,689** |
| **Validación GoEmotions** | `go_emotions_es_mt` | 106 | 8 | 109 | 1,592 | 84 | **1,899** |
| **Validación sintética** | `sintetico_claude` | 422 | 136 | 266 | 678 | 430 | **1,932** |

### Hallazgos clave del dataset

1. **Cero fugas verificadas:** el solape exacto de textos y de `grupo` entre entrenamiento, validación GoEmotions y validación sintética es **0** en todos los pares.
2. **Desbalanceo marcado:** en entrenamiento, `neutral` representa el **59.25 %** de las frases (`14,036 / 23,689`) y en `validacion_go.csv` llega al **83.83 %** (`1,592 / 1,899`). Por eso la exactitud (*accuracy*) sola es engañosa y la métrica principal de comparación es **macro-F1**.
3. **Aproximación de nerviosismo a ansiedad:** en GoEmotions no existe una categoría clínica de ansiedad; la etiqueta `nervousness` (*nerviosismo*) se usó como aproximación a `ansiedad` y aporta únicamente **84 casos en entrenamiento** y **8 casos en `validacion_go.csv`**. La mayor parte de los ejemplos de `ansiedad` proviene de datos sintéticos (`1,864` de Claude y `48` asistidos por IA). En `validacion_go.csv`, cada acierto o fallo individual sobre esos 8 casos mueve el recall de `ansiedad` en 12.5 puntos porcentuales.

---

## 2. Comparación de modelos en validación

Se entrenaron seis modelos desde cero únicamente sobre `train.csv` (`SEMILLA = 42`): una referencia simple, tres variantes de Regresión Logística Multinomial para aislar el aporte de unigramas/bigramas/trigramas, y dos algoritmos adicionales mencionados en los apuntes del curso (Complement Naive Bayes y SVM Lineal).

| Modelo | N-gramas | Vocab. | Go Macro-F1 | Go Exact. | Sint. Macro-F1 | Sint. Exact. | Comb. Macro-F1 | Prom. Macro-F1 | Prob. nativas |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Referencia simple: BoW + Naive Bayes | `(1, 1)` | 9,037 | 0.3686 | 0.8494 | 0.8874 | 0.9073 | 0.8243 | 0.6280 | Sí |
| TF-IDF + Regresión Logística balanceada | `(1, 1)` | 9,037 | 0.5699 | 0.8552 | 0.9641 | 0.9669 | 0.8905 | 0.7670 | Sí |
| TF-IDF + Regresión Logística balanceada | `(1, 2)` | 29,068 | 0.6008 | 0.8699 | 0.9709 | 0.9746 | 0.9037 | 0.7858 | Sí |
| **TF-IDF + Regresión Logística balanceada (Ganador)** | **`(1, 3)`** | **44,761** | **0.6101** | **0.8705** | **0.9822** | **0.9850** | **0.9141** | **0.7961** | **Sí** |
| TF-IDF + Complement Naive Bayes | `(1, 3)` | 44,761 | 0.4421 | 0.8483 | 0.9768 | 0.9819 | 0.8956 | 0.7095 | Sí |
| TF-IDF + SVM Lineal balanceado | `(1, 3)` | 44,761 | 0.6152 | 0.8826 | 0.9882 | 0.9886 | 0.9257 | 0.8017 | No |

*(Nota: "Sint." corresponde exclusivamente a `validacion_sintetica.csv` generada con Claude; "Go" corresponde a `validacion_go.csv` de GoEmotions traducido; "Comb." evalúa las 3,831 frases juntas; "Prom. Macro-F1" es el promedio simple entre el macro-F1 de GoEmotions y el macro-F1 sintético para no dejar que una sola partición domine.)*

---

## 3. Comparación controlada: bigramas (1,2) frente a trigramas (1,3)

Para comprobar de forma limpia qué aportan los trigramas, se mantuvieron **todos los demás parámetros exactamente fijos**: `TfidfVectorizer(min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b[a-zñ]+\b")` y `LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)`.

| Indicador | Hasta bigramas `(1, 2)` | Hasta trigramas `(1, 3)` | Aporte de trigramas |
|---|---:|---:|---:|
| Tamaño de vocabulario | 29,068 | 44,761 | +15,693 términos |
| Validación GoEmotions — Macro-F1 | 0.6008 | 0.6101 | **+0.0093** |
| Validación GoEmotions — Exactitud | 0.8699 | 0.8705 | +0.0006 |
| Validación sintética — Macro-F1 | 0.9709 | 0.9822 | **+0.0113** |
| Validación sintética — Exactitud | 0.9746 | 0.9850 | +0.0104 |
| Validación combinada — Macro-F1 | 0.9037 | 0.9141 | **+0.0104** |
| Promedio Macro-F1 (Go + Sintética) | 0.7858 | 0.7961 | **+0.0103** |

**Lectura:** incorporar grupos de hasta tres palabras mejora de manera consistente **todas** las particiones de validación. En GoEmotions, la mayor mejora puntual se refleja en `ansiedad` (F1 sube de `0.4000` con bigramas a `0.4615` con trigramas, reduciendo falsos positivos sobre frases neutrales) y en validación sintética el macro-F1 sube `1.13` puntos porcentuales al capturar expresiones compuestas de tres palabras como `da mucha rabia` o `hoy me siento`.

---

## 4. Por qué se elige al ganador

Se selecciona **`TF-IDF (1,3) + Regresión Logística Multinomial balanceada`** como modelo ganador y modelo base del chatbot por cuatro razones concretas:

1. **Supera claramente a la referencia simple y a versiones más cortas:** frente a la referencia de Bolsa de Palabras + Naive Bayes, sube el macro-F1 en GoEmotions de `0.3686` a `0.6101` (`+0.2415`) y en validación sintética de `0.8874` a `0.9822` (`+0.0948`).
2. **Probabilidades nativas para pedir contexto:** la Regresión Logística Multinomial entrega directamente probabilidades por clase mediante la función Softmax (Semana 12, Sección 8.5), lo que permite mostrar barras de confianza reales y aplicar abstención transparente.
3. **Comparación honesta frente a SVM Lineal (`LinearSVC`):** `LinearSVC` obtiene un macro-F1 ligeramente superior cuando clasifica el 100 % de los mensajes (`0.6152` vs `0.6101` en GoEmotions, apenas `+0.0051`; `0.9882` vs `0.9822` en sintética). Sin embargo, `LinearSVC` devuelve distancias a hiperplanos, no probabilidades; al envolverlo en calibración (`CalibratedClassifierCV`) durante la exploración previa, su macro-F1 en GoEmotions cayó a `0.5180`, por debajo de la Regresión Logística.
4. **Explicabilidad y velocidad:** el algoritmo se estudió en profundidad en las Semanas 11 y 12, permite inspeccionar exactamente qué términos sumaron peso a cada emoción y se entrena en ~4 segundos, haciendo viable el reentrenamiento local desde la pestaña Aprendizaje.

---

## 5. Desempeño detallado del ganador (separando datos traducidos vs. sintéticos)

Es fundamental **no mezclar** en la interpretación el desempeño sobre `validacion_sintetica.csv` (`macro-F1 = 0.9822`) con el desempeño sobre `validacion_go.csv` (`macro-F1 = 0.6101`). La validación sintética comparte el estilo limpio y explícito de los ejemplos generados con Claude en entrenamiento, por lo que sus cifras son optimistas; `validacion_go.csv` refleja texto más ruidoso, corto, irónico y traducido automáticamente desde Reddit, donde la tarea es mucho más exigente.

### 5.1 Validación GoEmotions (`validacion_go.csv`, n = 1,899, origen `go_emotions_es_mt`)

- **Exactitud:** `0.8705` · **Macro Precisión:** `0.6699` · **Macro Recall:** `0.5674` · **Macro-F1:** `0.6101`

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| `alegria` | 0.6951 | 0.5377 | 0.6064 | 106 |
| `ansiedad` | 0.6000 | 0.3750 | 0.4615 | 8 |
| `enojo` | 0.5051 | 0.4587 | 0.4808 | 109 |
| `neutral` | 0.9118 | 0.9416 | 0.9265 | 1,592 |
| `tristeza` | 0.6377 | 0.5238 | 0.5752 | 84 |

**Matriz de confusión — Validación GoEmotions (filas = real, columnas = predicho):**

| Real \ Predicho | alegria | ansiedad | enojo | neutral | tristeza |
|---|---:|---:|---:|---:|---:|
| **alegria** | **57** | 0 | 3 | 45 | 1 |
| **ansiedad** | 1 | **3** | 0 | 4 | 0 |
| **enojo** | 1 | 0 | **50** | 58 | 0 |
| **neutral** | 22 | 2 | 45 | **1,499** | 24 |
| **tristeza** | 1 | 0 | 1 | 38 | **44** |

**Dónde falla en GoEmotions:** casi todos los errores ocurren entre cada emoción y `neutral` (por ejemplo, `58` mensajes reales de `enojo`, `45` de `alegria` y `38` de `tristeza` se predicen como `neutral`, mientras `45` mensajes `neutral` se predicen como `enojo`). Las confusiones directas entre emociones opuestas son mínimas (solo 1 caso de `alegria` predicho como `tristeza` y 1 de `tristeza` como `alegria`).

### 5.2 Validación sintética (`validacion_sintetica.csv`, n = 1,932, origen `sintetico_claude`)

- **Exactitud:** `0.9850` · **Macro Precisión:** `0.9852` · **Macro Recall:** `0.9795` · **Macro-F1:** `0.9822`

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| `alegria` | 0.9951 | 0.9716 | 0.9832 | 422 |
| `ansiedad` | 0.9847 | 0.9485 | 0.9663 | 136 |
| `enojo` | 0.9851 | 0.9925 | 0.9888 | 266 |
| `neutral` | 0.9839 | 0.9941 | 0.9890 | 678 |
| `tristeza` | 0.9771 | 0.9907 | 0.9838 | 430 |

**Matriz de confusión — Validación sintética (filas = real, columnas = predicho):**

| Real \ Predicho | alegria | ansiedad | enojo | neutral | tristeza |
|---|---:|---:|---:|---:|---:|
| **alegria** | **410** | 0 | 2 | 5 | 5 |
| **ansiedad** | 0 | **129** | 0 | 3 | 4 |
| **enojo** | 0 | 0 | **264** | 2 | 0 |
| **neutral** | 0 | 2 | 1 | **674** | 1 |
| **tristeza** | 2 | 0 | 1 | 1 | **426** |

---

## 6. Evaluación de la abstención (pedir contexto)

El chatbot etiqueta una emoción solo cuando la probabilidad máxima es `>= 0.45` **y** el margen entre la primera y la segunda emoción es `>= 0.10`. Si no se cumple alguna de las dos condiciones, se abstiene y pide más contexto al usuario.

| Partición | Total | Etiquetados | Pide contexto | Cobertura | Aciertos en etiquetados | Exactitud selectiva | Macro-F1 selectivo |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validación GoEmotions | 1,899 | 1,697 | 202 | 89.36 % | 1,530 | **90.16 %** (+3.11 pp) | **0.6179** (+0.0078) |
| Validación sintética | 1,932 | 1,851 | 81 | 95.81 % | 1,843 | **99.57 %** (+1.07 pp) | **0.9951** (+0.0129) |
| Validación combinada | 3,831 | 3,548 | 283 | 92.61 % | 3,373 | **95.07 %** (+2.25 pp) | **0.9442** (+0.0301) |

En las tres vistas, abstenerse ante señales débiles aumenta simultáneamente la exactitud selectiva y el macro-F1 selectivo manteniendo coberturas altas (~89 % a ~96 %).

---

## 7. Comprobaciones diagnósticas exploratorias (distintas de una evaluación final)

Se probaron **16 frases cotidianas escritas a mano** (todas verificadas con `en_entrenamiento = false`) para observar cómo responde el modelo ante negaciones, emociones indirectas y modismos locales. Estas comprobaciones son **exploratorias** para diagnosticar fortalezas y debilidades, y no sustituyen la evaluación en test.

| Frase diagnóstica | Tipo | Salida del modelo | Confianza | Margen | Observación honesta |
|---|---|---|---:|---:|---|
| No me fue del todo bien | Negación indirecta | `tristeza` | 46.9 % | 24.6 % | Captura desánimo leve justo sobre el umbral |
| No me fue del todo bien en el examen | Negación indirecta | `pedir_contexto` | 45.0 % | 25.6 % | Pide contexto (indicios entre ansiedad y tristeza) |
| No puedo estar más feliz con esta noticia | Negación alegre | `alegria` | 73.1 % | 62.5 % | Resuelve correctamente el «no» alegre |
| No me lo puedo creer, al fin aprobé el curso | Negación alegre | `pedir_contexto` | 26.0 % | 0.6 % | Evita adivinar mal y pide más contexto |
| No tengo clases mañana y luego iré al banco | Negación neutral | `neutral` | 52.4 % | 37.7 % | Resuelve correctamente el «no» neutral |
| No sé si la reunión empieza a las cuatro o a las cinco | Negación neutral | `neutral` | 75.7 % | 66.8 % | Resuelve correctamente el «no» informativo |
| No me preocupa en absoluto, todo salió genial | Negación difícil | `ansiedad` | 69.8 % | 59.6 % | **Fallo:** el peso alto de `me preocupa` domina sobre `no` |
| No estoy triste, solo estoy pensando en mis planes | Negación difícil | `tristeza` | 80.9 % | 74.8 % | **Fallo:** el unigrama `triste` (+1.85) supera a `no estoy` |
| No estoy enojado, solo cansado | Indirecta ambigua | `enojo` | 54.6 % | 34.0 % | **Fallo:** el unigrama `enojado` (+1.06) domina la suma lineal |
| Bueno, supongo que así son las cosas | Señal débil | `neutral` | 72.5 % | 62.9 % | Clasifica como neutral (lectura plausible) |
| No sé qué va a pasar mañana, tengo el pecho apretado | Ansiedad cotidiana | `ansiedad` | 57.8 % | 37.6 % | Detecta correctamente ansiedad |
| No dormí nada pensando en la presentación de mañana | Ansiedad cotidiana | `ansiedad` | 54.2 % | 31.5 % | Detecta correctamente ansiedad |
| Me da mucha rabia que no cumplan lo acordado | Enojo cotidiano | `enojo` | 85.8 % | 79.1 % | Detecta enojo apoyado en el trigrama `da mucha rabia` |
| Hoy me siento vacío y sin ganas de salir | Tristeza cotidiana | `tristeza` | 94.3 % | 91.1 % | Detecta tristeza con `hoy me siento` y `vacio` |
| Qué bueno que pudimos terminar el proyecto a tiempo | Alegría cotidiana | `alegria` | 50.2 % | 19.5 % | Detecta alegría con `que bueno` |
| No esperaba aprobar pero saqué veinte | Modismo local | `neutral` | 65.9 % | 55.3 % | **Fallo por vocabulario local:** ideal para corregir en Aprendizaje |

### Síntesis de limitaciones detectadas

1. **Negación directa de una palabra emocional muy fuerte:** cuando un mensaje niega explícitamente una emoción fuerte (`"No estoy triste..."`, `"No me preocupa..."`), un modelo lineal basado en TF-IDF suma el gran peso positivo de `triste` o `preocupa` y los trigramas negados pueden no haber aparecido con suficiente frecuencia en entrenamiento para invertir el signo.
2. **Brecha de dominio (traducción vs. sintético vs. español local):** expresiones propias de Perú como `"saqué veinte"` no aparecen en GoEmotions ni en el corpus sintético; la pestaña **Aprendizaje** permite corregir estos casos de manera controlada y reversible.
3. **Soporte muy bajo de `ansiedad` en GoEmotions:** con solo 8 mensajes en `validacion_go.csv`, la estimación de F1 en esa clase tiene alta varianza.
4. **Sarcasmo y contexto conversacional:** cada mensaje se analiza de forma individual sin memoria semántica profunda ni detección de ironía compleja.

---

## 8. Comando documentado para la evaluación final en test

Cuando entregues el archivo de test reservado, ejecuta una sola vez:

```bash
source .venv/bin/activate
python -m entrenamiento.p10_evaluar_test --test /ruta/al/archivo_test.csv
```

El script evaluará el modelo base congelado (`modelos/modelo_base.joblib`), verificará si existe solape con `train.csv` y guardará permanentemente `modelos/evaluacion_final/metricas_test.json` y `modelos/evaluacion_final/predicciones_test.csv`.
