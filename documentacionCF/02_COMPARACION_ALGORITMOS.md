# 02. Comparación de los 9 algoritmos y Reducción de Dimensionalidad

Esta es una de las partes más importantes para tu exposición. La profesora quiere ver **que comparaste varios algoritmos con datos reales** y que **entiendes por qué ganó uno y por qué perdieron los demás**.

---

## 1. ¿Por qué no nos quedamos solo con la Exactitud (*Accuracy*)?

Imagina un examen donde 84 de cada 100 frases son **neutrales**. Si un modelo flojo dice *"neutral"* a absolutamente todo, sacaría **84 % de exactitud** sin saber detectar ni alegría, ni tristeza, ni enojo, ni ansiedad.

Por eso usamos **Macro-F1** como métrica principal:
- Primero medimos el **F1** de cada emoción por separado (equilibrio entre *no dar falsas alarmas* y *no dejar pasar casos reales*).
- Luego sacamos el **promedio simple de las 5 emociones**. Así todas valen lo mismo (20 % cada una), y ningún modelo puede hacer trampa diciendo solo *"neutral"*.

Además, evaluamos siempre en **dos validaciones separadas**:
- **Validación GoEmotions (1,899 frases):** mensajes reales de internet traducidos al español. Es la prueba más difícil y realista (tiene ruido, frases cortas y 83.8 % de frases neutrales).
- **Validación sintética (1,932 frases):** frases escritas de forma limpia y clara. Sirve para comprobar si el modelo aprende bien patrones claros, pero siempre la separamos de GoEmotions para no inflar los resultados.

---

## 2. Tabla comparativa de los 9 modelos

Todos se entrenaron con los **mismos 26,469 ejemplos** y la **misma semilla (`42`)**:

| # | Modelo | Qué hace en palabras simples | GoEmotions Macro-F1 | Sintética Macro-F1 | Promedio Macro-F1 | ¿Da probabilidades directas? |
|---|---|---|---:|---:|---:|:---:|
| 1 | **Bolsa de palabras (1,1) + Naive Bayes** | Cuenta palabras sueltas y calcula probabilidades básicas (nuestra referencia inicial). | `0.3413` | `0.8833` | `0.6123` | Sí |
| 2 | **TF-IDF (1,1) + Regresión Logística** | Cambia el conteo simple por TF-IDF y balancea las 5 emociones, solo con palabras sueltas. | `0.5924` | `0.9552` | `0.7738` | Sí |
| 3 | **TF-IDF (1,2) + Regresión Logística** | Agrega parejas de palabras seguidas (*bigramas*, como `"no quiero"`). | `0.6070` | `0.9691` | `0.7880` | Sí |
| 4 | **TF-IDF (1,3) + Regresión Logística** | Agrega tríos de palabras seguidas (*trigramas*, como `"da mucha rabia"`). | `0.6102` | `0.9797` | `0.7950` | Sí |
| **5** | **TF-IDF híbrido (palabras 1-3 + subpalabras 3-5) + Regresión Logística (GANADOR)** | **Une palabras (hasta 3 seguidas) + pedacitos de 3 a 5 letras + ajuste `max_df=0.5` y `C=3.0`.** | **`0.6654`** | **`0.9895`** | **`0.8275`** | **Sí** |
| 6 | **TF-IDF (1,3) + Complement Naive Bayes** | Versión de Naive Bayes pensada para datos desbalanceados. | `0.4305` | `0.9809` | `0.7057` | Sí |
| 7 | **TF-IDF (1,3) + SVM Lineal** | Busca la frontera más ancha entre clases. | `0.6015` | `0.9902` | `0.7958` | No |
| 8 | **TF-IDF híbrido + TruncatedSVD (150) + Red Neuronal MLP** | Comprime las 84,826 columnas a 150 (reducción de dimensionalidad) y las pasa por una red neuronal. | `0.5157` | `0.8821` | `0.6989` | Sí |
| 9 | **TF-IDF híbrido + Ensamble suave (LR + ComplementNB)** | Combina las probabilidades de Regresión Logística y Complement Naive Bayes. | `0.6388` | `0.9915` | `0.8152` | Sí |

---

## 3. Cómo explicar cada familia de algoritmos en voz alta

### A. Naive Bayes (Modelos 1 y 6)
- **Cómo funciona:** asume que cada palabra aporta de forma independiente, como si mezclaras todas las palabras de una frase en una bolsa y contaras en qué emoción suele aparecer cada una.
- **Cómo le fue:** en frases muy limpias (validación sintética) funciona bien (`0.8852` y `0.9809`), pero en texto real de GoEmotions le cuesta mucho (`0.3433` y `0.4310`) porque tiende a mandar demasiadas cosas a la clase más grande (`neutral`).
- **Para qué nos sirvió:** como **punto de partida** (línea base) para demostrar cuánto mejora el sistema al pasar a modelos más completos.

### B. Regresión Logística Multinomial (Modelos 2, 3, 4 y 5 — el ganador)
- **Cómo funciona:** aprende un **peso** (positivo o negativo) para cada palabra o grupo de palabras en cada una de las 5 emociones. Cuando llega un mensaje nuevo, suma los pesos y pasa el resultado por la función **Softmax**, que convierte esos puntajes en porcentajes que suman 100 %.
- **Por qué fue mejorando paso a paso:**
  - Al pasar de conteo simple a **TF-IDF + balanceo (`class_weight="balanced"`)**, GoEmotions saltó de `0.3433` a `0.5694`.
  - Al sumar **bigramas (1,2)** subió a `0.6101` y con **trigramas (1,3)** subió a `0.6121` (y en sintética de `0.9709` a `0.9814`).
  - Al sumar **subpalabras de 3 a 5 letras (`char_wb`)** y ajustar `max_df=0.5` y `C=3.0`, dio el mayor salto: **`0.6654` en GoEmotions** y **`0.8285` de promedio**.
- **Por qué es el ganador:**
  1. Tiene el **mejor Macro-F1 en GoEmotions (`0.6654`)** y el **mejor promedio global (`0.8285`)** de los 9 modelos.
  2. Entrega **probabilidades nativas**, indispensables para que el chatbot sepa cuándo pedir más contexto.
  3. Es **100 % explicable**: podemos multiplicar cada palabra por su peso y mostrarle al usuario exactamente qué términos inclinaron la decisión.
  4. Entrena en pocos segundos en una laptop normal.

### C. SVM Lineal — `LinearSVC` (Modelo 7)
- **Cómo funciona:** en vez de pensar en probabilidades, busca trazar la "pared" (hiperplano) más separada posible entre las emociones.
- **Por qué no lo elegimos:** obtuvo `0.6049` en GoEmotions (por debajo del ganador `0.6654`), y además **no entrega probabilidades directamente**. Cuando intentamos agregarle una calibración extra para sacar porcentajes, su rendimiento en GoEmotions bajó todavía más.

### D. Técnica de Reducción de Dimensionalidad (`TruncatedSVD`) + Red Neuronal (`MLPClassifier`) (Modelo 8 — Semana 14)
Esta pregunta suele salir mucho en la exposición. Aquí está explicada de la forma más fácil posible:

- **¿Qué problema queríamos resolver con la reducción de dimensionalidad?**
  Nuestro vectorizador híbrido genera **84,700 columnas** (una columna por cada palabra, grupo de palabras o pedacito de letras). Una Red Neuronal clásica (`MLPClassifier`, vista en la Semana 14) no puede entrenarse rápido ni bien con 84,700 entradas llenas de ceros en una computadora normal.
- **¿Qué técnica de reducción de dimensionalidad usamos?**
  Usamos **`TruncatedSVD`** (con `150` componentes) seguida de `Normalizer`. En palabras simples, `TruncatedSVD` es la **versión de PCA pensada para texto** (matrices con muchos ceros). Comprime las 84,700 columnas originales en **150 columnas resumen** que intentan guardar los patrones más importantes.
- **¿Qué hicimos después?**
  Esas 150 columnas resumen entraron a una **Red Neuronal Perceptrón Multicapa (`MLPClassifier`)** con una capa oculta de 64 neuronas y parada temprana (`early_stopping=True`).
- **¿Qué pasó en la prueba real?**
  Obtuvo `0.5157` en GoEmotions y `0.8821` en validación sintética (`0.6989` de promedio), bastante por debajo de nuestra Regresión Logística (`0.6654` y `0.9895`).
- **¿Por qué pasó eso? (la respuesta que impresiona a la profesora):**
  Porque en clasificación de texto, muchas palabras clave de una emoción aparecen pocas veces (por ejemplo, *"robaron"* o *"vacío"*). Al aplastar 84,700 columnas a solo 150 resúmenes, esos detalles poco frecuentes se pierden; además, `MLPClassifier` no balancea automáticamente las clases minoritarias como sí hace `LogisticRegression(class_weight="balanced")`.
  > **Idea clave para decir en clase:** *"No elegimos el algoritmo más famoso ni el más complejo por moda; probamos Reducción de Dimensionalidad con Red Neuronal, medimos con los mismos datos y nos quedamos con el que demostró mejores resultados."*

### E. Método Ensamblado — `VotingClassifier` (Modelo 9 — Semana 12)
- **Cómo funciona:** combina dos modelos distintos (Regresión Logística con peso 3 + Complement Naive Bayes con peso 1) y promedia sus probabilidades (*votación suave*).
- **Cómo le fue:** en validación sintética fue muy alto (`0.9920`), pero en GoEmotions bajó a `0.6363` frente a `0.6654` de la Regresión Logística sola, y además pierde la explicación directa de qué palabras pesaron más. Por eso tampoco superó al ganador.

---

## 4. Cómo afinamos el modelo ganador (sin adivinar)

En `p08_experimentos.py` probamos de forma ordenada distintos valores de **`max_df`** (descartar palabras comodín demasiado repetidas en más del 50 % de los textos) y de **`C`** (qué tan estricto es el modelo para no memorizar ruido — regularización L2, Semana 11):

| `max_df` (palabras) | `C` | GoEmotions Macro-F1 | Sintética Macro-F1 | Promedio Macro-F1 |
|---:|---:|---:|---:|---:|
| `1.0` | `1.0` | `0.6604` | `0.9890` | `0.8247` |
| `0.5` | `1.0` | `0.6640` | `0.9900` | `0.8270` |
| `0.5` | `2.0` | `0.6671` | `0.9908` | `0.8290` |
| **`0.5`** | **`3.0`** | **`0.6654`** | **`0.9915`** | **`0.8285`** |
| `0.5` | `5.0` | `0.6635` | `0.9902` | `0.8268` |

Elegimos **`max_df = 0.5`** (solo en palabras, dejando libres las subpalabras de letras) y **`C = 3.0`** porque logra el mejor equilibrio entre GoEmotions, validación sintética y buen desempeño cuando el chatbot pide contexto.
