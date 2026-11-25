# 02. Comparación de los 9 algoritmos y Reducción de Dimensionalidad

La profesora quiere ver **que comparaste varios algoritmos con los mismos datos** y que **sabes explicar con palabras simples por qué ganó uno**.

---

## 1. ¿Por qué medimos con Macro-F1 y no solo Exactitud?

Imagina un examen donde 84 de cada 100 frases son **neutrales**. Si un modelo perezoso dice *"neutral"* a todo, sacaría **84 % de exactitud** sin saber reconocer ni alegría, ni tristeza, ni enojo, ni ansiedad.

Por eso usamos **Macro-F1**:
- Mide el acierto (**F1**) de cada una de las **5 emociones por separado**.
- Saca el **promedio simple de las 5** (cada emoción vale 20 %). Así ninguna clase grande tapa a las demás.

Además, medimos siempre en **dos pruebas separadas**:
- **Prueba real (GoEmotions, 1,899 frases):** comentarios reales traducidos al español. Es la más difícil porque hay ruido y el **83.8 %** es neutral.
- **Prueba sintética (1,932 frases):** frases limpias y claras. Se separa de la real para no inflar los números.

---

## 2. Tabla de los 9 modelos (todos con 26,469 textos y semilla 42)

| # | Modelo | Qué hace en palabras simples | Prueba real (Go F1) | Prueba sintética (F1) | Promedio F1 | ¿Da % directo? |
|---|---|---|---:|---:|---:|:---:|
| 1 | **Conteo simple + Naive Bayes** | Solo cuenta palabras sueltas (modelo base de comparación). | `0.3413` | `0.8833` | `0.6123` | Sí |
| 2 | **TF-IDF (1 palabra) + Reg. Logística** | Cambia el conteo por TF-IDF y balancea las 5 emociones. | `0.5924` | `0.9552` | `0.7738` | Sí |
| 3 | **TF-IDF (1 a 2 palabras) + Reg. Logística** | Suma parejas de palabras seguidas (como `"no quiero"`). | `0.6070` | `0.9691` | `0.7880` | Sí |
| 4 | **TF-IDF (1 a 3 palabras) + Reg. Logística** | Suma tríos de palabras seguidas (como `"da mucha rabia"`). | `0.6102` | `0.9797` | `0.7950` | Sí |
| **5** | **TF-IDF Híbrido (palabras + letras) + Reg. Logística ★** | **Une palabras (1–3) + pedazos de letras (3–5) + `max_df=0.5`, `C=3.0`.** | **`0.6654`** | **`0.9895`** | **`0.8275`** | **Sí** |
| 6 | **TF-IDF (1–3) + Complement Naive Bayes** | Naive Bayes adaptado para clases desbalanceadas. | `0.4305` | `0.9809` | `0.7057` | Sí |
| 7 | **TF-IDF (1–3) + SVM Lineal** | Dibuja la frontera más ancha entre emociones. | `0.6015` | `0.9902` | `0.7958` | No |
| 8 | **TF-IDF Híbrido + SVD (150) + Red Neuronal MLP** | Comprime `84,826` columnas a `150` y las pasa por una red neuronal. | `0.5157` | `0.8821` | `0.6989` | Sí |
| 9 | **TF-IDF Híbrido + Ensamble (LR + CNB)** | Promedia los porcentajes de Regresión Logística y Naive Bayes. | `0.6388` | `0.9915` | `0.8152` | Sí |

---

## 3. Cómo explicar cada algoritmo en voz alta

### 1. Naive Bayes (Modelos 1 y 6)
- **Cómo funciona:** cuenta en qué emoción suele aparecer cada palabra por separado, como meter todas las palabras en una bolsa.
- **Cómo le fue:** en frases muy limpias responde bien, pero en texto real (`0.3413` y `0.4305`) manda demasiados mensajes a *neutral*. Sirvió como **punto de partida** para medir cuánto mejoramos.

### 2. Regresión Logística (Modelos 2, 3, 4 y 5 — el ganador)
- **Cómo funciona:** le da un peso positivo o negativo a cada palabra y pedacito de letras; al final suma todo y con **Softmax** lo convierte en porcentajes que suman 100 %.
- **Por qué mejoró paso a paso:**
  - Con palabras sueltas y balanceo sacó **`0.5924`**.
  - Con parejas (`0.6070`) y tríos de palabras (`0.6102`) captó frases completas.
  - Al sumar **pedazos de 3 a 5 letras** (`char_wb`) dio el gran salto a **`0.6654`** en texto real y **`0.8275`** de promedio, porque conecta verbos como `robaron ↔ robo`.
- **Por qué ganó:** tiene el mejor F1 real, da porcentajes directos para saber cuándo pedir contexto, explica qué palabras pesaron y entrena en segundos.

### 3. SVM Lineal (Modelo 7)
- **Cómo funciona:** busca separar las emociones trazando una pared divisoria.
- **Por qué no ganó:** sacó **`0.6015`** (menor que `0.6654`) y **no da porcentajes directos** sin añadir otro paso extra.

### 4. Reducción de Dimensionalidad (`TruncatedSVD`) + Red Neuronal (`MLPClassifier`) (Modelo 8 — Semana 14)
- **Para qué la probamos:** nuestro modelo genera **`84,826` columnas** (palabras + letras). Una red neuronal clásica necesita menos columnas para correr bien en una laptop normal.
- **Qué hicimos:** usamos **`TruncatedSVD`** (que es el equivalente de **PCA** para tablas de texto llenas de ceros) para comprimir esas **`84,826` columnas en solo `150` columnas resumen**, y las pasamos por una **Red Neuronal MLP** de 64 neuronas.
- **Qué pasó:** en texto real bajó de **`0.6654` a `0.5157`**.
- **Por qué pasó (frase clave para decir en clase):** *"Al comprimir 84 mil palabras en solo 150 columnas resumen se pierden palabras emocionales poco frecuentes como «robaron» o «vacío». Probamos la red neuronal, la medimos con los mismos datos y nos quedamos con el que demostró mejores números."*

### 5. Ensamble (`VotingClassifier`) (Modelo 9 — Semana 12)
- **Cómo funciona:** une las votaciones de Regresión Logística y Complement Naive Bayes.
- **Por qué no ganó:** en texto real bajó a **`0.6388`** (frente a `0.6654` de Regresión Logística sola) y hace más difícil explicar palabra por palabra.

---

## 4. Cómo afinamos al ganador (sin adivinar)

En `p08_experimentos.py` probamos 5 combinaciones cambiando `max_df` (ignorar palabras demasiado repetidas) y `C` (cuánto se ajusta al entrenamiento):

| `max_df` | `C` | GoEmotions F1 | Sintética F1 | Promedio F1 |
|---:|---:|---:|---:|---:|
| `1.0` | `1.0` | `0.6585` | `0.9871` | `0.8228` |
| `0.5` | `1.0` | `0.6570` | `0.9871` | `0.8220` |
| `0.5` | `2.0` | `0.6636` | `0.9899` | `0.8267` |
| **`0.5`** | **`3.0`** | **`0.6654`** | **`0.9895`** | **`0.8275`** |
| `0.5` | `5.0` | `0.6601` | `0.9897` | `0.8249` |

Elegimos **`max_df = 0.5`** y **`C = 3.0`** porque logra el mejor equilibrio en texto real (`0.6654`) y promedio (`0.8275`) sin memorizar de más.
