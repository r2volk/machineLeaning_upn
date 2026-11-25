# 01. Explicación sencilla del proyecto (de cero y sin enredos)

Esta guía explica **cómo funciona todo el chatbot** en palabras normales. Si alguna palabra te suena rara, revisa primero **[00_GLOSARIO_FACIL.md](00_GLOSARIO_FACIL.md)**.

---

## 1. ¿Qué hace este proyecto?

Es un **chatbot que corre 100 % en tu computadora** (no usa internet ni APIs externas). Lee una frase en español y detecta señales de **5 emociones**:

1. **Alegría**
2. **Tristeza**
3. **Enojo**
4. **Ansiedad / nerviosismo**
5. **Neutral** (frases informativas o sin emoción clara)

---

## 2. ¿Qué pasa desde que escribes un mensaje hasta que responde?

Imagina que escribes: **"Ayer me robaron el teléfono"**. El programa sigue **4 pasos**:

### Paso 1: Revisión de seguridad (reglas fijas)
Antes de pasar la frase al modelo de Machine Learning, `p04_seguridad.py` revisa si hay señales de peligro (como *"no quiero vivir"* o *"hacerme daño"*). También sabe ignorar chistes o expresiones comunes como *"me muero de risa"* o *"me muero de hambre"*.
- Como *"Ayer me robaron el teléfono"* no es autolesión, sigue al paso 2.

### Paso 2: Limpiar el texto y cuidar el «no»
Para la computadora, `"Teléfono"` y `"telefono"` serían palabras distintas. Por eso `p03_preprocesamiento.py`:
- Pasa todo a minúsculas y quita las tildes, **pero conserva la `ñ`** (porque *"año"* no es *"ano"*).
- Acorta letras estiradas (`"nooooo"` → `"no"`).
- **No borra la palabra `"no"`**. Cuando ve `"no"`, `"nunca"` o `"tampoco"`, pega `neg_` a las 2 palabras principales siguientes (sin gastar turno en `"me"`, `"te"`, `"se"`).
  - Ejemplo: `"No me siento feliz"` → `"no me neg_siento neg_feliz"` (así el modelo no lee `"feliz"` suelto).
  - Ejemplo: `"No estoy enojado, solo cansado"` → `"no neg_estoy neg_enojado solo cansado"` (corta en `"solo"` o en una coma).

### Paso 3: Convertir palabras en números y sacar porcentajes
Una computadora solo hace cuentas con números:
- Primero convertimos la frase en números usando **TF-IDF**: da **mucho peso a palabras clave** (`"robaron"`) y **poco peso a palabras comunes** (`"el"`, `"de"`).
- Miramos dos cosas a la vez:
  1. **Palabras sueltas y grupos de hasta 3 palabras** (por ejemplo `"me robaron"` o `"neg_feliz"`).
  2. **Pedacitos de 3 a 5 letras** (`char_wb`). Sirve para que si el modelo vio `"robo"`, también entienda `"robaron"` porque comparten `"rob"`.
- Luego, nuestro ganador (**Regresión Logística**) suma los puntos de cada palabra o pedazo y los transforma en **porcentajes que suman 100 %** (función **Softmax**).

### Paso 4: Responder o pedir más contexto
El modelo **no adivina cuando duda**:
- Si la emoción ganadora tiene **al menos 45 %** y le gana a la segunda por **al menos 10 puntos**, responde con esa emoción y muestra qué palabras influyeron.
- Si está inseguro (por ejemplo, `"No me fue del todo bien en el examen"`), **pide más contexto**. Cuando tú le respondes en el siguiente mensaje, **recuerda tu frase anterior durante 1 turno** y junta ambas para responder mejor.

---

## 3. ¿Qué librerías de Python usamos?

Si la profesora pregunta *"¿qué librerías usaron?"*:

| Librería | Para qué sirve en palabras simples | Dónde se usa |
|---|---|---|
| **scikit-learn** | La caja de herramientas de Machine Learning (TF-IDF, los 9 modelos, `TruncatedSVD`, métricas). | `p05`, `p06`, `p08` |
| **Flask** | Crea la página web local en tu navegador (`http://127.0.0.1:5000`). | `app.py` |
| **NumPy** | Hace cuentas rápidas con listas de números y porcentajes. | `p06`, `p07` |
| **SciPy** | Maneja por dentro las tablas gigantes llenas de ceros que genera TF-IDF. | Dentro de `scikit-learn` |
| **joblib** | Guarda el modelo ya entrenado en disco para cargarlo en milisegundos. | `app.py`, `chat_terminal.py`, `p08`–`p10` |

Y usamos herramientas que ya trae Python por defecto: `csv` (leer tablas), `re` (buscar texto), `unicodedata` (quitar tildes), `json` (guardar reportes) y `unittest` (11 pruebas automáticas).

---

## 4. Dos formas de ver el programa funcionando

1. **En consola (`chat_terminal.py`):** demo rápida para mostrar en clase el chatbot puro (escribe tu mensaje, muestra su respuesta, la emoción detectada, barras y palabras clave; escribes `salir` para terminar).
2. **En la web (`app.py`):** página en Flask que incluye ese mismo chat más visual, junto a las pestañas **Laboratorio** (gráficos y tabla de 9 modelos) y **Aprendizaje** (enseñar frases nuevas de forma reversible).

---

## 5. Mapa rápido de los archivos del proyecto

- **`chat_terminal.py`**: chat simple en terminal para la primera parte de tu exposición.
- **`app.py`**: servidor web Flask que conecta el navegador con el modelo.
- **`templates/index.html`**: toda la interfaz web (*Conversación*, *Laboratorio* y *Aprendizaje*).
- **`entrenamiento/p01_config.py`**: rutas, las 5 emociones, semilla fija (`42`) y umbrales.
- **`entrenamiento/p02_datos.py`**: carga los CSV y verifica que haya **0 frases repetidas** entre entrenar y probar.
- **`entrenamiento/p02b_generar_corpus_implicito.py`**: crea las 2,780 frases de situaciones cotidianas sin emoción explícita.
- **`entrenamiento/p03_preprocesamiento.py`**: limpia el texto y agrega `neg_` tras negaciones.
- **`entrenamiento/p04_seguridad.py`**: filtro prioritario de crisis (Línea 113, Opción 5).
- **`entrenamiento/p05_modelos.py`**: define los **9 modelos** que comparamos.
- **`entrenamiento/p06_metricas.py`**: mide exactitud, Macro-F1 y matriz de confusión.
- **`entrenamiento/p07_inferencia.py`**: lógica en vivo cuando entra un mensaje.
- **`entrenamiento/p08_experimentos.py`**: entrena los 9 modelos, prueba ajustes y guarda el ganador.
- **`entrenamiento/p09_reentrenamiento.py`**: guarda frases nuevas del usuario, reentrena y permite volver al modelo base.
- **`entrenamiento/p10_evaluar_test.py`**: evalúa el archivo de prueba final **una sola vez**.
