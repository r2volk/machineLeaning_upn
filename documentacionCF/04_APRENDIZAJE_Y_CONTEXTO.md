# 04. Aprendizaje con correcciones y memoria de 1 turno

El chatbot tiene dos funciones interactivas que lo hacen mucho más útil que un clasificador estático: puede **aprender frases nuevas que tú le enseñes** y puede **recordar tu mensaje anterior cuando pide contexto**. Aquí se explica cómo funcionan en palabras sencillas.

---

## 1. ¿Cómo funciona la pestaña «Aprendizaje»?

Imagina que pruebas una frase con un modismo peruano que el modelo nunca vio, por ejemplo: **"No esperaba aprobar pero saqué veinte"**. Como `"saqué veinte"` no estaba en el entrenamiento, el modelo base responde `neutral`.

En la pestaña **Aprendizaje** puedes enseñársela en 4 pasos controlados:

1. **Guardar el ejemplo (siempre voluntario):**
   - Escribes la frase (o pulsas *"Usar mi último mensaje del chat"*), eliges la emoción correcta (`Alegría`) y pulsas guardar.
   - Nada se guarda en secreto ni de forma automática. Las correcciones se guardan en un archivo local (`datos/correcciones_usuario.json`).
   - **Dos candados de seguridad en `p09_reentrenamiento.py`:**
     - Si la frase tiene señales de riesgo de suicidio o autolesión, **se bloquea** y nunca se guarda para entrenar.
     - Si la frase ya existe en `validacion_go.csv` o `validacion_sintetica.csv`, **también se bloquea** para no hacer trampa ni contaminar las métricas.

2. **Entrenar nueva versión:**
   - Al pulsar *"Entrenar nueva versión"*, el programa junta el entrenamiento original (`26,469` frases) con tus correcciones locales.
   - Cada corrección se incluye **3 veces** (`PESO_REPETICION_CORRECCION = 3`) para que las palabras nuevas superen el filtro mínimo (`min_df = 2`) sin desequilibrar el resto del modelo.
   - Reentrena el pipeline completo desde cero en unos segundos, guarda el resultado en `modelos/modelo_activo.joblib` (como `reentrenado_v1`, `reentrenado_v2`, etc.) y **nunca toca `modelo_base.joblib`**.

3. **Medir el cambio real en validación:**
   - Apenas termina de entrenar, vuelve a medir el modelo en `validacion_go.csv` y `validacion_sintetica.csv` y muestra una tabla comparando el **modelo base vs. el modelo activo**. Así ves con total transparencia si mejoró o si cambió algo.

4. **Reversible en un clic:**
   - Puedes borrar ejemplos individuales, borrar todos o pulsar *"Restaurar modelo base"* para volver exactamente al estado inicial.

### ¿Por qué reentrenamos el modelo completo y no usamos aprendizaje incremental (`partial_fit`)?
Porque si una frase trae una palabra nueva que no existía antes (como `"veinte"`), un vocabulario fijo no podría agregarla. Además, actualizar un modelo solo con 1 o 2 frases puede hacer que olvide lo que ya sabía. Reentrenar todo junto tarda solo unos segundos, es 100 % reproducible y mantiene estable el modelo.

---

## 2. ¿Cómo funciona «Pedir contexto» y la memoria de 1 turno?

En la Semana 03 del curso vimos que un **agente inteligente** toma decisiones a partir de su **secuencia de percepciones**, no solo de una foto aislada.

Nuestro chatbot aplica esa idea de forma sencilla y transparente:

1. **Cuándo pide contexto (abstención):**
   - Si la probabilidad de la emoción ganadora es **menor a 45 %** (`0.45`) **o** la diferencia frente a la segunda emoción es **menor a 10 puntos** (`0.10`), el chatbot prefiere no adivinar y responde pidiéndote un poco más de contexto.
2. **Qué recuerda (memoria de 1 turno):**
   - Cuando el bot entra en estado `pedir_contexto`, la interfaz guarda tu mensaje anterior en `pendingContext` durante **un solo turno**.
   - Cuando tú le respondes con una aclaración (por ejemplo, primero dijiste *"No me fue del todo bien en el examen"* y luego aclaras *"reprobé el curso y me siento sin fuerzas"*), la página envía ambos textos a `/api/chat`: `mensaje` y `contexto_previo`.
   - En `p07_inferencia.py`, primero revisa seguridad sobre tu mensaje nuevo; si es seguro, une ambos textos y los clasifica juntos, marcando en pantalla **`· con contexto previo`**.
3. **Cuándo se borra esa memoria:**
   - Apenas el chatbot logra clasificar la emoción (o cuando inicias una nueva conversación), `pendingContext` se limpia para que mensajes antiguos no interfieran con temas nuevos.
