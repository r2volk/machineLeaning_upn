# 04. Cómo aprende frases nuevas y cómo recuerda 1 turno

El chatbot tiene dos funciones pensadas para hacerlo práctico en la vida real: puede **aprender frases que tú le enseñes** (y deshacerlo cuando quieras) y puede **recordar tu mensaje anterior** cuando te pide más contexto.

---

## 1. Pestaña «Aprendizaje» (enseñar frases nuevas sin malograr el modelo)

Imagina que escribes un modismo peruano: **"No esperaba aprobar pero saqué veinte"**. Como `"saqué veinte"` nunca salió en los datos originales, el modelo base dice `Neutral`.

Así lo corriges en **4 pasos seguros**:

1. **Guardas el ejemplo (siempre voluntario):**
   - Escribes la frase (o pulsas *"Usar mi último mensaje del chat"*), eliges **Alegría** y guardas en `datos/correcciones_usuario.json`.
   - **Dos bloqueos de seguridad (`p09_reentrenamiento.py`):** si la frase tiene riesgo de autolesión **o** ya está en validación, el programa la rechaza para cuidar al usuario y no hacer trampa.
2. **Entrenar nueva versión:**
   - Junta las `26,469` frases originales con tus ejemplos nuevos (cada ejemplo nuevo se repite 3 veces para que supere el filtro mínimo `min_df = 2`).
   - Entrena todo en ~6 segundos, guarda el resultado aparte en `modelos/modelo_activo.joblib` y **nunca toca el `modelo_base.joblib` original**.
3. **Ver el impacto real:**
   - Apenas termina, compara el **modelo base vs. el modelo activo** en validación y te muestra si subió o bajó algún número.
4. **Volver atrás con 1 clic:**
   - Pulsando *"Restaurar modelo base"*, el sistema regresa exactamente al modelo original.

> **¿Por qué reentrena todo en ~6 segundos en vez de actualizar solo con la frase nueva?**
> Porque si solo actualizamos con 1 frase, el modelo no sabría agregar palabras nuevas que antes no existían (como `"veinte"`) y podría olvidar lo que ya sabía. Volver a entrenar todo junto tarda segundos y mantiene estable el sistema.

---

## 2. Pedir contexto y memoria de 1 turno (Semana 03)

En la Semana 03 vimos que un **agente inteligente** no mira solo un instante aislado, sino la **secuencia de mensajes** recientes.

1. **Cuándo pide contexto:**
   - Si la emoción ganadora tiene **menos de 45 %** de seguridad **o** le saca **menos de 10 puntos** a la segunda emoción, el chatbot prefiere **no adivinar** y te pide un poco más de contexto.
2. **Qué recuerda (1 turno):**
   - Mientras espera tu aclaración, la página guarda temporalmente tu mensaje anterior (`contexto_previo`).
   - Ejemplo: primero dices *"No me fue del todo bien en el examen"* (pide contexto). Luego respondes *"reprobé el curso y me siento sin fuerzas"*. El sistema junta ambos mensajes (después de revisar seguridad) y clasifica **Tristeza** con **`95.6 %`**, avisando **`· con contexto previo`**.
3. **Cuándo olvida:**
   - Apenas logra clasificar la emoción (o al iniciar un chat nuevo), borra ese recuerdo para no mezclar conversaciones pasadas.
