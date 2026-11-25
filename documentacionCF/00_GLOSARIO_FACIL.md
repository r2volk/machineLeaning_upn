# 00. Glosario fácil (palabras técnicas explicadas en español normal)

Si lees algún término raro en los otros documentos o en el código, búscalo aquí. La idea es que lo entiendas en 10 segundos y lo puedas decir con tus propias palabras.

| Término técnico | Qué significa en palabras normales | Ejemplo fácil |
|---|---|---|
| **TF-IDF** | Forma de convertir palabras en números dándole **más puntos a las palabras especiales** y menos puntos a palabras repetidas en todos lados. | En *"Hoy me robaron"*, `"hoy"` aparece mil veces y vale poco; `"robaron"` vale mucho para **Enojo**. |
| **Bigramas y Trigramas** *(n-gramas)* | Mirar parejas de 2 palabras (**bigramas**) o tríos de 3 palabras (**trigramas**) juntas, en lugar de palabras sueltas. | `"da mucha rabia"` junto dice mucho más que `"da"`, `"mucha"` y `"rabia"` por separado. |
| **Subpalabras (`char_wb`)** | Partir cada palabra en **pedacitos de 3 a 5 letras** para reconocer palabras parecidas aunque cambie el final del verbo. | Si el modelo sabe `"robo"`, entiende `"robaron"` o `"robando"` por el pedazo `"rob"`. |
| **Alcance de negación (`neg_`)** | Pegar la etiqueta `neg_` a las 2 palabras siguientes después de `"no"`, `"nunca"` o `"tampoco"`. | `"No me siento feliz"` se convierte en `no me neg_siento neg_feliz` para que no marque Alegría. |
| **Regresión Logística** *(el ganador)* | Algoritmo que suma el puntaje de cada palabra o pedazo de letras y decide cuánto pesa para cada emoción. | Suma puntos a favor de Alegría, Tristeza, Enojo, Ansiedad o Neutral. |
| **Softmax** | Fórmula final que convierte esos puntajes en **porcentajes que suman 100 %**. | Por ejemplo: Tristeza `84 %`, Alegría `8 %`, resto `8 %`. |
| **Exactitud** (*Accuracy*) | Porcentaje total de mensajes acertados sobre 100. **Ojo:** si casi todo es neutral, puede engañar. | Si 84 de 100 textos son neutrales, responder siempre "neutral" daría 84 % aunque no sirva. |
| **Macro-F1** | Promedio justo del acierto (**F1**) de las **5 emociones por igual** (cada emoción vale 20 %). | Obliga al modelo a hacerlo bien también en Alegría, Tristeza, Enojo y Ansiedad. |
| **Matriz de confusión** | Tabla cuadrada que muestra cuántas veces acertó en cada emoción (diagonal) y con cuál otra la confundió. | Muestra si confundió algunos mensajes de Enojo con Neutral. |
| **Reducción de dimensionalidad (`TruncatedSVD`)** | Comprimir una tabla gigante de **84,826 columnas** de palabras en solo **150 columnas resumen** (es la versión de **PCA** para texto). | Como resumir un libro de 500 páginas en 2 hojas: se pierden palabras poco frecuentes. |
| **Red Neuronal MLP** (`MLPClassifier`) | Red de neuronas artificiales vista en la Semana 14. Recibió las 150 columnas comprimidas y usó 64 neuronas ocultas. | Bajó de `0.6654` a `0.5157` porque comprimir borró pistas clave. |
| **SVM Lineal** (`LinearSVC`) | Algoritmo que dibuja una frontera separando emociones, pero **no da porcentajes directos**. | Sacó `0.6015` (menos que `0.6654` del ganador). |
| **Ensamble (`VotingClassifier`)** | Juntar dos modelos distintos para que voten juntos y promediar sus porcentajes. | Bajó a `0.6388` en texto real y hace más difícil ver qué palabra pesó. |
| **Balanceo (`class_weight="balanced"`)** | Avisarle al modelo durante el entrenamiento que cuide más las emociones con menos ejemplos (para que Neutral no tape todo). | Sin esto, casi todo se iría a Neutral porque es mayoría. |
| **Fuga de datos** (*Data leakage*) | Hacer trampa sin querer: que una frase de examen esté repetida dentro del material de estudio. | Revisamos que haya **0 frases repetidas** entre entrenar y probar. |
| **Abstención (pedir contexto)** | Que el chatbot prefiera preguntar en lugar de adivinar cuando está poco seguro o dos emociones empatan. | Pide contexto si la ganadora tiene `< 45 %` o ventaja `< 10` puntos. |
| **Semilla (`SEMILLA = 42`)** | Número fijo para que al volver a entrenar salga **exactamente el mismo resultado** siempre. | Garantiza que en clase dé los mismos números que en el informe. |
| **`max_df = 0.5` y `C = 3.0`** | Ajustes del modelo: `max_df=0.5` ignora palabras que salen en más de la mitad de textos; `C=3.0` regula cuánto memoriza. | Se eligieron comparando opciones en validación, no al azar. |
