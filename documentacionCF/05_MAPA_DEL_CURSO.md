# 05. Mapa del curso (qué tema de clase está en cada archivo)

Esta tabla te sirve para mostrarle a la profesora cómo cada parte del proyecto aplica temas reales del curso.

| Semana | Tema visto en clase (en palabras simples) | Cómo lo aplicamos en el chatbot | Archivo |
|---|---|---|---|
| **Sem. 02** | Sistemas automáticos con cuidado humano en situaciones críticas | El chatbot funciona solo en tu PC, pero si detecta riesgo de autolesión detiene el modelo y muestra ayuda humana (**Línea 113 Salud, Opción 5**). | `p04_seguridad.py` |
| **Sem. 03** | Agentes inteligentes (entrada, decisión, respuesta y memoria) | Entrada = tu mensaje; decisión = seguridad + modelo; memoria = recuerda 1 turno si pide contexto (`contexto_previo`); aprendizaje = pestaña *Aprendizaje*. | `p07_inferencia.py`, `app.py`, `p09_reentrenamiento.py` |
| **Sem. 04** | Desarrollo ordenado por etapas | Problema → datos y limpieza (`p02`–`p03`) → comparar modelos (`p05`–`p08`) → evaluar (`p06`, `p10`) → consola y web (`chat_terminal.py`, `app.py`). | Todo `entrenamiento/` |
| **Sem. 05** | Porcentajes que suman 100 % y semilla fija (`42`) | Las 5 emociones siempre suman 100 %; si hay duda pide contexto, y la semilla `42` garantiza que siempre dé los mismos resultados. | `p01_config.py`, `p07_inferencia.py` |
| **Sem. 07** | Datos sintéticos y reglas fijas | Combinamos datos reales traducidos (`go_emotions_es_mt`) con frases sintéticas y 2,780 cotidianas, siempre etiquetando su origen. La seguridad usa reglas fijas. | `p02_datos.py`, `p02b_generar_corpus_implicito.py`, `p04_seguridad.py` |
| **Sem. 09** | Reglas + ML y limpieza de texto (`unicodedata`, `re`) | Reglas fijas para crisis + ML para emociones. Limpiamos tildes y signos cuidando la `ñ`, y **no borramos el `"no"`** porque cambia la emoción (`neg_`). | `p03_preprocesamiento.py`, `p04_seguridad.py` |
| **Sem. 11** | Mitigar sesgos, aprender de ejemplos nuevos (como filtro de spam) y métrica F1 | 0 textos repetidos, corpus equilibrado, `class_weight="balanced"`, regularización `C=3.0`, pedir contexto cuando duda y métrica **Macro-F1**. | `p02_datos.py`, `p05_modelos.py`, `p06_metricas.py`, `p09_reentrenamiento.py` |
| **Sem. 12** | Comparar algoritmos, dividir Train / Val / Test y Matriz de confusión | Comparamos **9 modelos** (Naive Bayes, Regresión Logística, SVM, Red MLP, Ensamble) sin mezclar entrenamiento con validación ni test final. | `p05_modelos.py`, `p08_experimentos.py`, `p10_evaluar_test.py` |
| **Sem. 14** | Reducción de dimensionalidad y Red Neuronal MLP | Probamos comprimir `84,826` columnas a `150` con `TruncatedSVD` (PCA para texto) y pasarlas por una red neuronal `MLPClassifier` de 64 neuronas. | `p05_modelos.py`, `p08_experimentos.py` |
