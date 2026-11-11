# Relación verificable con los apuntes

Se revisaron los apuntes disponibles en `/Users/ricardo/Desktop/machineLearning`; las secciones pertinentes a este proyecto se leyeron antes de implementar. No se copiaron datasets ni resultados de las diapositivas.

| Archivo del curso y sección | Aplicación concreta |
|---|---|
| `semana-04-metodologia-desarrollo.md`, §2 | Problema → datos → selección → entrenamiento → implementación; se plasma en el protocolo y módulos ordenados. |
| `semana-07-procesamiento-senales-digitales.md`, §2.2 y §3 | Precedente de datos sintéticos y sistemas basados en reglas. Las reglas de precaución se separan del clasificador aprendido. |
| `semana-09-representacion-conocimiento-razonamiento.md`, §1.9, §2.4 y §4 | Reglas de producción, NLP clásica, minúsculas, tildes y tokenización. Se conservan negaciones; no se usa lematización preentrenada. |
| `semana-11-machine-learning-datasets-sesgos.md`, §2, §4, §5, §7 y §8 | Aprendizaje supervisado, filas/atributos, sesgos, bolsa de palabras + logística, `.fit`, `.predict`, accuracy, precision, recall y F1. |
| `semana-12-modelos-de-regresion.md`, §2.3, §4, §5, §8.5, §8.6 y §9 | Naive Bayes mencionado entre clasificadores; train/validación/test; logística multinomial, softmax, matriz y flujo de scikit-learn. |

Los apuntes de señales e imágenes no justifican incorporar procesamiento de audio o visión: no aportaría a una tarea de texto. No se usan redes neuronales ni embeddings pese a estar mencionados en clase. Esta es una decisión de alcance y explicabilidad.

## Ampliaciones justificadas

| Ampliación | Qué añade y cómo explicarla |
|---|---|
| TF-IDF | La bolsa de palabras del curso da conteos; TF-IDF pondera la frecuencia en un mensaje y reduce el peso de términos comunes en todo train. Con `sublinear_tf=True`, tf=1+log(conteo); idf=log((1+n)/(1+df))+1; el vector se normaliza a longitud L2=1. |
| Bigramas | Parejas adyacentes como `no estoy`. Preservan algo de orden local, sin pretender comprensión semántica. Se comparan con unigramas. |
| Pipeline | Une transformación y clasificación para que el vocabulario y los pesos de IDF se ajusten solo a train y se reutilicen sin cambios. Previene fuga de datos. |
| Grupos de paráfrasis | Las frases relacionadas no son observaciones independientes. Dividir por escenario evita que una variante enseñe directamente la otra en test. |
| Referencia constante | `DummyClassifier` establece un mínimo: comprobar si aprender palabras aporta algo. Con empate, la clase elegida sigue el orden interno determinista. |
| Naive Bayes multinomial | Está mencionado en semana 12, pero su implementación y suavizado no se desarrollan allí. Estima asociaciones palabra/clase bajo independencia condicional; alpha evita probabilidades cero. Aquí TF-IDF es una variante práctica no equivalente al modelo generativo de conteos exactos. |
| Abstención | Si las puntuaciones son cercanas o débiles, se pide contexto. Cobertura informa cuántos mensajes sí se etiquetan. No es un mecanismo clínico. |
| Explicación contrastiva | Para dos clases, sumar `x_j*(w_clase,j-w_alternativa,j)` más la diferencia de interceptos produce la diferencia de logits. La interfaz muestra hasta cinco aportes positivos, no una descomposición completa. Para NB usa log-probabilidades de características. |
| Bootstrap por escenario | Reutiliza predicciones finales, remuestrea grupos y estima variabilidad aproximada. No vuelve a entrenar ni ofrece garantías poblacionales con datos sintéticos. |
| Flask, CSP, pytest | Ingeniería de la aplicación y verificación, no técnicas de ML. Mantienen ejecución simple y evitan regresiones de seguridad. |

Conservar `no` en el texto evita perder información, pero **no garantiza entender la negación**. El modelo puede aprender mal o no tener ejemplos suficientes. Normalizar sin un modelo preentrenado facilita reproducibilidad a cambio de perder tildes, emojis y distinciones como ñ/n.

Fuentes técnicas de consulta: [extracción de características de scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html), [errores habituales y fuga de datos](https://scikit-learn.org/stable/common_pitfalls.html), [Flask](https://flask.palletsprojects.com/en/stable/quickstart/). La documentación de la versión instalada está reflejada por los parámetros usados y las pruebas ejecutadas; `stable` puede avanzar a una versión posterior.
