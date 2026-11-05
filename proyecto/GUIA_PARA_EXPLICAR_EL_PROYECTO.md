# De los datos al pronóstico en la web
## Guía para entender y explicar nuestro proyecto de Machine Learning

**Estado documentado: 5 de noviembre de 2026.**

**Proyecto:** prototipo de pronóstico de ventas diarias con datos de Kaggle y presentación de sus resultados en la web existente de Kapaq.

> **Cómo leer esta guía:** primero lee las secciones 1 a 12 en orden. Cada paso explica **qué hicimos, para qué y por qué**. Las secciones finales sirven como guion, demostración y respuestas para la profesora. No necesitas memorizar código ni todas las cifras.
>
> **La distinción principal:** ya implementamos una demostración con pronósticos de Kaggle. Todavía no tenemos un modelo entrenado y validado con ventas propias de Kapaq. La web muestra predicciones calculadas previamente; no ejecuta el modelo al cambiar un filtro.

---

## 1. ¿Qué problema escogimos y qué queremos conseguir?

Nuestro grupo ya tenía una página web de Kapaq, desarrollada en ciclos anteriores. Para este curso decidimos agregar un componente de Machine Learning, no volver a construir el sistema comercial.

La pregunta escogida fue:

> **¿Cuántas unidades de un artículo podrían venderse en una fecha futura, tomando como referencia su historial de ventas?**

Por ejemplo, saber aproximadamente cuántas unidades podrían venderse ayudaría, en una implementación real, a planificar el inventario. Pero **predecir ventas no es lo mismo que decidir cuánto comprar**: para una recomendación de compra también harían falta stock actual, plazos de reposición, reglas del negocio y otros datos que no incorporamos en esta demo.

### Dos objetivos que no debemos confundir

| Objetivo | Estado |
|---|---|
| Desarrollar y evaluar un método de pronóstico con datos públicos, y mostrar sus resultados en la web | Implementado como demostración |
| Pronosticar productos reales de Kapaq con su propio historial de ventas | Objetivo posterior; no implementado ni validado |

**Para decirlo en clase:**

> “Partimos de una web existente y añadimos una demostración de pronóstico de ventas. Primero desarrollamos y evaluamos el método con datos públicos de Kaggle; su adaptación a ventas propias de Kapaq aún está pendiente.”

## 2. ¿Por qué usamos Kaggle y no la base de Kapaq?

Un modelo necesita ejemplos históricos. Con un único día no puede aprender cómo cambian las ventas entre semanas o meses.

En la revisión agregada que hicimos de la base de Kapaq se encontró una sola línea de pedido entregado, para un solo producto y un solo día. Además, el sistema contiene pedidos de demostración creados por una migración. **Esa revisión no encontró historial propio suficiente para entrenar y evaluar el pronóstico.** Es una descripción de lo encontrado en esa revisión, no una garantía de que la base siga igual después de nuevos registros.

Por eso usamos el dataset público **Store Item Demand Forecasting** de Kaggle para desarrollar el experimento.

**¿Para qué?** Para disponer de muchos ejemplos con fechas y cantidades conocidas y poder medir errores.

**¿Por qué aclararlo?** Porque el comportamiento de las tiendas de ese dataset no demuestra cómo se comportarán los clientes de Kapaq. No cambiamos los nombres de los artículos para fingir que eran productos del minimarket.

## 3. ¿Qué significa una fila del archivo?

El archivo `datasets/train.csv` es una tabla de texto, parecida a una hoja de cálculo:

```csv
date,store,item,sales
2013-01-01,1,1,13
2013-01-02,1,1,11
2013-01-03,1,1,14
```

La primera fila de datos significa:

> “Para el 1 de enero de 2013, la tienda identificada como 1 y el artículo identificado como 1 tienen registradas 13 unidades vendidas en el dataset.”

| Columna | Traducción | Función |
|---|---|---|
| `date` | Fecha | Permite ordenar el tiempo y construir calendario |
| `store` | Tienda | Distingue una tienda de otra |
| `item` | Artículo | Distingue un artículo de otro |
| `sales` | Unidades vendidas | Es la respuesta que queremos estimar |

### ¿El artículo 1 es arroz o papel higiénico?

**No lo sabemos a partir de estas columnas.** Son identificadores, no nombres. Tampoco significa que el artículo 2 sea el doble del artículo 1.

El modelo no necesita comprender qué es una escoba para estudiar cómo variaron las ventas del artículo identificado como 1. Puede aprender diferencias estadísticas entre artículos si existen en sus historiales. **No aprende el uso ni el significado de los productos.**

### ¿No podrían ser números generados con un script?

Sí, es posible generar tablas así con un script. Ver estas filas por sí solas no demuestra su procedencia. Nosotros las describimos como datos públicos del desafío de Kaggle; no afirmamos que comprobamos su origen comercial real.

La cuestión para el experimento es si hay patrones aprovechables y si el modelo mejora una regla sencilla en fechas que no utilizó para entrenarse. Una secuencia completamente aleatoria e independiente no se vuelve predecible solo por aplicarle Machine Learning.

### Tamaño de los datos utilizados

- `train.csv`: **913 000 filas**, del **2013-01-01 al 2017-12-31**.
- **10 tiendas y 50 artículos**: 500 combinaciones de tienda y artículo.
- `test.csv`: **45 000 filas**, del **2018-01-01 al 2018-03-31**, sin las ventas que tenemos que estimar.
- `sample_submission.csv`: ejemplo del formato de entrega `id,sales`; **no contiene respuestas verificadas para medir nuestro error**.

“Futuro” aquí significa posterior a diciembre de 2017 dentro del desafío. **No estamos pronosticando las fechas actuales de Kapaq.**

## 4. ¿Qué hicimos con Python? El recorrido completo

```text
DATOS DE KAGGLE: ventas históricas de 2013 a 2017
                     │
                     ▼
1. Leer, revisar y ordenar las filas
                     │
                     ▼
2. Construir entradas: calendario + historial anterior
                     │
                     ▼
3. Separar pasado para entrenar y fechas posteriores para validar
                     │
                     ▼
4. Entrenar regresión lineal y Random Forest
   Compararlos también con una referencia semanal
                     │
                     ▼
5. Medir los errores en tres períodos históricos
   Evaluación diaria y evaluación recursiva de 90 días
                     │
                     ▼
6. Elegir el candidato con menor error promedio: Random Forest
                     │
                     ▼
7. Entrenarlo con las filas utilizables del historial completo
   y pronosticar las fechas de test.csv (enero–marzo de 2018)
                     │
                     ▼
8. Guardar predicciones_kaggle.csv
                     │
                     ▼
9. Unir cada predicción con su fecha, tienda y artículo → JSON
                     │
                     ▼
10. La pantalla web lee el JSON y muestra filtros, gráfico y tabla
```

**Python es la herramienta para trabajar con los datos y los modelos.** No es el modelo por sí mismo. Las bibliotecas utilizadas incluyen pandas para tablas, NumPy para cálculos, scikit-learn para modelos y métricas, y Matplotlib para gráficos del análisis.

## 5. Paso 1: leer, revisar y ordenar

**Qué hicimos:** cargamos el CSV, revisamos columnas, fechas, valores faltantes y duplicados por fecha/tienda/artículo. La inspección inicial reportó que no había valores vacíos ni duplicados con esa clave. Ordenamos las filas por tienda, artículo y fecha.

**Para qué:** asegurar que las entradas tienen una estructura utilizable y que el historial esté en orden.

**Por qué importa:** si las fechas estuvieran desordenadas, “la venta anterior” podría no ser realmente anterior. Y si mezcláramos artículos, usaríamos ventas de un producto para calcular el historial de otro.

**Archivo principal:** `src/pronostico/datos.py` (rutas de esta guía relativas a `proyecto/`, salvo que se indique otra raíz).

## 6. Paso 2: convertir el historial en información útil

No entrenamos solamente con la fecha escrita como texto. A partir de las filas construimos nuevas columnas, llamadas **variables de entrada** o **características**.

| Entrada | Qué significa | Para qué puede servir |
|---|---|---|
| Tienda y artículo | Identifican la combinación | Distinguir comportamientos entre tiendas/artículos |
| Día de la semana | Lunes, martes, etc., representados numéricamente | Estudiar posibles patrones semanales |
| Mes | Mes de la fecha que queremos estimar | Estudiar posibles diferencias entre meses |
| `day_index` | Días transcurridos desde una fecha base fija | Representar la posición en el tiempo |
| `lag_7` | Venta registrada siete días antes | Aprovechar historial semanal |
| `lag_14` | Venta registrada catorce días antes | Incorporar otra referencia semanal |
| `media_7_dias` | Promedio de los siete días anteriores | Resumir el nivel de ventas reciente |
| `media_28_dias` | Promedio de los 28 días anteriores | Resumir un período más amplio |

### Ejemplo didáctico, no resultado del modelo

Si para un artículo las ventas de los siete días anteriores fueron:

```text
10, 12, 8, 14, 11, 13, 9
```

Su promedio es `77 / 7 = 11`. El modelo recibe ese 11 como una entrada más, no como una respuesta obligatoria.

Cada historial se calcula **por tienda y artículo**, sin mezclarlos. Los promedios excluyen la venta del día que queremos predecir mediante `shift(1)`: desplazar el historial un día hacia atrás antes de promediar. Se eliminan del entrenamiento las primeras filas que aún no tienen las 28 observaciones previas necesarias.

> **Idea clave: no mirar la respuesta.** Para predecir la venta de hoy no podemos utilizar la venta de hoy como entrada. Eso sería fuga de información: dar al modelo información que no existiría al emitir el pronóstico.

En este dataset diario y completo, desplazar siete filas dentro de un historial equivale a siete días. Para datos de Kapaq habría que verificar primero los días faltantes; esa equivalencia no se puede asumir sin revisar.

### ¿Cómo tratamos los identificadores numéricos?

Usamos **one-hot encoding** para tienda y artículo. En términos sencillos: transformamos cada identidad en indicadores de “pertenece/no pertenece”. Así evitamos tratarlos como cantidades o como un orden de mayor a menor. La transformación forma parte del proceso que se ajusta con el entrenamiento.

**Archivos:** `src/pronostico/variables.py` y `src/pronostico/modelos.py`.

## 7. Paso 3: separar las entradas de la respuesta y entrenar

En los archivos verás dos letras:

```text
X = entradas: calendario, tienda/artículo e historial previo
y = respuesta conocida: sales, unidades vendidas de ese día
```

Ejemplo inventado para entenderlo:

```text
ENTRADAS DE UNA FILA                         RESPUESTA CONOCIDA
Tienda 1, artículo 1, lunes,
venta hace 7 días = 10,
promedio reciente = 11, otras entradas...  →  sales = 12
```

Durante el entrenamiento damos muchos pares de entradas y respuestas. El algoritmo ajusta sus parámetros o sus árboles para aproximarse a las respuestas del entrenamiento. No entiende los productos como una persona ni “adivina” por el nombre: trabaja con relaciones numéricas.

### Las dos operaciones centrales de scikit-learn

```python
modelo.fit(X_entrenamiento, y_entrenamiento)
predicciones = modelo.predict(X_validacion)
```

- **`fit` = entrenar:** ajustar el modelo utilizando ejemplos con respuestas conocidas.
- **`predict` = predecir:** aplicar lo aprendido a otras entradas. Aquí no se le entrega la respuesta como una columna de entrada.

**Aprendizaje supervisado:** el entrenamiento incluye respuestas conocidas. **Regresión:** la respuesta es una cantidad numérica, no una etiqueta como “spam/no spam”.

El modelo no memoriza necesariamente la tabla completa: intenta construir una regla o estructura que también sirva para otras filas. Para comprobar si lo consigue necesitamos validación, no solo mirar qué bien responde con los ejemplos de entrenamiento.

## 8. Paso 4: comparar métodos, no elegir uno a ciegas

### A. Referencia semanal

Regla: “para hoy, repetir la venta de hace siete días”. Si el lunes anterior fueron 15 unidades, la regla devuelve 15.

**¿Para qué?** Para tener un punto de comparación sencillo. No basta con que un modelo complicado produzca cifras: queremos saber si mejora una regla fácil de implementar.

**No es un tercer algoritmo entrenado:** es nuestra referencia o *baseline*.

### B. Regresión lineal

Aprende una combinación ponderada de las entradas, más un término constante. Durante el ajuste calcula cuánto contribuye cada entrada a la estimación.

**¿Por qué la usamos?** Es una alternativa de regresión introductoria, interpretable en su forma general, que permite comparar un modelo lineal con uno capaz de representar relaciones más complejas.

### C. Random Forest para regresión

Un árbol divide los ejemplos mediante preguntas sobre las entradas, como “¿el promedio reciente supera cierto valor?”. Esos cortes se aprenden de los datos; no escribimos a mano todas las preguntas. Al llegar a una hoja del árbol se obtiene una estimación basada en los ejemplos agrupados allí.

Random Forest entrena varios árboles con variación entre ellos y **promedia sus predicciones**. Nuestro bosque usa 20 árboles, profundidad máxima 12, al menos 10 observaciones por hoja y una muestra del 10 % de las filas de entrenamiento por árbol. La semilla 42 fija la aleatoriedad para facilitar la reproducción con las mismas condiciones.

**¿Por qué compararlo?** Puede representar relaciones no lineales entre las variables. Eso no garantiza que siempre gane: se decide mediante resultados de validación.

En las predicciones se aplica un límite inferior de cero porque no tiene sentido devolver unidades vendidas negativas. Las predicciones pueden tener decimales: son estimaciones, no un registro de objetos físicamente vendidos ni una orden de compra.

**Archivos:** `src/pronostico/modelos.py`, `src/pronostico/entrenar.py`.

## 9. Paso 5: evaluar sin entrenar con el futuro

Para comprobar un pronóstico usamos un período histórico del que conocemos las respuestas, pero lo tratamos como si aún no hubiera ocurrido.

```text
FECHAS ANTERIORES                         FECHAS POSTERIORES
Entrenamiento                            Validación
El algoritmo aprende aquí                Predice aquí y luego medimos el error
──────────────────────────────────────────────────────────────► tiempo
```

**¿Por qué no mezclar al azar?** Nuestro uso es temporal. Queremos aproximar la situación de aprender del pasado y pronosticar lo que viene después, no entrenar con fechas posteriores a las que estamos evaluando.

Evaluamos tres bloques consecutivos de 90 días de 2017:

| Bloque | Fechas de validación | Fechas de entrenamiento |
|---|---|---|
| 1 | 6 de abril a 4 de julio | Anteriores al 6 de abril |
| 2 | 5 de julio a 2 de octubre | Anteriores al 5 de julio |
| 3 | 3 de octubre a 31 de diciembre | Anteriores al 3 de octubre |

Para cada bloque se vuelve a entrenar. El historial de entrenamiento crece: en el bloque 2 ya pueden usarse ventas del bloque 1 porque, desde esa nueva fecha de inicio, son pasado. **No se incorporan ventas futuras respecto del inicio de cada bloque al entrenamiento.**

### Dos escenarios distintos que evaluamos

**A. Pronóstico diario con historial observado.** Para cada fecha tenemos las ventas registradas hasta el día anterior. Los modelos se mantienen entrenados para el bloque, pero las entradas históricas reflejan las ventas que se habrían ido conociendo. Es útil para simular una operación que recibe ventas nuevas cada día.

**B. Pronóstico recursivo de 90 días.** Al iniciar el bloque no conocemos ninguna venta de ese período. Pronosticamos día por día y añadimos las propias estimaciones al historial. Las ventas observadas del bloque quedan reservadas para medir errores.

```text
Historial observado hasta antes del bloque
                   │
                   ▼
Pronóstico día 1 → se agrega al historial
                   │
                   ▼
Pronóstico día 2 → se agrega al historial
                   │
                   ▼
             ... hasta el día 90

Las respuestas observadas del bloque se usan para medir, no para alimentar
el historial de esta simulación recursiva.
```

**¿Por qué hicimos la segunda prueba?** Porque al producir de una vez varios días futuros no tendremos sus ventas intermedias. Los errores pueden influir en días posteriores. La evaluación diaria y la recursiva no son la misma prueba y no deben mezclarse como si lo fueran.

**Archivos:** `src/pronostico/validacion_temporal.py` y `src/pronostico/validacion_recursiva.py`.

## 10. ¿Cómo medimos errores y qué obtuvimos?

### MAE: error absoluto medio

Ejemplo didáctico:

| Venta observada | Predicción | Diferencia absoluta |
|---:|---:|---:|
| 20 | 17 | 3 |
| 10 | 15 | 5 |

`MAE = (3 + 5) / 2 = 4 unidades`.

Se promedian las diferencias sin importar si el modelo quedó por arriba o por abajo. **No es un porcentaje ni una garantía de error máximo.**

### RMSE: raíz del error cuadrático medio

Se elevan los errores al cuadrado, se promedian y se extrae la raíz. Así los errores grandes pesan más. Se expresa también en unidades. Para los errores 3 y 5 del ejemplo sería aproximadamente 4.12 unidades.

En ambas métricas, **menor es mejor dentro de la misma evaluación**. El error se mide por registro: tienda, artículo y día; no es el error del total de toda la cadena.

### Resultados promedio de los tres bloques

| Método | MAE diario | RMSE diario | MAE recursivo | RMSE recursivo |
|---|---:|---:|---:|---:|
| Referencia semanal | 9.1883 | 12.0233 | 11.7130 | 15.4075 |
| Regresión lineal | 7.2411 | 9.4621 | 8.8747 | 11.7129 |
| Random Forest | **6.7248** | **8.7661** | **8.0865** | **10.5055** |

La tabla resume promedios de las métricas calculadas por bloque. **Random Forest obtuvo el mejor promedio en ambos protocolos.** Los dos modelos entrenados mejoraron la referencia semanal en promedio.

### No ganó siempre el mismo modelo

En el segundo bloque recursivo:

- Regresión lineal: MAE **8.4953**, RMSE **11.0624**.
- Random Forest: MAE **9.5382**, RMSE **12.0678**.

Por eso decimos que Random Forest fue **el candidato con menor error promedio en estas pruebas**, no que sea universalmente superior. Tampoco convertimos su MAE en un porcentaje de “precisión”. Para decidir si un error de ocho unidades es aceptable comercialmente haría falta contexto del producto y del negocio.

**Límite de la evidencia:** estos bloques se utilizaron para comparar y seleccionar modelos. No son una prueba independiente adicional hecha después de la selección. No hay un resultado final independiente con respuestas conocidas de `test.csv`.

### Análisis complementario

Revisamos importancia de variables en un Random Forest del análisis: `lag_7` obtuvo 0.5273 y `media_7_dias`, 0.3891. Eso indica que ese modelo utilizó principalmente historial reciente. **No significa causalidad, exactitud del 91 % ni que las demás variables no sirvan.** La importancia puede repartirse entre entradas relacionadas.

También generamos gráficos de ventas agregadas observadas/predichas y errores por artículo. Esos gráficos disponibles corresponden a la **evaluación diaria**, no al pronóstico recursivo del `test.csv`.

## 11. Paso 6: generar los pronósticos que muestra la web

Después de la comparación usamos Random Forest como candidato para generar la salida de Kaggle.

El script `src/pronostico/predecir_test.py`:

1. Lee `train.csv` y construye las entradas históricas.
2. Entrena Random Forest con las filas utilizables de todo ese historial, hasta diciembre de 2017.
3. Lee las combinaciones fecha/tienda/artículo solicitadas en `test.csv`.
4. Pronostica en orden, día por día, de enero a marzo de 2018.
5. Actualiza el historial con sus propias predicciones, no con ventas futuras desconocidas.
6. Guarda `resultados/predicciones_kaggle.csv`, con columnas `id,sales`.

**¿Por qué volver a entrenar con todo el historial?** La evaluación ya permitió comparar candidatos; para generar esta salida futura aprovechamos los ejemplos históricos disponibles. Eso no proporciona una nueva medición del error.

El archivo contiene **45 000 predicciones**: 90 días × 10 tiendas × 50 artículos.

**¿Por qué no medimos su MAE?** Porque nuestro `test.csv` no contiene las ventas observadas de esas fechas. Para medir error necesitamos tanto predicción como respuesta conocida. Las métricas anteriores se obtuvieron de bloques reservados de `train.csv`, no de este archivo.

## 12. Paso 7: conectar los resultados con la página existente

### La conexión actual es por archivos

```text
EN EL PROYECTO DE PYTHON

test.csv                          predicciones_kaggle.csv
id, fecha, tienda, artículo        id, cantidad pronosticada
             │                         │
             └───────────┬─────────────┘
                         ▼
              preparar_demo_web.py
        Une ambos archivos por el mismo id de Kaggle
                         │
                         ▼
               pronosticos-kaggle.json
           fecha + tienda + artículo + predicción
                         │
                         ▼
EN LA WEB EXISTENTE DE KAPAQ

       Nueva sección: “Pronóstico ML (demo)”
                         │
                         ▼
           Lee el JSON y permite filtrar filas
                         │
                         ▼
          Resumen + gráfico diario + tabla
```

**¿Por qué necesitamos unir los archivos?** La salida del desafío tiene `id,sales`. La pantalla necesita además fecha, tienda y artículo para organizarlos. El ID permite unir cada cantidad con la fila correcta, sin confiar en que ambos archivos casualmente estén en el mismo orden.

**¿Qué es JSON?** Un formato de texto para transportar datos estructurados. No es un modelo ni otro entrenamiento. Una fila de la demo puede verse así:

```json
{
  "date": "2018-01-01",
  "store": 1,
  "item": 1,
  "sales": 13.6
}
```

La predicción para ese registro es 13.6 unidades. Su presencia en la pantalla no demuestra que esa haya sido la venta observada.

### ¿Qué validaciones hace la preparación del JSON?

Comprueba columnas e IDs, rechaza IDs duplicados o que no coincidan entre los archivos, y verifica que no haya ventas estimadas faltantes, no finitas o negativas. Redondea las predicciones a dos decimales para mostrarlas.

### ¿Qué agregamos a la web y qué hace?

- Una opción del panel: **Pronóstico ML (demo)**.
- Selector de **tienda Kaggle** y **artículo Kaggle**.
- Fecha inicial y final, dentro del período disponible.
- Resumen del número de días, suma de las unidades pronosticadas y promedio diario.
- Gráfico de las cantidades estimadas por fecha.
- Tabla con fecha, tienda, artículo y cantidad pronosticada.
- Avisos que identifican los datos externos y el carácter precomputado de los pronósticos.

**Los totales se calculan sumando las predicciones seleccionadas**, no consultando ventas de Kapaq. El promedio es esa suma dividida entre el número de días seleccionados.

### ¿Qué ocurre cuando cambio un filtro?

La página selecciona otras filas del JSON y vuelve a dibujar sus resultados. **No llama a Python, no entrena y no hace una nueva inferencia del modelo.** El entrenamiento y las predicciones ocurrieron antes de abrir la pantalla.

Por eso la implementación es una **integración por lotes de resultados del modelo**: Python calcula primero y la web los publica después. No es un servicio de inferencia en línea. Para actualizarla hay que volver a generar las predicciones y el JSON.

### ¿Qué NO hace esta sección?

- No consulta ventas reales de Kapaq para alimentar el modelo.
- No relaciona “artículo 1” con un nombre de producto del catálogo de Kapaq.
- No cambia stock, pedidos ni registros de venta.
- No permite predecir una fecha arbitraria fuera del archivo.
- No vuelve a entrenar ni produce pronósticos nuevos al pulsar un botón.
- No determina cuánto se debe comprar al proveedor.

## 13. ¿Qué comprobamos de la integración?

En las verificaciones locales realizadas antes de esta guía:

| Comprobación | Resultado observado |
|---|---|
| Acceso autenticado a la nueva pantalla | Se pudo entrar al panel y abrir la demo |
| Carga del JSON | Respuesta HTTP 200 |
| JSON frente a `test.csv` y predicciones CSV | 45 000 filas; sin combinaciones ausentes o duplicadas; coincidencia dentro del redondeo a dos decimales |
| Cambio de tienda/artículo | Cambiaron título, tabla y resumen |
| Rango de siete días | Siete filas y resumen correspondiente |
| Selección de un solo día | Una fila y un punto en el gráfico |
| Rango invertido | Mensaje de que no hay pronósticos para ese rango |
| Consola durante la prueba | Sin errores observados |
| Compilación de la aplicación | Finalizó correctamente |

No se probó en el navegador la ausencia deliberada del JSON ni se hicieron pruebas exhaustivas de todos los dispositivos o todas las selecciones. **Estas comprobaciones validan la carga y presentación de datos; no prueban precisión en Kapaq ni sustituyen la evaluación del modelo.**

## 14. Relación con lo visto en el curso

La referencia general son los materiales disponibles de las semanas **2 a 12**, no solo las dos últimas. No afirmamos haber implementado todos los temas del curso ni necesitamos forzar reconocimiento de imágenes u otros temas que no corresponden al problema.

En este proyecto se aplican especialmente:

- **Metodología:** problema → datos → preparación → experimentación → evaluación → demostración → limitaciones.
- **Datos y estadística:** variables, promedios, variación y medidas de error.
- **Preparación de datasets:** revisión de estructura, orden temporal y construcción de características.
- **Aprendizaje supervisado y regresión:** aprender una salida numérica con ejemplos etiquetados.
- **Codificación categórica:** identificar tiendas y artículos sin tratarlos como magnitudes.
- **Validación:** distinguir aprender de evaluar y evitar usar el futuro indebidamente.
- **Sesgos y límites de los datos:** un dataset externo no representa automáticamente a Kapaq.

Los materiales están en `/Users/ricardo/Desktop/machineLearning/`. En la revisión de esa carpeta estaban las semanas 2, 3, 4, 5, 7, 8, 9, 11 y 12; no aparecían archivos de las semanas 6 y 10. Para los detalles de datasets y regresión, las semanas 11 y 12 son las referencias más directas, sin reemplazar el marco general del curso.

## 15. Cómo mostrarlo en clase, paso a paso

**Antes de salir:** lleva esta guía y los archivos o capturas que quieras mostrar. La demostración en vivo requiere que los servicios de la aplicación estén encendidos; abrir este Markdown no inicia la web. No dependas de que esta conversación esté disponible.

1. **Presenta el objetivo:** “Agregamos a nuestra web existente un prototipo para mostrar pronósticos de unidades vendidas”.
2. **Aclara el origen:** “Desarrollamos la parte de ML con Kaggle porque Kapaq aún no tiene historial suficiente”.
3. **Muestra una fila de `train.csv`:** explica fecha, tienda, artículo y cantidad. No inventes nombres para los artículos.
4. **Explica la preparación:** “Construimos datos de calendario y referencias de ventas anteriores, sin usar la respuesta del día”.
5. **Explica la comparación:** “Probamos una regla semanal, regresión lineal y Random Forest”.
6. **Muestra la tabla recursiva de la sección 10:** “Random Forest tuvo menor error promedio, pero no ganó todos los bloques”.
7. **Abre el panel y entra a Pronóstico ML (demo):** ruta `http://localhost:4200/admin/pronostico-demo` si la aplicación está en la configuración local habitual.
8. **Selecciona tienda 1 y artículo 1:** muestra que las fechas disponibles son enero–marzo de 2018, propias del desafío.
9. **Cambia a tienda 2 y artículo 2:** explica que son identificadores Kaggle y que cambian las predicciones consultadas.
10. **Selecciona del 2018-02-01 al 2018-02-07:** deben mostrarse siete días. Explica resumen, gráfico y tabla como tres vistas de los mismos pronósticos.
11. **Cierra con el alcance:** “El modelo calculó estos valores antes; la web los consulta. El siguiente objetivo es adaptar y evaluar el flujo con ventas genuinas de Kapaq”.

**Evita entrenar de nuevo durante la exposición:** no hace falta para explicar ni mostrar las predicciones existentes.

## 16. Guion de exposición de unos tres minutos

> “Nuestro grupo ya tenía una web del minimarket Kapaq y decidió añadir un componente de Machine Learning. La idea es estimar unidades vendidas por artículo y día como primer paso hacia una posible ayuda para planificar inventario.
>
> Al revisar la base disponible no encontramos historial real suficiente para entrenar y validar ese pronóstico. Por eso desarrollamos primero el experimento con el dataset público Store Item Demand Forecasting de Kaggle, que contiene registros de 2013 a 2017 para diez tiendas y cincuenta artículos. Los artículos se identifican con números; no los presentamos como productos de Kapaq.
>
> En Python organizamos los datos y construimos variables: día de la semana, mes, posición temporal, ventas de hace siete y catorce días y promedios de los días anteriores. Las entradas históricas excluyen la venta del día que queremos estimar.
>
> Comparamos regresión lineal y Random Forest con una referencia sencilla que repite las ventas de hace siete días. Evaluamos tres períodos de noventa días, entrenando siempre con fechas anteriores. Hicimos una evaluación diaria con historial observado y otra recursiva, que alimenta las predicciones anteriores al historial para imitar un horizonte futuro sin ventas intermedias conocidas.
>
> En la evaluación recursiva, Random Forest obtuvo el mejor promedio: MAE de 8.0865 unidades y RMSE de 10.5055. La referencia semanal tuvo un MAE de 11.7130. No obstante, regresión lineal fue mejor en el segundo bloque; no afirmamos que Random Forest gane siempre ni que ese error sea el de Kapaq.
>
> Después entrenamos el candidato con el historial utilizable completo y generamos las 45 000 predicciones solicitadas para enero a marzo de 2018. Preparamos un JSON con fecha, tienda, artículo y cantidad estimada. Añadimos una pantalla al panel para filtrar y mostrar esos resultados en resumen, gráfico y tabla.
>
> La integración actual muestra predicciones precomputadas de Kaggle: no ejecuta el modelo al cambiar los filtros ni modifica inventario. Nuestro siguiente objetivo sería reunir ventas genuinas de Kapaq, preparar su historial y entrenar y validar un pronóstico propio antes de darle un uso comercial.”

## 17. Preguntas de la profesora y respuestas claras

### ¿Dónde está el Machine Learning si la página solo muestra un archivo?

En el proceso de Python: entrenamiento de regresión lineal/Random Forest, evaluación y generación de los pronósticos. El archivo contiene sus resultados. La integración actual publica esos resultados por lotes; no ejecuta el modelo en línea. Son partes diferentes del mismo prototipo.

### ¿Por qué no pusieron nombres como arroz o escoba?

Porque Kaggle entrega identificadores y no establecimos una correspondencia con productos Kapaq. El modelo estudia el historial de cada identidad, no comprende semánticamente qué producto es. Inventar esos nombres daría una interpretación falsa.

### ¿Cómo demostraron que no son predicciones escritas a mano?

El código de `predecir_test.py` entrena Random Forest, construye las entradas por fecha, llama a `predict` y guarda la salida. `preparar_demo_web.py` une esa salida con `test.csv`; no inventa nuevas cantidades. Podemos mostrar el código, los archivos y la evaluación; la mera presencia de un gráfico no demuestra el entrenamiento.

### ¿Qué significa “error de 8.0865”?

Es el promedio entre bloques del error absoluto medio recursivo por tienda, artículo y día en Kaggle. No es 8 % ni significa que todos los pronósticos se equivoquen exactamente ocho unidades. Algunos errores son menores y otros mayores.

### ¿Qué porcentaje de precisión tiene?

Reportamos MAE y RMSE en unidades, no una exactitud de clasificación. No los convertimos arbitrariamente en porcentaje. Las cifras deben leerse con el volumen de ventas y el contexto del producto.

### ¿Por qué entrenar y validar no son lo mismo?

Entrenar es ajustar el modelo con ejemplos conocidos. Validar es medir sus predicciones en fechas reservadas que ese ajuste no utilizó. Responder bien sobre el entrenamiento no garantiza responder bien en fechas posteriores.

### ¿Por qué no miden el error de las predicciones que muestra la web?

Porque para ese `test.csv` no tenemos las respuestas observadas. Medimos errores en períodos de `train.csv` tratados como futuros durante la validación.

### ¿El sistema ya sabe cuánto comprar para Kapaq?

No. Hoy muestra estimaciones de ventas de un dataset externo. Una decisión de reposición necesitaría datos propios, validación, stock y reglas adicionales.

### ¿Vender pocas unidades significa que hubo poca demanda?

No necesariamente: puede que faltara stock y hubiera clientes sin atender. Entrenamos para estimar ventas registradas; no observamos toda la demanda no satisfecha. Es una limitación importante antes de usar el resultado comercialmente.

### ¿Cuántos días de Kapaq necesitan?

No hay una cifra universal. Las variables actuales requieren al menos 28 observaciones diarias previas por historial, pero eso solo permite calcular las entradas: **no significa que 28 días basten para un modelo confiable**. Necesitamos más historia, diversidad temporal y períodos separados para evaluar.

### ¿Podrían simular ventas para llenar ese historial?

Para demostrar una conexión técnica, sí, si se etiquetan como simuladas. Pero no serviría para demostrar precisión sobre ventas reales de Kapaq ni reemplazaría su historial genuino.

### ¿Qué sigue para predecir con ventas propias?

Registrar ventas genuinas y distinguirlas de demos; definir la fecha de venta y los pedidos que cuentan; agregar unidades por producto y día; revisar faltantes, días sin ventas y faltas de stock; construir entradas sin usar el futuro; entrenar y validar temporalmente. Solo después decidir cómo actualizar y mostrar el pronóstico en la web. Esa adaptación aún no se realizó.

## 18. Mapa de archivos para mostrar evidencia

### Repositorio de Machine Learning

Raíz: `/Volumes/Ricardo/Documentos/MachineLearning`

| Archivo desde esa raíz | Qué demuestra |
|---|---|
| `datasets/train.csv` | Ejemplos históricos con respuestas `sales` |
| `datasets/test.csv` | Fechas, tiendas y artículos que se quieren pronosticar |
| `proyecto/src/pronostico/datos.py` | Lectura, estructura y orden de datos |
| `proyecto/src/pronostico/variables.py` | Ventas desplazadas y promedios anteriores |
| `proyecto/src/pronostico/modelos.py` | Calendario, codificación, regresión y bosque |
| `proyecto/src/pronostico/entrenar.py` | Comparación en un corte cronológico |
| `proyecto/src/pronostico/validacion_temporal.py` | Comparación diaria en tres bloques |
| `proyecto/src/pronostico/validacion_recursiva.py` | Comparación recursiva en tres bloques |
| `proyecto/resultados/comparacion_recursiva_90_dias.csv` | Errores obtenidos por bloque y método |
| `proyecto/src/pronostico/predecir_test.py` | Entrenamiento final y generación de predicciones |
| `proyecto/resultados/predicciones_kaggle.csv` | Salida `id,sales` del pronóstico Kaggle |
| `proyecto/src/pronostico/preparar_demo_web.py` | Puente entre la salida del modelo y la web |
| `proyecto/RESULTADOS_ML.md` | Informe técnico, gráficos y fuentes |

### Cambios de ML en la web existente

Raíz: `/Volumes/Ricardo/Documentos/Proyecto-Minimarket`

| Archivo desde esa raíz | Función añadida |
|---|---|
| `frontend/src/assets/data/pronosticos-kaggle.json` | Datos que consulta la pantalla |
| `frontend/src/app/caracteristicas/admin/pronostico-demo/pronostico-demo.component.ts` | Lectura, filtros, resumen, gráfico y tabla |
| `frontend/src/app/app.routes.ts` | Ruta de acceso a la pantalla |
| `frontend/src/app/caracteristicas/admin/admin-diseno/admin-diseno.component.ts` | Enlace del menú hacia Pronóstico ML (demo) |

No hace falta explicar la arquitectura completa de la web anterior para defender este avance: lo nuevo que nos corresponde es la producción y presentación de estos pronósticos.

## 19. Frases que sí y que no conviene decir

| Evitar | Decir en su lugar |
|---|---|
| “Ya predecimos las ventas de Kapaq” | “Desarrollamos una demo con datos externos y queremos adaptarla a Kapaq” |
| “La web entrena la inteligencia artificial” | “Python entrenó el modelo; la web consulta sus resultados guardados” |
| “Artículo 1 es arroz” | “Artículo 1 es un identificador de Kaggle; no tenemos esa correspondencia” |
| “Random Forest es siempre el mejor” | “Tuvo el menor error promedio en las evaluaciones realizadas” |
| “Tenemos 92 % de precisión” | “Reportamos MAE y RMSE en unidades; no calculamos ese porcentaje” |
| “El gráfico prueba que acierta” | “El gráfico ayuda a visualizar; las métricas comparan predicción con respuesta observada” |
| “Solo faltan 28 días de ventas para que funcione bien” | “28 días permiten ciertas variables; la suficiencia del historial y el rendimiento deben evaluarse” |

## 20. Si solo recuerdas cinco ideas

1. **Queremos estimar unidades vendidas por fecha y artículo; no clasificar productos ni recomendar compras automáticamente.**
2. **Python aprendió relaciones entre historial/calendario y ventas en Kaggle; no sabe qué significa un producto.**
3. **Comparamos métodos en fechas posteriores al entrenamiento; Random Forest tuvo el mejor promedio, no ganó siempre.**
4. **La página muestra predicciones ya calculadas mediante un JSON; no calcula nuevos pronósticos en línea.**
5. **Para predecir Kapaq necesitamos historial genuino suficiente y una evaluación propia; la demo actual no demuestra ese rendimiento.**

---

### Fuentes y material de apoyo

Para mostrar evidencia visual, también están estos archivos dentro de `proyecto/`:

- `resultados/ventas_reales_vs_predichas.png`: ventas agregadas observadas y predichas en la validación diaria del período final.
- `resultados/mae_por_articulo.png`: diez artículos con mayor MAE en esa validación.
- `resultados/importancia_variables_random_forest.csv`: importancia de las entradas en el modelo analizado.

Si se regeneran las predicciones, hay que actualizar también el JSON de la web. Desde la raíz de `MachineLearning`, el comando es:

```bash
venv/bin/python proyecto/src/pronostico/preparar_demo_web.py
```

Ese paso prepara los datos para la pantalla; no reentrena el modelo. Después se debe comprobar nuevamente la demo.

- Materiales de clase disponibles de las semanas 2–12 en `/Users/ricardo/Desktop/machineLearning/`.
- Dataset/desafío: https://www.kaggle.com/competitions/demand-forecasting-kernels-only
- Regresión lineal: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- Métricas: https://scikit-learn.org/stable/modules/model_evaluation.html
- Resultados locales: [RESULTADOS_ML.md](RESULTADOS_ML.md).

**Nota:** las cifras de esta guía describen las ejecuciones documentadas y la demo comprobada; no se reentrenó ningún modelo para redactarla.
