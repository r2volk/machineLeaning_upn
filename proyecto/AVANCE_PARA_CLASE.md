# Avance del proyecto — Machine Learning

## Título provisional

**Predicción de ventas diarias por artículo mediante modelos de regresión y validación temporal**

> Este avance usa el dataset externo de Kaggle *Store Item Demand Forecasting*. No contiene datos reales del Minimarket Kapaq.

## Diapositiva 1 — Problema y objetivo

En un minimarket, estimar la demanda futura puede ayudar a planificar el inventario. Para desarrollar y evaluar el componente de Machine Learning, se utilizará inicialmente un conjunto de datos público de ventas diarias. El objetivo del experimento es estimar las unidades vendidas para cada combinación de tienda, artículo y fecha, comparar métodos de regresión y medir sus errores.

**Alcance actual:** estudiar el método con Kaggle y preparar una demostración. Aún no se afirma que las predicciones representen la demanda de Kapaq.

## Diapositiva 2 — Datos utilizados

- `train.csv`: 913 000 registros desde 2013-01-01 hasta 2017-12-31.
- 10 tiendas y 50 artículos.
- Columnas: fecha (`date`), tienda (`store`), artículo (`item`) y unidades vendidas (`sales`).
- En la inspección no se encontraron valores vacíos ni duplicados por fecha/tienda/artículo.
- `test.csv` contiene 45 000 filas futuras, pero no incluye las ventas reales.

## Diapositiva 3 — Variables y métodos

Se utilizaron variables de calendario (día de la semana, mes y posición temporal) e historial de ventas: unidades de hace 7 y 14 días y promedios de los 7 y 28 días previos. Los promedios excluyen el propio día que se predice.

Se compararon:

1. **Referencia semanal:** usar las ventas de hace siete días.
2. **Regresión lineal:** modelo de regresión introductorio visto en clase.
3. **Random Forest:** conjunto de árboles de decisión que combina sus predicciones.

## Diapositiva 4 — Evaluación

No se mezclaron fechas aleatoriamente. Se evaluaron tres bloques consecutivos de 90 días de 2017; para cada bloque, el entrenamiento utilizó fechas anteriores. Además de evaluar el pronóstico diario con historial observado, se hizo una prueba recursiva: durante los 90 días se alimentan al historial las predicciones del propio modelo, como ocurriría al pronosticar un período futuro sin ventas observadas.

- **MAE:** error absoluto promedio, en unidades vendidas por registro.
- **RMSE:** error en unidades que penaliza más los errores grandes.

## Diapositiva 5 — Resultados preliminares

En la evaluación recursiva —la más parecida a pronosticar varios días futuros—, promedio en tres bloques:

| Método recursivo | MAE (unidades) | RMSE (unidades) |
|---|---:|---:|
| Referencia semanal | 11.7130 | 15.4075 |
| Random Forest | **8.0865** | **10.5055** |

Random Forest presentó menor error que la referencia en los tres bloques recursivos. En la evaluación diaria con historial real, el promedio de Random Forest fue MAE 6.7248 y RMSE 8.7661. La diferencia entre protocolos muestra que pronosticar muchos días a la vez es más difícil.

En el análisis de importancia de variables, `lag_7` tuvo importancia 0.5273 y `media_7_dias` 0.3891. Esto muestra que el modelo dependió principalmente de las ventas recientes en este experimento; la importancia no implica causalidad.

## Diapositiva 6 — Estado y limitaciones

- Se generaron 45 000 predicciones para `test.csv` con el formato de Kaggle. Como este archivo no trae ventas reales, no se pudo calcular su precisión.
- Los resultados no miden el desempeño en Kapaq: las tiendas y artículos de Kaggle no se corresponden con el catálogo real.
- La evaluación recursiva se hizo con Kaggle; no demuestra precisión sobre ventas de Kapaq.
- La integración con la web todavía no se realizó.

## Próximos pasos hacia la entrega final

1. Revisar si la base de Kapaq contiene historial real suficiente de ventas por producto y fecha.
2. Si solo se demuestra la integración con Kaggle, etiquetar claramente la salida como **demostración con datos externos**.
3. Completar el informe con bibliografía, metodología, métricas, gráficos y limitaciones.

## Guion breve para exponer

> “Mi proyecto busca predecir ventas diarias por artículo. Empecé con datos públicos de Kaggle y comparé una referencia semanal, regresión lineal y Random Forest. Para imitar un pronóstico futuro de 90 días hice una evaluación recursiva en tres períodos: el modelo usa sus predicciones anteriores como historial. Random Forest obtuvo MAE promedio 8.0865 y RMSE 10.5055, menores que la referencia semanal. Esos resultados son solo de Kaggle; no son ventas de Kapaq. El siguiente paso es revisar si hay historial real suficiente en el sistema.”

## Preguntas que podrían hacer

**¿Por qué no separaste los datos al azar?**

Porque el objetivo es predecir fechas futuras. Entrenar con fechas posteriores a las de evaluación podría dar una estimación engañosa.

**¿Cuál método obtuvo menor error?**

Random Forest. En la evaluación recursiva de 90 días obtuvo menor error que la referencia semanal en los tres períodos de Kaggle.

**¿Ya predice las ventas reales del minimarket?**

No. El experimento actual usa Kaggle, que no representa necesariamente los productos, clientes ni ventas de Kapaq.

**¿Por qué generaste predicciones de `test.csv` si no puedes medir su error?**

Para producir la salida solicitada por el desafío de Kaggle. Al no haber ventas reales en `test.csv`, esa salida no permite calcular MAE ni RMSE.

## Archivos que se pueden mostrar

- `resultados/ventas_reales_vs_predichas.png`: comparación diaria en el último período de validación.
- `resultados/mae_por_articulo.png`: diez artículos con mayor MAE.
- `resultados/importancia_variables_random_forest.csv`: importancia de las variables.
- `resultados/validacion_recursiva_random_forest.csv`: métricas del backtest recursivo.
- `RESULTADOS_ML.md`: detalle metodológico y resultados comprobados.
