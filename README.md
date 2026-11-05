# Pronóstico de demanda — proyecto de Machine Learning

Proyecto académico desarrollado localmente en Python para estimar las ventas diarias por tienda y artículo. Utiliza el dataset externo **Store Item Demand Forecasting** de Kaggle. Estos datos no representan las ventas reales del Minimarket Kapaq.

## Estructura

```text
MachineLearning/
├── datasets/                   # train.csv, test.csv y sample_submission.csv
├── proyecto/
│   ├── src/pronostico/         # módulos de datos, variables, modelos y evaluación
│   ├── resultados/             # predicciones, métricas y gráficos generados
│   ├── sql/                    # auditoría de ventas de Kapaq, solo lectura
│   ├── GUIA_PARA_EXPLICAR_EL_PROYECTO.md # explicación paso a paso y guion para clase
│   ├── RESULTADOS_ML.md        # síntesis académica de los experimentos
│   └── regresion_lineal.py     # primer experimento introductorio
├── requirements.txt
└── README.md
```

El entorno virtual (`venv/`) no se versiona: se crea localmente con los comandos siguientes y las dependencias quedan descritas en `requirements.txt`.

## Requisitos

- Python 3.13 recomendado (el proyecto fue ejecutado con Python 3.13).
- Visual Studio Code con la extensión de Python.

## Preparación en macOS

Desde la carpeta raíz del repositorio:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En VS Code, abre la carpeta raíz, selecciona `venv/bin/python` como intérprete y abre una terminal integrada.

## Ejecutar

Cada script puede ejecutarse con el intérprete del entorno virtual:

```bash
venv/bin/python proyecto/src/pronostico/datos.py
venv/bin/python proyecto/src/pronostico/evaluacion.py
venv/bin/python proyecto/src/pronostico/referencias.py
venv/bin/python proyecto/src/pronostico/variables.py
venv/bin/python proyecto/src/pronostico/entrenar.py
venv/bin/python proyecto/src/pronostico/validacion_temporal.py
venv/bin/python proyecto/src/pronostico/validacion_recursiva.py
venv/bin/python proyecto/src/pronostico/analizar_validacion.py
venv/bin/python proyecto/src/pronostico/graficar_validacion.py
venv/bin/python proyecto/src/pronostico/importancia_variables.py
venv/bin/python proyecto/src/pronostico/predecir_test.py
venv/bin/python proyecto/src/pronostico/preparar_demo_web.py
```

Para entrenar y comparar la referencia semanal, la regresión lineal y Random Forest en la validación cronológica, ejecutar `entrenar.py`. `validacion_temporal.py` repite la comparación en tres períodos de 90 días con ventas previas observadas; `validacion_recursiva.py` imita un horizonte de 90 días usando predicciones anteriores como historial. `predecir_test.py` genera `proyecto/resultados/predicciones_kaggle.csv` con el formato `id,sales`. Para preparar la demostración del panel web, ejecutar después `preparar_demo_web.py`; este paso no reentrena el modelo y crea un JSON a partir de ese CSV y el `test.csv` de Kaggle.

## Resultados existentes

Los CSV y gráficos actualmente incluidos en `proyecto/resultados/` documentan los experimentos realizados. Promedio de la validación recursiva en tres períodos:

| Método | MAE recursivo | RMSE recursivo |
|---|---:|---:|
| Referencia: ventas de hace 7 días | 11.7130 | 15.4075 |
| Regresión lineal | 8.8747 | 11.7129 |
| Random Forest | **8.0865** | **10.5055** |

Random Forest tuvo el mejor promedio, pero la regresión lineal fue mejor en el segundo período; no hubo un ganador en todos los bloques. MAE y RMSE se miden en unidades de venta por fila evaluada. Las métricas son únicamente del dataset de Kaggle; no deben interpretarse como precisión sobre Kapaq ni como recomendaciones reales de inventario. El detalle y las fuentes están en `proyecto/RESULTADOS_ML.md`.

## Datos

Los archivos CSV presentes en `datasets/` son datos externos descargados de Kaggle. El archivo `test.csv` no contiene ventas reales, por lo que las predicciones generadas para él no tienen métricas observables en este proyecto. El panel administrativo incluye una pantalla de demostración separada, con identificadores Kaggle y una advertencia explícita: no se conecta al inventario ni representa ventas reales de Kapaq.

Para revisar si existe historial propio suficiente, `proyecto/sql/auditoria_ventas_kapaq.sql` contiene consultas agregadas de solo lectura; no modifican la base de datos ni consultan datos personales.
