# 07. Instalación, reproducibilidad y verificación

Esta guía resume los comandos exactos para correr el proyecto en cualquier computadora, repetir el entrenamiento desde cero y verificar que las 11 pruebas pasen sin errores.

---

## 1. Requisitos e instalación

- **Python** 3.10 o superior (entorno probado en macOS con entorno virtual `.venv`).
- Dependencias directas en `requirements.txt`: `Flask`, `scikit-learn`, `numpy`, `scipy`, `joblib`.

Si necesitas recrear el entorno en otra computadora:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## 2. Arrancar la aplicación web

```bash
source .venv/bin/activate
python app.py
```

Abre en el navegador la dirección que aparece en consola: **`http://127.0.0.1:5000`**.

---

## 3. Reproducir el entrenamiento completo desde cero

Todo es determinista (`SEMILLA = 42`). Si quieres volver a generar el corpus curado de 2,780 frases, reentrenar los 9 modelos, ejecutar la rejilla de hiperparámetros y regenerar `modelos/modelo_base.joblib` y `modelos/resultados_experimentos.json`:

```bash
source .venv/bin/activate
python -m entrenamiento.p02b_generar_corpus_implicito
python -m entrenamiento.p08_experimentos
```

---

## 4. Ejecutar las 11 pruebas automáticas (`unittest`)

```bash
source .venv/bin/activate
python -m unittest discover -s pruebas -v
```

Qué verifican las 11 pruebas:
1. Conteos exactos de datos (`26,469` en entrenamiento, `1,899` en GoEmotions, `1,932` en sintética) y **0 fugas** entre particiones.
2. Limpieza de texto preservando `ñ` y marcado de negaciones (`neg_`).
3. Detección de riesgo de suicidio/autolesión sin falsos positivos en frases figuradas (*"me muero de risa"*).
4. Prioridad de seguridad con **Línea 113 Opción 5** y probabilidades que suman 1.
5. Coherencia de resultados en validación (trigramas sobre bigramas y ganador híbrido sobre trigramas) y ausencia de frases diagnósticas en entrenamiento.
6. Protección de evaluación única en el script de test final (`p10_evaluar_test.py`).
7. Formato uniforme de comentarios en todo el código Python.
8. Interfaz web limpia, sin degradados CSS y con rutas relativas.
9. Funcionamiento de `/api/chat` en mensajes normales, crisis y **memoria de 1 turno con `contexto_previo`**.
10. Ciclo completo de aprendizaje local: bloqueo de frases de riesgo, guardar ejemplo, reentrenar nueva versión, comprobar predicción, restaurar modelo base y eliminar ejemplo.
11. Endpoint `/api/laboratorio` devolviendo los **9 modelos** y las métricas calculadas.

---

## 5. Evaluación única sobre el conjunto test final

Cuando la profesora entregue el archivo CSV de prueba final (con columnas `texto` y `etiqueta`), se evalúa el **modelo base congelado una sola vez**:

```bash
source .venv/bin/activate
python -m entrenamiento.p10_evaluar_test --test /ruta/al/test_final.csv
```

El script verifica si existe solape con entrenamiento y guarda en `modelos/evaluacion_final/`:
- `metricas_test.json` (Macro-F1, exactitud, matriz de confusión y abstención).
- `predicciones_test.csv` (predicción fila por fila).
