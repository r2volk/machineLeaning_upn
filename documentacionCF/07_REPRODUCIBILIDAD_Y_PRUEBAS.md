# 07. Cómo ejecutar, probar y repetir todo paso a paso

Comandos exactos para ejecutar el chatbot en terminal, abrir la web y correr las pruebas automáticas.

---

## 1. Requisitos e instalación

- **Python** 3.10 o superior (probado en macOS dentro del entorno `.venv`).
- Librerías en `requirements.txt`: `Flask`, `scikit-learn`, `numpy`, `scipy`, `joblib`.

Si mueves la carpeta del proyecto a otro lugar o cambias de computadora, recrea el entorno `.venv`:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

---

## 2. Correr el demo en consola (`chat_terminal.py`)

Ideal para mostrar primero en clase el chatbot puro:

```bash
source .venv/bin/activate
python3 chat_terminal.py
```

Escribe `salir` para terminar.

---

## 3. Correr la aplicación web (`app.py`)

```bash
source .venv/bin/activate
python3 app.py
```

Abre en el navegador la dirección que sale en consola: **`http://127.0.0.1:5000`**.

---

## 4. Repetir todo el entrenamiento desde cero

Usa semilla fija (`42`), así que siempre da exactamente las mismas cifras:

```bash
source .venv/bin/activate
python3 -m entrenamiento.p02b_generar_corpus_implicito
python3 -m entrenamiento.p08_experimentos
```

---

## 5. Ejecutar las 11 pruebas automáticas (`unittest`)

```bash
source .venv/bin/activate
python3 -m unittest discover -s pruebas -v
```

Verifican:
1. Conteos exactos (`26,469` entrenamiento, `1,899` GoEmotions, `1,932` sintética) y **0 frases repetidas** entre particiones.
2. Limpieza cuidando la `ñ` y marcado de negaciones (`neg_`).
3. Detección de riesgo sin falsos positivos en frases como *"me muero de risa"*.
4. Prioridad de **Línea 113 Opción 5** y porcentajes que suman 1.
5. Coherencia de resultados entre bigramas, trigramas y ganador híbrido.
6. Protección de evaluación única en el archivo de prueba final.
7. Estilo uniforme de comentarios en código Python.
8. Interfaz web limpia y rutas `/api/...` correctas.
9. `/api/chat` normal, crisis y memoria de 1 turno (`contexto_previo`).
10. Ciclo completo de Aprendizaje (bloquear riesgo → guardar → entrenar → comprobar → restaurar → borrar).
11. `/api/laboratorio` devolviendo los **9 modelos** y conteos reales.

---

## 6. Evaluación única del examen final (`test`)

Cuando la profesora entregue el CSV de prueba final (`texto` y `etiqueta`), se ejecuta una sola vez:

```bash
source .venv/bin/activate
python3 -m entrenamiento.p10_evaluar_test --test /ruta/al/test_final.csv
```

Guarda `modelos/evaluacion_final/metricas_test.json` y `modelos/evaluacion_final/predicciones_test.csv`.
