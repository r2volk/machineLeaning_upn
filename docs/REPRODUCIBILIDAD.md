# Instalación y reproducción del experimento

## Preparación en otro equipo

Se necesita Python 3.11 o superior. La entrega se verificó en macOS ARM con Python 3.14.2. El proyecto incluye corpus, modelo y resultados: no descarga datos ni modelos al arrancar. La instalación de librerías sí requiere acceso a PyPI o un repositorio de paquetes local.

Desde la raíz del proyecto:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` fija dependencias directas. `requirements-lock.txt` conserva todas las versiones del entorno verificado; puede instalarse con `python -m pip install -r requirements-lock.txt` para replicar ese entorno. En Windows la activación es `.venv\Scripts\Activate.ps1`; el resto de comandos usa `python`.

Después sigue el README y abre la dirección de Flask; no abras `index.html` directamente.

## Leer y regenerar evidencia

```sh
python -m entrenamiento informe
python -m pytest
```

El primer comando regenera los gráficos y el CSV de errores **leyendo resultados existentes**. El segundo ejecuta pruebas de software con mensajes de desarrollo y datos de juguete. Ninguno vuelve a inferir sobre el test estadístico. No confundir `pytest` con el conjunto de prueba del modelo.

```sh
python -m entrenamiento todo
```

En la entrega congelada este comando verifica datos, lee la selección y el test existentes, y regenera los gráficos y el CSV de errores. El informe escrito se conserva en `INFORME_RESULTADOS.md`. **No vuelve a entrenar ni a evaluar** cuando ya hay resultados. Es idempotente para la evidencia principal. Los timestamps de los SVG exportados pueden cambiar; eso no altera métricas.

## Etapas del experimento

El experimento se ejecuta con estos comandos, en este orden:

```sh
python -m entrenamiento preparar
python -m entrenamiento comparar
python -m entrenamiento evaluar
python -m entrenamiento informe
```

- Preparar: crea CSV y manifiesto si faltan; si existen verifica los hashes.
- Comparar: solo usa train y validación, guarda modelos candidatos resumidos, curva, elección, umbral y artefacto. Si ya hay selección, la devuelve.
- Evaluar: verifica las huellas de los datos, el manifiesto y el modelo (no las del código Python), reserva test con modo exclusivo `x`, ejecuta un único lote de scores y guarda todas las predicciones. Si el resultado existe lo devuelve sin inferencia. Si solo existe una reserva incompleta, se detiene: no borres el marcador para forzar otra evaluación.
- Informe: solo interpreta los resultados guardados.

El artefacto servido es exactamente el seleccionado y evaluado. No se aprovecha validación para reentrenarlo después: hacerlo cambiaría la correspondencia entre el modelo medido y el modelo entregado.

## Demostrar entrenamiento sin reabrir test

El siguiente comando reconstruye en memoria un candidato usando únicamente train y muestra su vocabulario. Sirve para explicar `.fit()` durante la defensa; no reemplaza el artefacto ni mide test:

```sh
python -c "from entrenamiento.p01_datos import cargar,xy; from entrenamiento.p02_modelos import crear_modelo; m=crear_modelo('logistica_bigramas').fit(*xy(cargar('train'))); print(len(m['vectorizador'].vocabulary_))"
```

La función `cargar_modelo()` también reconstruye en memoria desde train si falta `modelo.joblib`; no sobrescribe la evidencia ni reabre test. Con otras versiones de librerías puede haber diferencias numéricas: usa las versiones fijadas y el modelo entregado para defender las cifras originales. No cargar archivos joblib recibidos de terceros: este formato ejecuta deserialización Python. El hash detecta modificaciones accidentales; no autentica al autor.

## Hacer un nuevo estudio

No borres resultados ni cambies la semilla para buscar una cifra mejor. Si ajustas a partir de errores ya observados, necesitas un corpus de prueba nuevo, congelado antes del ajuste, y otro directorio de experimento. Repetir este test con otro modelo ya no constituye evaluación independiente. Un reentrenamiento idéntico de train puede comprobar reproducibilidad, pero no crea nueva evidencia de generalización.

## Archivos de auditoría

`datos/manifiesto.json` registra origen, conteos, asignación de grupos y SHA-256. `resultados/seleccion.json` registra versiones, decisiones, resultados de validación y hashes informativos del código p00–p05, que no se comprueban para permitir la evaluación. `test_reservado.json` registra la única apertura y hash de `test_final.json`. Este último incluye cada texto sintético con su etiqueta, predicción, scores y abstención. Los datos de test son inspeccionables **después** de finalizar el experimento, para transparencia, no para ajustar.

Estos mecanismos no son un sistema de seguridad contra manipulación deliberada. Documentan y previenen errores rutinarios. La selección no se repitió después de observar test.

## API local

| Ruta | Contrato |
|---|---|
| `GET /` | Página de la aplicación. |
| `GET /api/health` | Estado del modelo; 503 en modo degradado. |
| `POST /api/chat` | JSON `message` (texto), `turn` (entero opcional), `safety_context` (booleano opcional). Devuelve `reply`, `safety` y `analysis`. |
| `GET /api/resultados` | Métricas guardadas; no expone corpus completo ni ejecuta evaluación. |
| `GET /documentos/informe`, `GET /documentos/protocolo` | Descargas de documentación permitida. |

Máximo de conversación: 1500 caracteres. Máximo de cuerpo HTTP: 32768 bytes. Dentro del cuerpo admitido, seguridad se comprueba antes del límite de conversación. El servidor no guarda sesiones; el navegador envía un indicador para mantener la precaución tras una alerta hasta que el usuario inicia una nueva conversación. Un cliente API independiente debe conservar ese indicador. No hay memoria semántica ni seguimiento clínico.
