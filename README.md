# Ánimo

Chatbot académico en Python que clasifica señales de alegría, tristeza, enojo, ansiedad y neutralidad en textos breves en español. Su interfaz permite conversar e inspeccionar predicciones y resultados.

## Ejecutar

Primero prepara el entorno y las dependencias siguiendo las instrucciones de [instalación](docs/REPRODUCIBILIDAD.md).

1. Activa el entorno: `source .venv/bin/activate`
2. Ejecuta: `python app.py`
3. Abre la dirección exacta que imprime la terminal, normalmente **http://127.0.0.1:5000**. Si ese puerto está ocupado, el programa elige otro e imprime su dirección.

## Presentar en clase

Conversación permite probar mensajes y Laboratorio consultar resultados. Ambas vistas se abren desde la dirección impresa por Flask.

## Decisiones técnicas

| Decisión | Motivo |
|---|---|
| Flask + un único `index.html` | Sirve página y API localmente; JavaScript envía mensajes y muestra resultados. |
| Todo el ML en `entrenamiento/`, módulos `p00` a `p06` | Lectura progresiva: configuración → datos → modelos → comparación → test → inferencia → informe. El informe escrito se conserva en `INFORME_RESULTADOS.md`. `app.py` solo coordina HTTP. |
| Bolsa de palabras, TF-IDF y modelos clásicos | Conexión directa con NLP y regresión logística de los apuntes. Sin modelos preentrenados, embeddings ni APIs externas. |
| Elección por macro-F1 de validación | Compara cinco candidatos; el resultado determina el modelo servido. No se elige usando test. |
| Escenarios sintéticos con dos paráfrasis agrupadas | Procedencia explícita, sin datos personales; limita fuertemente las conclusiones sobre usuarios reales. |
| Particiones 60/20/20 por escenario y clase | Evita que las paráfrasis declaradas crucen particiones. Vocabulario e IDF solo aprenden de train. |
| Reserva de test, hashes y predicciones guardadas | Test final una sola vez; repetir comandos reutiliza evidencia. No se reajusta el modelo tras evaluar. |
| Abstención por score, margen y palabras conocidas | Permite pedir contexto. Scores sin calibrar: no son intensidad emocional ni probabilidad clínica. |
| Explicación mediante aportes de palabras | Visible y comprobable con coeficientes; asociaciones, no explicaciones psicológicas. |
| Reglas de seguridad antes de ML | Apoyo y Línea 113, opción 5 (Perú), sin etiquetar riesgo clínico. La alerta continúa en la sesión mediante un indicador local. |
| Respuestas redactadas, sin memoria semántica | Comportamiento predecible y defendible. No finge comprensión ni aprendizaje durante la conversación. |
| Sin persistencia de conversaciones | Los mensajes solo viven en memoria; exportar es una acción del usuario. Sin cuentas, analítica ni cookies. |
| Servidor en loopback y depuración desactivada | Adecuado para demostración en el equipo; no se presenta como un servicio público de producción. |

La base académica está en los apuntes de las semanas 4, 7, 9, 11 y 12 del curso. TF-IDF, Pipeline, partición por grupos, abstención y bootstrap son ampliaciones justificadas en el [mapa del curso](docs/MAPA_CURSO.md). No se usa la lematización con modelos descargados de los apuntes para respetar la restricción.

## Entrega

- [Informe de resultados y limitaciones](INFORME_RESULTADOS.md): cifras generadas por código, matriz, curvas y todos los errores de test.
- [Reproducibilidad e instalación en otro equipo](docs/REPRODUCIBILIDAD.md): preparación, experimentos y tests.
- [Protocolo fijado antes de evaluar](entrenamiento/PROTOCOLO.md) y [ficha del corpus](entrenamiento/datos/FICHA_DATOS.md).
- [Verificación de la entrega](docs/VERIFICACION.md): ejecución, pruebas y navegador.
- [Mapa del curso](docs/MAPA_CURSO.md) y [guion de exposición](docs/DEMOSTRACION.md).

El corpus es **100% sintético, redactado con asistencia de IA**. También se utilizó asistencia de IA para desarrollar este proyecto. No se atribuye a cuatro personas ni se afirma una duración de trabajo que no ocurrió. El alcance se demuestra con los entregables y la evidencia. Es un prototipo académico, **no una herramienta clínica**; puede fallar al interpretar emociones y al reconocer mensajes de riesgo.
