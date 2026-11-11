# Guion de defensa y demostración

La siguiente distribución es una **propuesta de exposición para cuatro integrantes**, no un registro ficticio de trabajo realizado. Estudiar todos los módulos: cualquiera debe poder explicar la separación train/validación/test.

1. **Problema y producto:** abrir la dirección impresa por `python app.py`. Mostrar el mensaje inicial y explicar que se clasifican señales textuales, no estados clínicos. Enviar «Estoy feliz porque terminé mi proyecto». Señalar que las respuestas están redactadas.
2. **Datos y aprendizaje:** abrir `entrenamiento/datos/escenarios.txt`, mostrar una línea con dos paráfrasis, luego `p01_datos.py` y el manifiesto. Explicar por qué no se separan al azar frases individuales. Seguir hasta `crear_modelo()` y mostrar `.fit()` solo con train.
3. **Experimentos y evidencia:** entrar a Laboratorio. Explicar la comparación, elección en validación, caída del resultado en test, la matriz y los errores. Decir «el resultado es limitado y eso está documentado», sin prometer precisión con usuarios reales. Mostrar una fila errónea del informe, incluso si fue aceptada.
4. **Ingeniería y límites:** probar «zzqwwx 🛸» para ver abstención. Mostrar un ejemplo de seguridad únicamente como prueba del sistema: «Estoy pensando en hacerme daño». Verificar que no aparezcan scores emocionales y que se indique apoyo humano. Explicar que las reglas fallan y nunca certifican seguridad. Mostrar Nueva conversación, exportación y archivos de pruebas.

## Recorrido de una frase

`index.html → POST /api/chat → validación → detectar_riesgo → predecir → vectorizador.transform → predict_proba → abstención y aportes → responder → JSON → textContent en el navegador`.

El riesgo corta el camino antes de ML. La bandera local conserva la respuesta de precaución para mensajes siguientes hasta iniciar una nueva conversación. No hay memoria semántica, diagnóstico ni contacto automático a servicios.

## Antes de exponer

Activa el entorno, arranca la aplicación y conserva la terminal abierta. Abre la dirección impresa. Ten visibles el README, el informe y los módulos p00–p06. No vuelvas a evaluar test ni modifiques datos para mejorar la demostración. Para explicar el entrenamiento usa el comando de demostración de train en `REPRODUCIBILIDAD.md`.

## Una conclusión defendible

«Implementamos un flujo completo de aprendizaje supervisado y un chatbot local explicable. La comparación determina el modelo con validación; el test se reserva una sola vez. El corpus es sintético y el resultado no permite afirmar rendimiento clínico o poblacional. El principal logro es un experimento reproducible con límites visibles y una aplicación que no oculta la incertidumbre».
