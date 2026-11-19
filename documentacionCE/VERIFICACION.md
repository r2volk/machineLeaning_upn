# Verificación de la entrega

Verificación realizada el 2026-11-11T10:46:35.142537+00:00 (UTC), en macOS ARM con el entorno incluido.

## Ejecución y servicio HTTP

Se ejecutó `python app.py` con `.venv` activo y se abrió en Chrome la dirección impresa: **http://127.0.0.1:5000**. Flask sirvió la página correctamente.

Las rutas de servicio son `/`, `/api/health` y `/api/resultados`. Las descargas permitidas son `/documentos/informe` y `/documentos/protocolo`.

Se probaron por HTTP una clasificación normal, abstención, señal explícita de precaución y una variante con una letra omitida.

## Pruebas automatizadas

Pruebas automatizadas ejecutadas el 2026-11-12: **61 pruebas aprobadas**, **0 fallos**, **0 errores**, **0 omitidas**. Registro: [pruebas.xml](pruebas.xml).

Cubren validación de JSON, texto vacío, longitud, tipos, cuerpo excesivo, estado degradado, prioridad de seguridad sin modelo, continuidad de precaución, menciones explícitas/negadas/informativas, acceso a documentos, Host, cabeceras, ausencia de cookies, respuesta normal, palabras desconocidas, conservación de negaciones, grupos, ausencia de ajuste sobre validación, métricas con ejemplos calculables, cobertura, no repetición de test, recuperación del modelo desde train y rechazo de un artefacto alterado.

`python -m pip check` terminó con `No broken requirements found.`

## Comprobación visual e interactiva

Se comprobó en Chrome la respuesta del chat, la abstención, la prioridad y continuidad de precaución y la coincidencia de Laboratorio con los resultados guardados. También se revisaron reinicio, privacidad y navegación móvil mediante emulación iPhone SE de 375 por 667; no fue un ensayo en teléfono físico ni una auditoría clínica.

Durante las pruebas se detectó la variante «dao» y se añadió a las reglas y a un test de software. Esto no garantiza reconocer todos los errores ortográficos ni modificó el clasificador emocional.

## Integridad experimental

Después de ejecutar `python -m entrenamiento todo` sobre la entrega, los SHA-256 de `seleccion.json`, `test_final.json` y `test_reservado.json` permanecieron idénticos. La operación reutiliza la evidencia; no reabre el test. El registro HTTP contiene las huellas verificadas. El resultado final mantiene una sola apertura registrada.

Las pruebas de software y mensajes de demostración son de desarrollo; no son nuevas muestras para elevar las cifras del informe. Las cifras de rendimiento proceden exclusivamente del experimento congelado.
