# Verificación de la entrega

Verificación realizada el 2026-11-11T10:46:35.142537+00:00 (UTC), en macOS ARM con el entorno incluido.

## Ejecución y servicio HTTP

Se activó `.venv` y se ejecutó literalmente `python app.py`. El programa reservó el puerto e imprimió **http://127.0.0.1:5000**. Esa dirección se abrió en Chrome y sirvió la interfaz desde Flask. No se abrió el archivo HTML manualmente.

Las rutas `/`, `/api/health`, `/api/resultados` y las tres descargas de documentación respondieron HTTP 200. Se probaron por HTTP una clasificación normal, abstención, señal explícita de precaución y una variante con una letra omitida. Evidencia legible por máquina: [http_verificacion.json](http_verificacion.json).

## Pruebas automatizadas

Resultado final: **57 pruebas aprobadas**, **0 fallos**, **0 errores**, **0 omitidas**. Registro: [pruebas.xml](pruebas.xml).

Cubren validación de JSON, texto vacío, longitud, tipos, cuerpo excesivo, estado degradado, prioridad de seguridad sin modelo, continuidad de precaución, menciones explícitas/negadas/informativas, acceso a documentos, Host, cabeceras, ausencia de cookies, respuesta normal, palabras desconocidas, conservación de negaciones, grupos, ausencia de ajuste sobre validación, métricas con ejemplos calculables, cobertura, no repetición de test, recuperación del modelo desde train y rechazo de un artefacto alterado.

`python -m pip check` terminó con `No broken requirements found.`

## Comprobación visual e interactiva

Se utilizó Chrome real mediante control de interfaz. El navegador integrado no estaba disponible; la comprobación continuó en Chrome.

- Vista de escritorio: carga del diseño, disponibilidad local y formulario.
- Ejemplo de alegría: respuesta de la API visible, distribución por clase y aportes de palabras.
- Texto desconocido: petición de contexto, sin forzar neutralidad.
- Mensaje explícito de autolesión: apoyo, Línea 113 opción 5 y supresión de scores emocionales.
- Mensaje siguiente «Estoy solo»: se conserva la precaución dentro de la conversación.
- Laboratorio: comparación, ganador, matriz de confusión, cobertura y curva tabular coinciden con los resultados guardados.
- Nueva conversación: historial y panel vuelven al estado inicial; exportación se deshabilita cuando no hay sesión.
- Diálogo de privacidad y vista El proyecto: textos y enlaces de documentación visibles.
- Emulación iPhone SE, **375 × 667**: navegación adaptable, formulario accesible mediante desplazamiento, panel bajo el chat y envío real de un mensaje de preocupación; se observó respuesta y etiqueta Ansiedad.
- Se restauró la vista de escritorio y una conversación vacía al finalizar.

Se corrigió un margen vertical innecesario y el favicon ausente observado en consola. Durante una entrada automatizada, el método de tecleo omitió la ñ de «daño»; se verificó el texto correcto mediante pegado y se añadió también la variante «dao» a la regla y a una prueba. Esto no garantiza reconocer cualquier error ortográfico. No se modificaron modelo ni datos tras evaluar test.

La inspección móvil fue emulación de navegador, no un ensayo en teléfono físico. La exportación utiliza Blob local y su código fue revisado; no se afirma una validación clínica ni una auditoría formal completa de accesibilidad.

## Integridad experimental

Después de ejecutar `python -m entrenamiento todo` sobre la entrega, los SHA-256 de `seleccion.json`, `test_final.json` y `test_reservado.json` permanecieron idénticos. La operación reutiliza la evidencia; no reabre el test. El registro HTTP contiene las huellas verificadas. El resultado final mantiene una sola apertura registrada.

Las pruebas de software y mensajes de demostración son de desarrollo; no son nuevas muestras para elevar las cifras del informe. Las cifras de rendimiento proceden exclusivamente del experimento congelado.


## Ajuste posterior de legibilidad

Se recuperó el diseño claro original y se aumentó el tamaño de los mensajes del chat a 18 px en escritorio y 16 px en móvil.


## Actualización del diseño claro

Se modernizó la interfaz con tipografía del sistema, tonos claros y colores sólidos sin degradados. Se comprobó en Chrome la carga y un envío del chat. La batería existente terminó con 57 pruebas aprobadas.


## Dock inferior

Se conservó una copia de la interfaz previa en `.versiones/antes_dock_20261111_180800/`. Se verificó en Chrome el cambio entre pestañas y en emulación móvil el arrastre del indicador hacia Laboratorio.
