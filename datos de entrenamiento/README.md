# Datos de entrenamiento de E y F

| Carpeta | Qué contiene |
|---|---|
| `preparados/` | Una copia compartida de `train.csv` (23.689 frases), `validacion_go.csv` (1.899) y `validacion_sintetica.csv` (1.932). Usan las mismas filas y etiquetas en E y F. |
| `fuentes/` | CSV original de Claude con 10.000 frases sintéticas y los tres Parquet originales de GoEmotions traducido. La ficha y la licencia de GoEmotions están junto a los Parquet. |
| `corpus_original_E/` | Escenarios, manifiesto y tres particiones del primer experimento de E. El entrenamiento de esa versión aporta 240 ejemplos al CSV compartido. |
| `diagnostico_E/` | Frases destinadas al análisis exploratorio de negaciones y emociones indirectas. |

La columna `origen` permite demostrar qué datos entraron en el entrenamiento. El CSV compartido contiene 15.381 frases traducidas de GoEmotions, 8.068 sintéticas de Claude y 240 sintéticas del corpus original de E. Conserva separadas las particiones de validación y prueba al entrenar nuevos candidatos.

Para repetir la preparación desde las fuentes, consulta la [guía de E](../documentacionCE/APRENDIZAJE.md). Para reconstruir cada clasificador con los CSV ya preparados, consulta [E](../chatbotE/README.md) o [F](../chatbotF/README.md).
