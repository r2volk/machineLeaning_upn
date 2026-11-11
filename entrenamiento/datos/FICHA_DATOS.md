# Ficha del corpus Ánimo v1

**Origen:** 100% sintético, redactado con asistencia de IA para esta entrega. No se extrajo de redes sociales ni se recogieron testimonios. No hay licencias de terceros sobre datasets, datos privados ni anotaciones clínicas. No es un conjunto preexistente de evaluación pública. No se afirma autoría humana independiente de la asistencia.

**Unidad:** un mensaje breve en español; dos mensajes expresan el mismo escenario y forman un grupo indivisible. `escenarios.txt` es la fuente legible. `p01_datos.py` materializa los CSV sin multiplicar plantillas ni crear frases nuevas en ejecución. Los conteos efectivos están en `manifiesto.json` y en el informe.

| Campo | Significado | Uso |
|---|---|---|
| `id` | Identificador estable de frase | Auditoría, nunca predictor |
| `grupo` | Escenario compartido por paráfrasis | División, bootstrap, nunca predictor |
| `texto` | Frase sintética original | Único predictor |
| `etiqueta` | Intención emocional de la redacción | Objetivo supervisado |
| `origen` | `sintetico_asistido_IA` | Procedencia |
| `particion` | train / validacion / test | Control de acceso experimental |

## Criterio de etiquetado

- `alegria`: satisfacción, entusiasmo, alivio o gratitud explícitos.
- `tristeza`: pena, soledad, desánimo o nostalgia expresados.
- `enojo`: molestia, frustración, indignación o rabia.
- `ansiedad`: miedo anticipatorio, tensión o preocupación; no diagnóstico de trastorno.
- `neutral`: descripción factual sin emoción explícita; no significa ausencia de emoción en una persona.

Se redactó una emoción dominante por escenario. No se anotaron mezclas, intensidad ni evolución temporal. Las etiquetas no se validaron con anotadores independientes; por tanto no se puede medir acuerdo interanotador ni compararlas con un estado real. Los géneros gramaticales varían, pero no representan grupos demográficos de participantes.

## Partición y control

Antes de dividir se rechazan textos duplicados tras normalizar. Se barajan escenarios por clase con semilla fija y se asignan 60/20/20; ambas paráfrasis siempre viajan juntas. Los hashes impiden cambios accidentales posteriores. Los límites de grupo son decisiones de autoría: escenarios distintos pueden compartir palabras, estructuras o temas. No se afirma ausencia total de semejanza semántica. Se preserva el corpus original aunque el test muestre errores.

## Usos admitidos y excluidos

Apto para demostrar clasificación, evaluación, comparación, explicación y limitaciones en un curso. No es evidencia de eficacia con personas reales ni un instrumento para decisiones médicas, educativas o laborales sobre individuos. No se debe reutilizar el test después de ajustar modelos a sus errores. Las expresiones de riesgo no se aprenden como una clase: hay una batería separada de reglas de desarrollo en `tests/casos_seguridad.json`.

## Sesgos previsibles

Lenguaje explícito y relativamente limpio; abundancia de contextos de estudio y vida cotidiana; misma autoría asistida para todas las particiones; clases balanceadas artificialmente; pocas variantes peruanas; sin población real de referencia; cobertura muy limitada de ironía, falta de ortografía, emojis, culturas, edades e idiomas. Balancear etiquetas no elimina estos sesgos.
