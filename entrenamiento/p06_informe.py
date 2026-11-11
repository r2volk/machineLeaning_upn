"""06 · Comunicar la evidencia guardada. No entrena ni vuelve a predecir test."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .p00_configuracion import CLASES, DATOS, SALIDAS, leer_json


def graficar(seleccion, final):
    """Exporta figuras estáticas a partir de métricas congeladas (PNG y SVG)."""
    carpeta = SALIDAS / 'figuras'
    carpeta.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fafbf7'})
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    candidatos = seleccion['candidatos']
    etiquetas = [c['nombre'].replace('_', '\n') for c in candidatos]
    x = np.arange(len(candidatos))
    ax.bar(x-.18, [c['train']['macro_f1'] for c in candidatos], .36, label='Train', color='#aebf8d')
    ax.bar(x+.18, [c['validacion']['macro_f1'] for c in candidatos], .36, label='Validación', color='#305b49')
    ax.set(xticks=x, xticklabels=etiquetas, ylim=(0, 1.05), ylabel='Macro-F1',
           title='Comparación previa al test · corpus sintético')
    ax.legend()
    guardar_figura(fig, carpeta / 'comparacion')
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    matriz = np.array(final['metricas']['matriz'])
    im = ax.imshow(matriz, cmap='YlGn', vmin=0)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(j, i, str(matriz[i, j]), ha='center', va='center',
                    color='white' if matriz[i, j] > matriz.max()*.65 else '#20392f')
    ax.set(xticks=range(len(CLASES)), xticklabels=CLASES, yticks=range(len(CLASES)),
           yticklabels=CLASES, xlabel='Predicción', ylabel='Etiqueta real', title='Matriz de confusión · test final sintético')
    fig.colorbar(im, ax=ax, label='Número de mensajes')
    guardar_figura(fig, carpeta / 'matriz_confusion')
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.plot([c['filas'] for c in seleccion['curva']], [c['macro_f1'] for c in seleccion['curva']],
            'o-', color='#305b49')
    ax.set(ylim=(0, 1), xlabel='Mensajes de entrenamiento (grupos completos)', ylabel='Macro-F1 de validación',
           title='Curva exploratoria · misma validación en cada punto')
    ax.grid(alpha=.2)
    guardar_figura(fig, carpeta / 'curva_aprendizaje')


def guardar_figura(figura, destino):
    """Guarda el mismo gráfico vectorial y rasterizado para informe o diapositivas."""
    figura.savefig(destino.with_suffix('.png'), dpi=160)
    figura.savefig(destino.with_suffix('.svg'))
    plt.close(figura)


def tabla(encabezados, filas):
    """Convierte datos en tabla Markdown sin dependencias de formato adicionales."""
    lineas = ['| ' + ' | '.join(encabezados) + ' |', '| ' + ' | '.join(['---']*len(encabezados)) + ' |']
    lineas += ['| ' + ' | '.join(str(v).replace('|', '/') for v in fila) + ' |' for fila in filas]
    return '\n'.join(lineas)


def generar_informe():
    """Genera informe completo y CSV de errores desde JSON; no toca la selección."""
    seleccion = leer_json(SALIDAS / 'seleccion.json')
    final = leer_json(SALIDAS / 'test_final.json')
    manifiesto = leer_json(DATOS / 'manifiesto.json')
    graficar(seleccion, final)
    m, s, intervalo = final['metricas'], final['selectivas'], final['intervalo_macro_f1']
    errores = [f for f in final['predicciones'] if f['etiqueta'] != f['predicha']]
    with (SALIDAS / 'errores_test.csv').open('w', encoding='utf-8', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=['id', 'grupo', 'texto', 'etiqueta', 'predicha', 'aceptada'])
        escritor.writeheader()
        escritor.writerows({k: f[k] for k in escritor.fieldnames} for f in errores)
    clases = tabla(['Clase', 'Precision', 'Recall', 'F1', 'Soporte'], [
        [c, f"{m['por_clase'][c]['precision']:.3f}", f"{m['por_clase'][c]['recall']:.3f}",
         f"{m['por_clase'][c]['f1-score']:.3f}", int(m['por_clase'][c]['support'])] for c in CLASES])
    comparacion = tabla(['Candidato', 'F1 train', 'F1 validación', 'Accuracy validación', 'Variables', 'Ajuste (s)'], [
        [c['nombre'], f"{c['train']['macro_f1']:.3f}", f"{c['validacion']['macro_f1']:.3f}",
         f"{c['validacion']['accuracy']:.3f}", c['vocabulario'], f"{c['segundos_ajuste']:.4f}"] for c in seleccion['candidatos']])
    particiones = tabla(['Partición']+CLASES+['Total'], [[p]+[v[c] for c in CLASES]+[sum(v.values())]
                                                           for p, v in manifiesto['conteos'].items()])
    curva = tabla(['Fracción de train', 'Mensajes', 'Grupos', 'F1 validación'], [[c['fraccion'], c['filas'],
                   c['grupos'], f"{c['macro_f1']:.3f}"] for c in seleccion['curva']])
    umbrales = tabla(['Umbral', 'Cobertura validación', 'Accuracy aceptados'], [[u['umbral'], f"{u['cobertura']:.3f}",
                       f"{u['accuracy_aceptados']:.3f}" if u['accuracy_aceptados'] is not None else 'N/A'] for u in seleccion['umbrales']])
    errores_tabla = tabla(['Texto sintético', 'Real', 'Predicha', 'Aceptada'], [[e['texto'], e['etiqueta'],
                             e['predicha'], e['aceptada']] for e in errores])
    matriz = tabla(['Real / predicha']+CLASES, [[c]+m['matriz'][i] for i, c in enumerate(CLASES)])
    accuracy_selectiva = f"{s['accuracy_aceptados']:.3f}" if s['accuracy_aceptados'] is not None else 'N/A'
    mejor = max(seleccion['candidatos'], key=lambda c: c['validacion']['macro_f1'])
    peor_clase = min(CLASES, key=lambda c: m['por_clase'][c]['recall'])
    peor_indice = CLASES.index(peor_clase)
    errores_aceptados = sum(e['aceptada'] for e in errores)
    informe = f'''# Ánimo · Informe de resultados

Este documento lo genera `entrenamiento/p06_informe.py` a partir de archivos ejecutados y congelados. Fecha de evaluación (UTC): {final['fecha_utc']}. No contiene cifras estimadas a mano. Las cifras redondeadas se presentan con tres decimales; los JSON conservan precisión completa.

## 1. Pregunta y alcance

¿Puede un clasificador sencillo distinguir cinco señales emocionales en mensajes breves en español, superando una referencia constante? Se implementó un chatbot académico local con respuestas redactadas, explicación de aportes de palabras, abstención y reglas de precaución independientes. La tarea es clasificar **texto**, no conocer el ánimo real ni identificar trastornos. «Ansiedad» es una etiqueta de preocupación textual.

## 2. Datos y validez

Corpus de **{manifiesto['n_filas']} mensajes**, **{manifiesto['n_grupos']} escenarios**, dos paráfrasis por escenario. **El 100% es sintético y fue redactado con asistencia de IA para este proyecto**; no hay conversaciones reales, consentimiento de participantes ni etiquetas de especialistas. Etiquetas asignadas por intención del texto, sin doble anotación ni acuerdo entre anotadores. Ningún modelo preentrenado forma parte del sistema entregado. La asistencia de IA en la autoría del corpus y código se declara, y no equivale a entrenamiento con datos humanos.

{particiones}

Semilla: {manifiesto['semilla']}. La división se hace por escenario y clase. Se rechazan duplicados normalizados antes de dividir y se congelan hashes de los archivos. Los grupos resuelven la fuga de las paráfrasis declaradas; no prueban independencia semántica completa entre distintos escenarios. El mismo proceso de autoría produce todas las particiones: se comparten estilo y vocabulario, por lo que los resultados pueden ser optimistas. El autor conoce el corpus; no es un test externo ciego.

## 3. Representación y selección

Se conservan negaciones. Se normalizan mayúsculas, tildes y signos, con tokenización de palabras. TF-IDF es una ampliación práctica de la bolsa de palabras mencionada en clase. El vocabulario y el IDF se aprenden **solo en train**, dentro de Pipeline. No hay embeddings, descargas en ejecución ni APIs externas. El modelo servido se mantiene entrenado solo con train: no se reajusta después de la selección.

{comparacion}

Ganador: **{seleccion['ganador']}**, por macro-F1 de validación sin redondear. Brecha F1 train-validación: **{mejor['train']['macro_f1']-mejor['validacion']['macro_f1']:.3f}**. Comparar conteos y TF-IDF observa el efecto de ponderar términos; comparar unigramas y bigramas observa el efecto de incorporar parejas de palabras. Son experimentos con una sola partición, no pruebas estadísticas concluyentes de superioridad. Los tiempos corresponden al equipo de ejecución, no son un benchmark general.

![Comparación](entrenamiento/resultados/figuras/comparacion.png)

## 4. Tamaño de entrenamiento

{curva}

Se usan subconjuntos anidados por grupo y balanceados, con la misma validación. Es un análisis exploratorio; no se vuelve a elegir el modelo por esta curva. No demuestra una ley de escalamiento ni sustituye validación cruzada repetida.

![Curva](entrenamiento/resultados/figuras/curva_aprendizaje.png)

## 5. Abstención

{umbrales}

Umbral elegido: **{seleccion['umbral']}**. Regla: {seleccion['motivo_umbral']}. Además, se exige margen >= 0.10 y al menos una característica conocida. Son heurísticas de producto, no calibración probabilística. Palabras conocidas en un mensaje ajeno al dominio aún pueden inducir respuestas equivocadas. No existe un detector universal de fuera de dominio.

## 6. Test final, utilizado una sola vez

El modelo, selección y código se congelaron antes de la reserva exclusiva del test. Se ejecutó una sola inferencia por lote sobre **{final['n']} mensajes**, guardando scores y predicciones. `test_reservado.json` registra la apertura. Repetir el comando lee el resultado existente. Los gráficos y el bootstrap reutilizan predicciones guardadas; no hay nueva selección ni ajuste tras el test.

- **Accuracy total: {m['accuracy']:.3f}**.
- **Macro-F1 total: {m['macro_f1']:.3f}**.
- Intervalo exploratorio percentil 95% de macro-F1 por escenario: **[{intervalo['inferior']:.3f}, {intervalo['superior']:.3f}]**, con **{intervalo['repeticiones']}** remuestreos. Considera la dependencia de las dos paráfrasis de cada grupo. No incorpora incertidumbre de la selección del modelo ni sesgo sintético.
- Con abstención: **cobertura {s['cobertura']:.3f}**, **{s['aceptados']} aceptados de {s['total']}** y **accuracy entre aceptados {accuracy_selectiva}**. Esta accuracy no debe confundirse con la del test completo.

{clases}

La debilidad más marcada es **{peor_clase}**: recall **{m['por_clase'][peor_clase]['recall']:.3f}**, con **{m['matriz'][peor_indice][peor_indice]} aciertos de {int(m['por_clase'][peor_clase]['support'])}**. La cifra global oculta esta diferencia. Además, **{errores_aceptados} errores pasaron la abstención**; pedir contexto no garantiza que las etiquetas mostradas sean correctas. La meta selectiva de validación no se debe anunciar como rendimiento garantizado en test.

Macro-F1 promedia las cinco F1 con igual peso. Accuracy cuenta aciertos/total. Precision por clase = TP/(TP+FP); recall = TP/(TP+FN); F1 es la media armónica. Una accuracy alta no basta para comprobar que todas las clases funcionen bien.

{matriz}

![Matriz de confusión](entrenamiento/resultados/figuras/matriz_confusion.png)

## 7. Errores observados, sin ocultarlos

**{len(errores)} errores** en el test completo. Se muestran todos, incluso los casos que el sistema rechaza por incertidumbre. `aceptada=True` significa que el chatbot habría mostrado la etiqueta errónea. Este análisis es posterior y no se utilizó para cambiar datos, parámetros ni modelo.

{errores_tabla}

Interpretación: los modelos de palabras aprenden asociaciones léxicas, no el significado completo. Un contexto compartido (exámenes, familia, trabajo) puede aparecer en emociones distintas. Los bigramas solo capturan orden local; no resuelven automáticamente negación, ironía, causas ni emociones mezcladas. Los errores listados permiten formular hipótesis para un futuro corpus nuevo, no justifican corregir este test y volverlo a medir.

## 8. Seguridad y privacidad

Una capa de reglas se ejecuta antes de ML y prioriza apoyo humano ante expresiones de suicidio o autolesión. No muestra score ni diagnóstico de riesgo. Incluye Línea 113, opción 5 (Perú), acompañamiento de alguien de confianza y atención de emergencias si hay peligro inmediato o daño ya ocurrido. No llama servicios ni puede determinar si alguien está a salvo.

Las reglas son conservadoras: también pueden activarse con negaciones, citas académicas, mensajes de terceros y frases ambiguas. Pueden fallar ante errores, eufemismos y lenguaje no contemplado. La batería de desarrollo en `tests/casos_seguridad.json` verifica comportamiento de software, **no sensibilidad ni especificidad clínica**. El recurso Línea 113 se consultó mediante la herramienta de líneas de ayuda; la opción 5 fue requisito explícito del proyecto. Referencia institucional: [MINSA](https://www.gob.pe/555?oai_link_source=model_response_hotline).

Sin persistencia de conversaciones en servidor ni navegador. No se aprende en línea. Solo se exporta un archivo si el usuario pulsa Exportar. Flask se enlaza a loopback y no se presenta como despliegue público. Cada mensaje es independiente; la conversación no tiene memoria semántica. Los logs HTTP contienen rutas y estados, sin cuerpos de mensajes.

## 9. Limitaciones y trabajo siguiente

La evaluación vale para este corpus pequeño, balanceado y sintético. No estima rendimiento en Perú, otros dialectos, grupos etarios, diagnósticos o conversaciones reales. No hay auditoría demográfica porque no existen esos atributos: no se afirma equidad. La distribución real probablemente será distinta; los scores no están calibrados. La normalización borra matices, signos y emojis y confunde ñ/n. El idioma esperado es español; no hay detector fiable de idioma. No se ha medido calidad terapéutica, satisfacción ni efecto sobre bienestar.

Una siguiente fase responsable requiere consentimiento, minimización de datos, dos anotadores independientes, revisión profesional de respuestas y evaluación externa con hablantes diversos. Debe crearse **otro test** antes de ajustar con los errores aquí observados. No reutilizar esta prueba como evidencia independiente. Una futura comparación podría incorporar validación cruzada por grupos, calibración y evaluación de fuera de dominio, solo si su beneficio compensa la complejidad.

## 10. Relación con los apuntes y reproducibilidad

Semanas 4 (flujo del proyecto), 7 y 9 (reglas y NLP), 11 (datasets, sesgo, logística y métricas), 12 (logística multinomial, validación/test y matriz de confusión). Se amplían TF-IDF, Pipeline, agrupación de paráfrasis, abstención y bootstrap, explicados en `docs/MAPA_CURSO.md`. No se usa la lematización preentrenada del laboratorio de semana 9 por la restricción del proyecto.

Entorno de ejecución: Python {seleccion['python']}, scikit-learn {seleccion['sklearn']}. Dependencias completas en `requirements-lock.txt`. Instrucciones experimentales en `docs/REPRODUCIBILIDAD.md`. Evidencia numérica: `seleccion.json`, `test_final.json`, `errores_test.csv`, figuras y hashes. Los números de este informe se reproducen a partir de esos archivos; volver a presentar el informe no constituye un nuevo test.

Fuentes técnicas: [extracción de texto de scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html), [prevención de fuga de datos](https://scikit-learn.org/stable/common_pitfalls.html), [Flask](https://flask.palletsprojects.com/en/stable/quickstart/). Los apuntes locales son la base académica principal.
'''
    (Path(__file__).resolve().parents[1] / 'INFORME_RESULTADOS.md').write_text(informe, encoding='utf-8')
