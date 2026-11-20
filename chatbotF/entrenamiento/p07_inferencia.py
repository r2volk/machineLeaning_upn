# inferencia del chatbot con prioridad de seguridad, abstencion y explicacion
import numpy as np

from entrenamiento.p01_config import (
    NOMBRES_CLASES,
    UMBRAL_MARGEN_MINIMO,
    UMBRAL_PROB_MINIMA,
)
from entrenamiento.p03_preprocesamiento import preparar_texto_modelo
from entrenamiento.p04_seguridad import construir_respuesta_crisis, evaluar_riesgo

# respuestas breves y empaticas segun la senal detectada
RESPUESTAS_EMOCION = {
    "alegria": (
        "Percibo señales de alegría o entusiasmo en tu mensaje. "
        "Me da gusto leer eso; si quieres contarme qué salió bien, te escucho."
    ),
    "tristeza": (
        "Percibo señales de tristeza o desánimo en lo que compartes. "
        "Lamento que te sientas así; si deseas contar un poco más de lo que pasó, puedes escribirlo aquí."
    ),
    "enojo": (
        "Percibo señales de enojo o frustración en tu mensaje. "
        "Se nota que la situación te molestó bastante; si quieres explicarme qué ocurrió, te leo."
    ),
    "ansiedad": (
        "Percibo señales de ansiedad o nerviosismo en tu mensaje. "
        "Cuando hay mucha tensión ayuda ordenar una cosa a la vez; ¿qué es lo que más te inquieta ahora?"
    ),
    "neutral": (
        "Tu mensaje suena mayormente neutral o informativo. "
        "Si había una emoción detrás que no quedó tan explícita, puedes darme un poco más de contexto."
    ),
}


def extraer_terminos_influyentes(modelo, texto_limpio, clase_objetivo, limite=5):
    # calcula que palabras y subpalabras del mensaje sumaron mas peso a la emocion
    # multiplica el valor tf-idf de cada termino por su peso en la regresion logistica
    if not hasattr(modelo, "named_steps"):
        return []

    vectorizador = modelo.named_steps.get("vectorizador")
    clasificador = modelo.named_steps.get("clasificador")
    if vectorizador is None or clasificador is None or not hasattr(clasificador, "coef_"):
        return []

    # transforma solo este mensaje y busca las columnas que quedaron distintas de cero
    matriz = vectorizador.transform([texto_limpio])
    indices_activos = matriz.nonzero()[1]
    if len(indices_activos) == 0:
        return []

    nombres = vectorizador.get_feature_names_out()
    clases = list(clasificador.classes_)
    if clase_objetivo not in clases:
        return []

    # recorre cada termino activo y separa palabras completas de subpalabras
    fila_pesos = clasificador.coef_[clases.index(clase_objetivo)]
    aportes_palabras = []
    aportes_subpalabras = []
    for idx in indices_activos:
        valor_tfidf = float(matriz[0, idx])
        peso = float(fila_pesos[idx])
        aporte = valor_tfidf * peso
        if aporte <= 0:
            continue

        nombre = str(nombres[idx])
        if nombre.startswith("caracteres__"):
            # etiqueta los fragmentos de caracteres con sub para leerlos facil en la web
            limpio = nombre.removeprefix("caracteres__").strip()
            if limpio:
                aportes_subpalabras.append(
                    {"termino": f"sub:{limpio}", "aporte": round(aporte, 4)}
                )
        else:
            limpio = nombre.removeprefix("palabras__")
            aportes_palabras.append(
                {"termino": limpio, "aporte": round(aporte, 4)}
            )

    # muestra primero las palabras completas mas fuertes y rellena con subpalabras
    aportes_palabras.sort(key=lambda item: item["aporte"], reverse=True)
    if len(aportes_palabras) >= limite:
        return aportes_palabras[:limite]

    aportes_subpalabras.sort(key=lambda item: item["aporte"], reverse=True)
    return (aportes_palabras + aportes_subpalabras)[:limite]


def analizar_mensaje(
    modelo,
    texto_original,
    umbral_prob=UMBRAL_PROB_MINIMA,
    umbral_margen=UMBRAL_MARGEN_MINIMO,
    contexto_previo=None,
):
    # analiza un mensaje en vivo: seguridad primero, luego contexto y probabilidades
    # decide si responder con una emocion o pedir mas contexto cuando hay duda
    riesgo = evaluar_riesgo(texto_original)
    if riesgo["es_riesgo"]:
        crisis = construir_respuesta_crisis()
        return {
            "es_riesgo": True,
            "motivo_riesgo": riesgo["motivo"],
            "estado": "riesgo",
            "etiqueta": None,
            "nombre_etiqueta": "Atención prioritaria",
            "pide_contexto": False,
            "uso_contexto": False,
            "confianza": None,
            "margen": None,
            "probabilidades": {},
            "terminos_clave": [],
            "respuesta": crisis["respuesta"],
            "recursos": crisis["recursos"],
        }

    # si el turno previo pidio contexto y era seguro combina ambos mensajes
    contexto_limpio = (contexto_previo or "").strip()
    uso_contexto = bool(contexto_limpio) and not evaluar_riesgo(contexto_limpio)["es_riesgo"]
    texto_entrada = f"{contexto_limpio} {texto_original}".strip() if uso_contexto else texto_original

    # limpia el texto y marca negaciones antes de pasarlo al clasificador
    texto_limpio = preparar_texto_modelo(texto_entrada)
    if not texto_limpio:
        return {
            "es_riesgo": False,
            "estado": "vacio",
            "etiqueta": None,
            "nombre_etiqueta": "Sin texto suficiente",
            "pide_contexto": True,
            "uso_contexto": False,
            "confianza": 0.0,
            "margen": 0.0,
            "probabilidades": {},
            "terminos_clave": [],
            "respuesta": (
                "No encontré palabras suficientes para analizar tu mensaje. "
                "¿Podrías contarme con una frase un poco más completa cómo te sientes?"
            ),
            "recursos": None,
        }

    # calcula las probabilidades de las 5 emociones y las ordena de mayor a menor
    probs = modelo.predict_proba([texto_limpio])[0]
    clases = list(modelo.classes_)
    orden = np.argsort(probs)[::-1]

    mejor_clase = str(clases[orden[0]])
    segunda_clase = str(clases[orden[1]])
    p1 = float(probs[orden[0]])
    p2 = float(probs[orden[1]])
    margen = p1 - p2

    mapa_probs = {
        str(clases[i]): round(float(probs[i]), 4)
        for i in orden
    }

    # obtiene las palabras que mas inclinaron la prediccion hacia la primera emocion
    terminos = extraer_terminos_influyentes(modelo, texto_limpio, mejor_clase)

    # si la confianza o la ventaja sobre la segunda emocion son bajas pide contexto
    if p1 < umbral_prob or margen < umbral_margen:
        nombre_1 = NOMBRES_CLASES.get(mejor_clase, mejor_clase)
        nombre_2 = NOMBRES_CLASES.get(segunda_clase, segunda_clase)
        respuesta = (
            f"La señal emocional de tu mensaje no es lo bastante clara "
            f"(veo indicios cercanos entre {nombre_1.lower()} y {nombre_2.lower()}). "
            "¿Podrías contarme un poco más de contexto sobre cómo te sientes?"
        )
        return {
            "es_riesgo": False,
            "estado": "pedir_contexto",
            "etiqueta": None,
            "etiqueta_tentativa": mejor_clase,
            "segunda_etiqueta": segunda_clase,
            "nombre_etiqueta": "Pide más contexto",
            "pide_contexto": True,
            "uso_contexto": uso_contexto,
            "confianza": round(p1, 4),
            "margen": round(margen, 4),
            "probabilidades": mapa_probs,
            "terminos_clave": terminos,
            "respuesta": respuesta,
            "recursos": None,
        }

    return {
        "es_riesgo": False,
        "estado": "clasificado",
        "etiqueta": mejor_clase,
        "etiqueta_tentativa": mejor_clase,
        "segunda_etiqueta": segunda_clase,
        "nombre_etiqueta": NOMBRES_CLASES.get(mejor_clase, mejor_clase),
        "pide_contexto": False,
        "uso_contexto": uso_contexto,
        "confianza": round(p1, 4),
        "margen": round(margen, 4),
        "probabilidades": mapa_probs,
        "terminos_clave": terminos,
        "respuesta": RESPUESTAS_EMOCION[mejor_clase],
        "recursos": None,
    }
