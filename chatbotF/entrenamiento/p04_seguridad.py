# capa prioritaria de deteccion de posible riesgo de suicidio o autolesion
import re

from entrenamiento.p03_preprocesamiento import normalizar_texto

# frases cotidianas que usan morir o matar en sentido figurado sin riesgo
PATRONES_FALSO_POSITIVO = [
    re.compile(r"\bme muer[oa] de (risa|hambre|sue[nñ]o|frio|calor|ganas|verguenza|curiosidad)\b"),
    re.compile(r"\bmorir(me|se)? de (risa|hambre|sue[nñ]o|frio|calor|ganas|verguenza)\b"),
    re.compile(r"\bmatar el tiempo\b"),
    re.compile(r"\bmatando el tiempo\b"),
    re.compile(r"\bme mato de risa\b"),
]

# expresiones directas e indirectas de riesgo de suicidio o autolesion
PATRONES_RIESGO = [
    re.compile(r"\bsuicid\w*\b"),
    re.compile(r"\bautolesion\w*\b"),
    re.compile(r"\bquitar(me|se)? la vida\b"),
    re.compile(r"\bacabar con mi vida\b"),
    re.compile(r"\bterminar con mi vida\b"),
    re.compile(r"\bponer fin a mi vida\b"),
    re.compile(r"\bme quiero morir\b"),
    re.compile(r"\bquiero morir(me)?\b"),
    re.compile(r"\bdeseo morir(me)?\b"),
    re.compile(r"\bno quiero vivir\b"),
    re.compile(r"\bno quiero seguir viviendo\b"),
    re.compile(r"\bya no quiero vivir\b"),
    re.compile(r"\bno vale la pena vivir\b"),
    re.compile(r"\bprefiero estar muert[oa]\b"),
    re.compile(r"\bestaria mejor muert[oa]\b"),
    re.compile(r"\bme voy a matar\b"),
    re.compile(r"\bquiero matarme\b"),
    re.compile(r"\bpensando en matarme\b"),
    re.compile(r"\bhacerme da[nñ]o\b"),
    re.compile(r"\bme hago da[nñ]o\b"),
    re.compile(r"\bme estoy cortando\b"),
    re.compile(r"\bme voy a cortar\b"),
    re.compile(r"\bquiero cortarme\b"),
    re.compile(r"\bcortar(me)? las venas\b"),
    re.compile(r"\bdesaparecer para siempre\b"),
    re.compile(r"\bquiero dejar de existir\b"),
    re.compile(r"\bno quiero existir\b"),
    re.compile(r"\btomar pastillas para morir\b"),
    re.compile(r"\bsobredosis\b"),
]

# informacion oficial verificada de la linea 113 salud del minsa en peru
RECURSOS_AYUDA_PERU = {
    "linea_principal": "Línea 113 Salud (MINSA Perú) — Opción 5 (Salud mental y psicología)",
    "horario": "Gratuita, disponible las 24 horas del día, los 365 días del año desde cualquier celular o teléfono fijo en Perú.",
    "canales_alternativos": "WhatsApp y Telegram MINSA: 955 557 000 / 952 842 623 · Correo: infosalud@minsa.gob.pe",
    "emergencia": "Si hay peligro inmediato, llama al SAMU 106, a la Policía Nacional 105 o acude al establecimiento de salud más cercano.",
}


def evaluar_riesgo(texto):
    # evalua primero si el mensaje muestra senales de suicidio o autolesion
    normalizado = normalizar_texto(texto)
    if not normalizado:
        return {"es_riesgo": False, "motivo": None}

    for patron_fp in PATRONES_FALSO_POSITIVO:
        if patron_fp.search(normalizado):
            # si solo coincide con una expresion figurada y ningun otro patron de riesgo
            texto_sin_fp = patron_fp.sub(" ", normalizado)
            if not any(p.search(texto_sin_fp) for p in PATRONES_RIESGO):
                return {"es_riesgo": False, "motivo": None}

    for patron in PATRONES_RIESGO:
        coincidencia = patron.search(normalizado)
        if coincidencia:
            return {
                "es_riesgo": True,
                "motivo": coincidencia.group(0),
            }

    return {"es_riesgo": False, "motivo": None}


def construir_respuesta_crisis():
    # genera una respuesta empatica con apoyo humano y limites claros del sistema
    mensaje = (
        "Siento mucho que estés pasando por un momento tan difícil. No tienes que "
        "afrontar esto solo o sola: te recomiendo hablar ahora mismo con una persona "
        "de confianza o con un profesional de salud mental.\n\n"
        "En Perú puedes comunicarte gratis con la Línea 113 Salud del MINSA, "
        "Opción 5 (Salud mental y psicología), disponible las 24 horas todos los días "
        "del año. También puedes escribir por WhatsApp o Telegram al 955 557 000 o "
        "952 842 623, o al correo infosalud@minsa.gob.pe. Si estás en peligro inmediato, "
        "llama al 106 (SAMU), al 105 (Policía) o acude al centro de salud más cercano.\n\n"
        "Importante: este chatbot es un proyecto académico de Machine Learning y no es "
        "una herramienta clínica ni reemplaza la atención profesional."
    )
    return {
        "es_riesgo": True,
        "respuesta": mensaje,
        "recursos": RECURSOS_AYUDA_PERU,
    }
