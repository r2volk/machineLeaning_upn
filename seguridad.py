"""Reglas de precaución independientes del ML; no estiman riesgo clínico."""
import re
from entrenamiento.p01_datos import normalizar

# Conservadoras: también responden a negaciones, citas y menciones de terceros.
# No se intenta decidir si alguien está «a salvo» mediante una expresión regular.
PATRONES = (
    r'\b(?:suicid\w*|autoles\w*)\b',
    r'\b(?:matarme|matarse|matarte|suicidarme|cortarme|cortandome|lastimarme|lesionarme)\b',
    r'\b(?:hacerme|hacerte|hacerse|haciendome) (?:mucho )?da[n]?o\b',
    r'\b(?:me|se) (?:quiero|quiere|voy a|va a|pienso|esta pensando en) (?:matar|cortar|lastimar)\b',
    r'\b(?:me estoy|se esta) (?:cortando|lastimando|lesionando)\b',
    r'\bme (?:voy a|quiero|pienso) hacer dano\b',
    r'\b(?:quiero|quisiera) dejar de existir\b',
    r'\bme (?:he cortado|he lastimado)\b',
    r'\b(?:quiero|quisiera|deseo|prefiero|pienso en) (?:morir|morirme|desaparecer)\b',
    r'\bno (?:quiero|deseo|puedo) (?:seguir viviendo|vivir|seguir con vida|seguir)\b',
    r'\b(?:quitarme|quitarse|acabar con|terminar con) (?:la|mi|su) vida\b',
    r'\b(?:ojala|quisiera) (?:ya )?no (?:despertar|existir)\b',
    r'\b(?:seria mejor|estar[ií]an mejor) (?:sin mi|si no estuviera|si no existiera)\b',
    r'\b(?:no vale la pena vivir|no tiene sentido vivir|no puedo mas con mi vida)\b',
    r'\b(?:me tome|tome|he tomado) (?:muchas|demasiadas|todas las|un frasco de) pastillas\b',
    r'\b(?:quiero|voy a|pienso) (?:saltar|tirarme) (?:de|del|desde)\b',
    r'\b(?:kill myself|hurt myself|end my life|want to die|self harm)\b',
)
REGLAS = [re.compile(p) for p in PATRONES]
RESPUESTA_RIESGO = (
    'Gracias por contarlo. Lo que mencionas merece atención y apoyo humano. '
    'Si te está pasando a ti, no tienes que afrontarlo a solas: busca a una persona de confianza '
    'que pueda acompañarte y aléjate de cualquier cosa con la que puedas hacerte daño, si puedes hacerlo con seguridad. '
    'En Perú puedes llamar a la Línea 113, opción 5, para orientación en salud mental. '
    'Si ya te hiciste daño o hay peligro inmediato, acude a un servicio de emergencias o pide a alguien que te acompañe ahora. '
    'Si hablas de otra persona, ayúdala a contactar apoyo. Soy un prototipo académico: no puedo evaluar tu seguridad, '
    'dar un diagnóstico ni contactar ayuda por ti. ¿Estás en un lugar seguro y hay alguien a quien puedas llamar?'
)


def detectar_riesgo(texto):
    """Devuelve precaución si encuentra una señal; ausencia de coincidencia NO significa seguridad."""
    limpio = normalizar(texto)
    # Variantes sencillas frecuentes; no es un detector general de ofuscaciones.
    for origen, destino in [('su1cid', 'suicid'), ('su!cid', 'suicid')]:
        if origen in texto.casefold():
            limpio += ' suicidio'
    return any(regla.search(limpio) for regla in REGLAS)
