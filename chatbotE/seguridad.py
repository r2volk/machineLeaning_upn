# importa las busquedas por patrones de texto
import re

# importa la lectura limpieza y verificacion de datos
from entrenamiento.p01_datos import normalizar

# define las expresiones que activan una respuesta de precaucion
PATRONES = (
    # detecta menciones de suicidio o autolesion
    r"\b(?:suicid\w*|autoles\w*)\b",
    # detecta verbos de dano dirigidos a una persona
    r"\b(?:matarme|matarse|matarte|suicidarme|cortarme|cortandome|lastimarme|lesionarme)\b",
    # detecta frases de hacerse dano con una variante de escritura
    r"\b(?:hacerme|hacerte|hacerse|haciendome) (?:mucho )?da[n]?o\b",
    # detecta intencion de matar cortar o lastimar
    r"\b(?:me|se) (?:quiero|quiere|voy a|va a|pienso|esta pensando en) (?:matar|cortar|lastimar)\b",
    # detecta dano que se describe como ocurrido en el momento
    r"\b(?:me estoy|se esta) (?:cortando|lastimando|lesionando)\b",
    # detecta planes de hacerse dano
    r"\bme (?:voy a|quiero|pienso) hacer dano\b",
    # detecta el deseo de dejar de existir
    r"\b(?:quiero|quisiera) dejar de existir\b",
    # detecta relatos de cortes o lesiones ya ocurridos
    r"\bme (?:he cortado|he lastimado)\b",
    # detecta deseos de morir o desaparecer
    r"\b(?:quiero|quisiera|deseo|prefiero|pienso en) (?:morir|morirme|desaparecer)\b",
    # detecta expresiones de no querer seguir viviendo
    r"\bno (?:quiero|deseo|puedo) (?:seguir viviendo|vivir|seguir con vida|seguir)\b",
    # detecta referencias a quitarse o terminar con la vida
    r"\b(?:quitarme|quitarse|acabar con|terminar con) (?:la|mi|su) vida\b",
    # detecta deseos de no despertar o no existir
    r"\b(?:ojala|quisiera) (?:ya )?no (?:despertar|existir)\b",
    # detecta expresiones de ausencia como algo preferible
    r"\b(?:seria mejor|estar[ií]an mejor) (?:sin mi|si no estuviera|si no existiera)\b",
    # detecta frases de desesperanza sobre la vida
    r"\b(?:no vale la pena vivir|no tiene sentido vivir|no puedo mas con mi vida)\b",
    # detecta menciones de tomar una gran cantidad de pastillas
    r"\b(?:me tome|tome|he tomado) (?:muchas|demasiadas|todas las|un frasco de) pastillas\b",
    # detecta planes de saltar o tirarse desde algun lugar
    r"\b(?:quiero|voy a|pienso) (?:saltar|tirarme) (?:de|del|desde)\b",
    # detecta algunas expresiones explicitas en ingles
    r"\b(?:kill myself|hurt myself|end my life|want to die|self harm)\b",
)
# prepara los patrones para buscarlos en cada mensaje
REGLAS = [re.compile(p) for p in PATRONES]
# define el apoyo humano y los limites del prototipo
RESPUESTA_RIESGO = (
    "Gracias por contarlo. Lo que mencionas merece atención y apoyo humano. "
    "Si te está pasando a ti, no tienes que afrontarlo a solas: busca a una persona de confianza "
    "que pueda acompañarte y aléjate de cualquier cosa con la que puedas hacerte daño, si puedes hacerlo con seguridad. "
    "En Perú puedes llamar a la Línea 113, opción 5, para orientación en salud mental. "
    "Si ya te hiciste daño o hay peligro inmediato, acude a un servicio de emergencias o pide a alguien que te acompañe ahora. "
    "Si hablas de otra persona, ayúdala a contactar apoyo. Soy un prototipo académico: no puedo evaluar tu seguridad, "
    "dar un diagnóstico ni contactar ayuda por ti. ¿Estás en un lugar seguro y hay alguien a quien puedas llamar?"
)


# busca senales definidas por reglas de precaucion
def detectar_riesgo(texto):
    # normaliza el texto con las mismas reglas del proyecto
    limpio = normalizar(texto)

    # recorre dos variantes explicitas de escritura de una senal
    for origen, destino in [("su1cid", "suicid"), ("su!cid", "suicid")]:
        # busca la variante en el texto original sin distinguir mayusculas
        if origen in texto.casefold():
            # anade una senal que los patrones pueden reconocer
            limpio += " suicidio"
    # activa precaucion si al menos una regla coincide
    return any(regla.search(limpio) for regla in REGLAS)
