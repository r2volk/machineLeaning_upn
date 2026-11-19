# guarda respuestas redactadas para cada etiqueta e incertidumbre
RESPUESTAS = {
    # respuestas para alegria
    "alegria": (
        (
            "Tu mensaje parece transmitir alegría. Qué bueno poder poner en "
            "palabras algo que te hace sentir bien. ¿Qué fue lo más bonito de ese "
            "momento?"
        ),
        (
            "Parece que hay algo que estás disfrutando. Si te apetece, cuéntame "
            "qué hizo especial tu día."
        ),
        (
            "Leo una señal de entusiasmo, aunque puedo equivocarme. ¿Hay alguien "
            "con quien te gustaría compartir esa buena noticia?"
        ),
    ),
    # respuestas para tristeza
    "tristeza": (
        (
            "Lo que escribes suena a tristeza. Gracias por compartirlo; no hace "
            "falta resolverlo todo ahora. ¿Qué parte de lo que pasó te está "
            "pesando más?"
        ),
        (
            "Parece que estás pasando por un momento difícil. Si te sirve, podemos "
            "ponerlo en palabras. ¿Qué te gustaría que alguien entendiera de cómo "
            "te sientes?"
        ),
        (
            "Tu mensaje tiene una señal de desánimo. Puedo equivocarme. ¿Hay "
            "alguien de confianza con quien te sentirías cómodo conversando?"
        ),
    ),
    # respuestas para enojo
    "enojo": (
        (
            "Tu mensaje parece expresar enojo o frustración. ¿Qué fue lo que más "
            "te molestó de esa situación?"
        ),
        (
            "Parece que algo te resultó injusto o frustrante. Si quieres, podemos "
            "separar lo que ocurrió de lo que necesitas ahora."
        ),
        (
            "Leo una señal de molestia. Una pausa antes de responder puede darte "
            "espacio. ¿Qué te gustaría poder decir con calma?"
        ),
    ),
    # respuestas para ansiedad
    "ansiedad": (
        (
            "Parece que hay preocupación o nervios en tu mensaje. Esto no es un "
            "diagnóstico. ¿Qué es lo que más te preocupa en este momento?"
        ),
        (
            "Suena a que tienes varias cosas en la cabeza. Si te apetece, podemos "
            "identificar una sola tarea pequeña para empezar."
        ),
        (
            "Leo una señal de tensión, aunque puedo equivocarme. ¿Qué parte de la "
            "situación está en tus manos hoy?"
        ),
    ),
    # respuestas para neutral
    "neutral": (
        (
            "Gracias por contarme. No encuentro una emoción clara en ese mensaje. "
            "¿Cómo te hizo sentir lo que describes?"
        ),
        (
            "Te leo. Parece una descripción de tu día. ¿Hay alguna emoción que "
            "quieras explorar a partir de eso?"
        ),
        "Podemos partir de ahí. ¿Cómo estás viviendo ese momento?",
    ),
    # respuestas para incierto
    "incierto": (
        (
            "No tengo suficiente claridad para interpretar la emoción de este "
            "mensaje. ¿Podrías contarme un poco más sobre cómo te sientes?"
        ),
        (
            "Podría estar interpretándolo de varias maneras. ¿Qué emoción describe "
            "mejor lo que estás viviendo?"
        ),
        (
            "Prefiero pedirte más contexto. ¿Esto te genera alegría, tristeza, "
            "enojo, preocupación u otra emoción?"
        ),
    ),
}


# elige una respuesta redactada segun etiqueta y turno
def responder(etiqueta, turno):
    # busca respuestas de la etiqueta o usa las de incertidumbre
    opciones = RESPUESTAS.get(etiqueta, RESPUESTAS["incierto"])
    # alterna las respuestas segun el numero de turno
    return opciones[turno % len(opciones)]
