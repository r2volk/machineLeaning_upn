# generador determinista del corpus de situaciones implicitas y frases cortas
import csv
import random
from pathlib import Path

from entrenamiento.p01_config import (
    CLASES,
    RUTA_CORPUS_IMPLICITO,
    RUTA_TRAIN,
    RUTA_VAL_GO,
    RUTA_VAL_SINTETICA,
    SEMILLA,
)
from entrenamiento.p02_datos import cargar_csv_etiquetado
from entrenamiento.p03_preprocesamiento import normalizar_texto

# genera 556 frases por cada emocion y reserva 60 cortas de chat diario
POR_CLASE_OBJETIVO = 556
CORTAS_OBJETIVO = 60

# expresiones de tiempo variadas para equilibrar palabras como ayer u hoy
TIEMPOS = [
    "Ayer",
    "Hoy",
    "Esta mañana",
    "Anoche",
    "Esta semana",
    "Hace un rato",
    "El lunes",
    "El viernes",
]

# cierres opcionales para multiplicar variantes cortas naturales
CIERRES_CORTOS = [
    "",
    "hoy",
    "otra vez",
]

# frases breves tipicas de chat por cada una de las cinco emociones
FRASES_CORTAS = {
    "enojo": [
        "que rabia me da esto",
        "me tiene harto este trafico",
        "que asco de servicio",
        "me da demasiada bronca",
        "no puede ser que fallen asi",
        "que coraje me da todo esto",
        "estoy harto de tantas excusas",
        "me sacan de quicio",
        "que injusticia tan grande",
        "me rompieron el celular",
        "se fue la luz justo ahora",
        "me dejaron esperando horas",
        "que mal servicio me dieron",
        "me molesta que no avisen",
        "ya me cansaron con lo mismo",
        "me da furia que mientan asi",
        "que pesimo trato recibi",
        "me robaron el telefono",
        "me robaron el celular",
        "me estafaron por internet",
        "que colera me da esta gente",
        "no respetan ni el horario",
        "me cortaron el internet",
        "me cobraron de mas en la cuenta",
        "que impotencia con esta situacion",
        "siempre hacen lo que les da la gana",
        "me arruinaron el dia con eso",
        "no soporto tanta irresponsabilidad",
        "me hartan con sus mentiras",
        "que fastidio tener que repetir todo",
    ],
    "tristeza": [
        "me fue horrible",
        "nadie me entiende",
        "me siento muy solo",
        "no tengo ganas de nada",
        "que mal dia tuve",
        "ojala mañana sea un mejor dia",
        "me siento vacio por dentro",
        "estoy muy desanimado",
        "todo me salio mal",
        "perdi mis llaves",
        "perdi mi billetera",
        "me duele mucho lo que paso",
        "ya me da igual todo",
        "extraño como eran las cosas antes",
        "me rompieron el corazon",
        "me dio mucha pena enterarme",
        "que tristeza ver esto asi",
        "me fue pesimo en el examen",
        "reprobe el curso",
        "me despidieron del trabajo",
        "siento un bajon enorme",
        "hoy fue un dia muy pesado",
        "me siento sin fuerzas",
        "me duele que nadie se acordara",
        "ojala mañana me vaya mejor",
        "estoy cansado de intentarlo sin exito",
        "me cuesta levantar el animo",
        "me quede solo toda la tarde",
        "perdi todo mi trabajo sin guardar",
        "que nostalgia siento al recordar eso",
    ],
    "ansiedad": [
        "ya no aguanto mas este estres",
        "tengo miedo de fallar",
        "estoy muy estresado",
        "me dan muchos nervios mañana",
        "no puedo dormir pensando en esto",
        "siento mucha presion encima",
        "me agobia tener tantos pendientes",
        "estoy demasiado tenso",
        "me tiemblan las manos",
        "tengo un nudo en el estomago",
        "no se si voy a lograrlo",
        "me preocupa demasiado el examen",
        "me duele la cabeza de tanto estres",
        "no me alcanza el tiempo para nada",
        "estoy super nervioso por la entrevista",
        "me inquieta no saber que pasara",
        "siento el corazon muy acelerado",
        "no dejo de darle vueltas a lo mismo",
        "me da panico quedarme en blanco",
        "estoy desesperado por las entregas",
        "siento mucha presion en el pecho",
        "no logro relajarme ni un minuto",
        "me estresa esperar los resultados",
        "tengo demasiada incertidumbre",
        "me agobia el tema del dinero",
        "estoy con los nervios de punta",
        "no se como voy a llegar a tiempo",
        "me angustia no tener respuesta",
        "siento que no voy a poder con todo",
        "estoy muy intranquilo por mañana",
    ],
    "alegria": [
        "aprobe el examen",
        "que buen dia tuve",
        "por fin termine todo",
        "me fue genial en la entrevista",
        "que alegria leer esto",
        "estoy super contento",
        "consegui el trabajo",
        "me dieron una gran noticia",
        "salio mejor de lo que esperaba",
        "que bonito dia pasamos",
        "logre terminar a tiempo",
        "me fue excelente",
        "estoy muy orgulloso del resultado",
        "que alivio tan grande siento",
        "por fin buenas noticias",
        "me siento genial",
        "ganamos el partido",
        "me subieron el sueldo",
        "quede seleccionado en las practicas",
        "me aprobaron el proyecto",
        "que felicidad me da esto",
        "todo salio perfecto",
        "me encantaron los resultados",
        "por fin lo logre",
        "que gran satisfaccion",
        "me fue de maravilla",
        "sali con una sonrisa enorme",
        "nos felicitaron por el trabajo",
        "recibi una noticia hermosa",
        "hoy todo fluyo super bien",
    ],
    "neutral": [
        "hola como estas",
        "todo bien gracias",
        "aqui pasando el rato",
        "estoy en clase ahora",
        "ya llegue a casa",
        "en un rato salgo a comprar",
        "mañana tengo clase temprano",
        "trabaje desde casa",
        "estoy revisando unos archivos",
        "el bus viene en camino",
        "aun no decido que almorzar",
        "estoy esperando un correo",
        "compre algunas cosas en la tienda",
        "tengo reunion a las tres",
        "solo pasaba a saludar",
        "todo tranquilo por aca",
        "nada nuevo por el momento",
        "estoy leyendo un documento",
        "deje el telefono cargando",
        "pague el recibo de luz",
        "estoy ordenando mis apuntes",
        "mañana debo ir al banco",
        "la clase empieza a las ocho",
        "estoy tomando un cafe",
        "fui a comprar pan temprano",
        "estoy actualizando la computadora",
        "no tengo clases en la tarde",
        "el archivo ya esta guardado",
        "estoy viendo el horario del ciclo",
        "todo sigue normal por aqui",
    ],
}

ESCENARIOS = {
    "enojo": [
        (
            ["me robaron el telefono", "me robaron el celular", "me robaron la billetera", "me robaron la mochila", "me robaron los audifonos", "me robaron la laptop"],
            ["en el paradero", "en el bus camino a la universidad", "al salir del trabajo", "en la esquina de mi casa", "cuando bajaba del carro", "en plena avenida"],
            ["y nadie hizo nada", "y encima se escaparon corriendo", "y me quede incomunicado", "y perdi todas mis cosas", "que impotencia tan grande", "con todo lo que me costo comprarlo"],
        ),
        (
            ["me estafaron con una compra", "me cobraron el doble en la cuenta", "me vendieron un equipo fallado", "me descontaron del sueldo sin aviso", "me bloquearon la tarjeta por error", "me subieron el alquiler de golpe"],
            ["por internet", "en la tienda del centro", "en el banco", "en la empresa", "en el servicio tecnico", "en la plataforma"],
            ["y ahora nadie responde", "y se negaron a devolverme el dinero", "y solo me dan excusas", "y me cortan la llamada cada vez que reclamo", "y no quieren aceptar el cambio", "y encima me trataron pesimo"],
        ),
        (
            ["llevo horas haciendo cola", "me dejaron plantado en la cita", "me cancelaron la atencion", "un tipo se colo en la fila", "el tecnico llego cuatro horas tarde", "me hicieron venir hasta la sede"],
            ["en la ventanilla", "en el consultorio", "en la oficina", "en la universidad", "en el paradero", "en la agencia"],
            ["y cerraron justo cuando me tocaba", "sin siquiera avisar con tiempo", "como si mi tiempo no valiera nada", "y dejo el problema exactamente igual", "para decirme que hoy no atienden", "sin pedir una sola disculpa"],
        ),
        (
            ["mi compañero no hizo su parte", "me culparon por un error ajeno", "me quitaron el credito del proyecto", "cambiaron las reglas a ultima hora", "se corto el internet justo al subir el examen", "el vecino puso musica a todo volumen"],
            ["en el trabajo grupal", "delante de toda la oficina", "cuando ya habiamos terminado", "a mitad de la entrega final", "hasta las cuatro de la madrugada", "en la reunion semanal"],
            ["y nos bajaron la nota a todos", "sin dejarme explicar lo que paso", "cuando yo estuve armandolo semanas", "y tuve que empezar todo otra vez", "y nadie respeto lo acordado", "y todavia se hacen los ofendidos"],
        ),
    ],
    "tristeza": [
        (
            ["ojala mañana sea un mejor dia", "espero que mañana sea un dia mejor", "ojala la proxima semana mejoren las cosas", "ojala pronto vuelva a tener animos", "solo espero que mañana sea distinto", "ojala algun dia deje de doler tanto"],
            ["porque hoy no me salio nada bien", "despues de una semana tan pesada", "porque siento que ya no me quedan fuerzas", "porque hoy todo me salio al reves", "porque llevo dias sintiendome decaido", "despues de tantas decepciones seguidas"],
            ["y solo quiero descansar", "y me cuesta seguir adelante", "y no tengo ganas de hablar con nadie", "y me pesa demasiado el animo", "y siento un vacio grande", "y ojala el animo me acompañe"],
        ),
        (
            ["me despidieron del trabajo", "me dijeron que no quede en la beca", "me rechazaron del puesto que queria", "cerraron el taller donde trabajaba", "cancelaron el proyecto de meses", "me quede fuera del equipo titular"],
            ["sin previo aviso", "despues de tanto esfuerzo", "justo cuando mas lo necesitaba", "pese a haberlo dado todo", "a ultimo momento", "al final del proceso"],
            ["y no se como contarle a mi familia", "y siento que todo mi esfuerzo no sirvio", "y me quede sin animos de nada", "y duele ver todo perdido", "y me cuesta aceptar lo que paso", "y siento que retrocedi otra vez"],
        ),
        (
            ["reprobe el curso otra vez", "jale el examen parcial", "no me alcanzo el puntaje por medio punto", "se cerro el archivo sin guardar", "perdi el informe de toda la semana", "perdi todas las fotos de mi telefono viejo"],
            ["a pesar de haber estudiado toda la semana", "justo antes de la entrega", "cuando pense que si lo lograba", "en el cierre del semestre", "despues de trasnochar tanto", "sin tener copia de respaldo"],
            ["y tendre que repetir todo el ciclo", "y me siento completamente agotado", "y me dio muchisima pena", "y no se de donde sacar fuerzas", "y me quede mirando la pantalla en silencio", "y me peso mas de lo que esperaba"],
        ),
        (
            ["fallecio mi perrito de toda la vida", "termino mi relacion de años", "despedi a mi mejor amigo en el aeropuerto", "nadie se acordo de mi cumpleaños", "extraño demasiado a mis abuelos", "mire fotos antiguas de mi familia"],
            ["y la casa se siente demasiado silenciosa", "y pase toda la tarde completamente solo", "porque se mudo a otro pais", "y me dio una nostalgia enorme", "y me dolio vernos tan lejos", "y recorde cuando estabamos todos juntos"],
            ["y no pude evitar llorar un rato", "y me cuesta aceptar que esa etapa termino", "y siento un vacio que no se llena", "y me siento muy desconectado de todos", "y cene solo mirando el telefono", "y hoy amaneci con el animo por el suelo"],
        ),
    ],
    "ansiedad": [
        (
            ["tengo que sustentar mi proyecto final", "me toca exponer frente a todo el salon", "tengo mi primera entrevista laboral", "debo presentar frente a los gerentes", "tengo el examen final del curso", "me llamaron de recursos humanos"],
            ["mañana a primera hora", "en menos de dos horas", "esta tarde", "el lunes temprano", "en unos minutos", "al mediodia"],
            ["y no logro conciliar el sueño", "y siento un nudo en el estomago", "y me tiemblan las manos", "y tengo el corazon acelerado", "y tengo miedo de quedarme en blanco", "y di vueltas en la cama toda la noche"],
        ),
        (
            ["todavia no publican las notas finales", "estoy esperando los resultados medicos", "no responden todavia mi postulacion", "me avisaron de una auditoria sorpresa", "escuchamos rumores de recortes en la empresa", "suena el telefono a cada rato"],
            ["desde el inicio de la semana", "desde temprano", "en el aula virtual", "en el correo", "afuera del consultorio", "en la oficina"],
            ["y actualizo la pagina cada cinco minutos", "y los minutos se me hacen eternos", "y no puedo dejar de darle vueltas", "y siento que puede ser una mala noticia", "y me sudan las manos al revisar", "y mi mente salta de un escenario negativo a otro"],
        ),
        (
            ["tengo tres entregas acumuladas", "me queda solo una hora para terminar", "el plazo cierra a medianoche", "el codigo sigue fallando antes de la demo", "no encuentro mis documentos del tramite", "mi vuelo sale pronto y el trafico no avanza"],
            ["y todavia me falta la mitad", "y el reloj corre demasiado rapido", "y siento que el tiempo no alcanza", "y cualquier pequeño error puede arruinarlo", "y el tramite vence hoy mismo", "y tengo demasiados pendientes sin cerrar"],
            ["y siento mucha presion en el pecho", "y tengo la mandibula tensa por el estres", "y no consigo relajarme ni un minuto", "y no he podido comer bien por los nervios", "y me agobia no llegar a tiempo", "y estoy muy intranquilo desde temprano"],
        ),
        (
            ["no se como voy a pagar la matricula", "se me juntaron varias deudas a fin de mes", "no se si llegare con el dinero del alquiler", "tengo que dar una noticia dificil en casa", "no se si tome la decision correcta", "tengo que manejar por primera vez en la autopista"],
            ["este mes", "sin saber si me depositan pronto", "porque todas las opciones tienen riesgo", "sin saber como van a reaccionar", "mientras revisaba los gastos", "con tanta incertidumbre alrededor"],
            ["y eso no me deja dormir tranquilo", "y me cuesta respirar con calma", "y me desperte a las tres de la madrugada pensando en eso", "y siento una inquietud constante", "y no pude concentrarme en nada mas", "y me da vueltas la cabeza sin parar"],
        ),
    ],
    "alegria": [
        (
            ["por fin quede seleccionado en las practicas", "aprobe todos los cursos del ciclo", "me avisaron que me subieron el sueldo", "me dieron la beca que tanto busque", "aceptaron mi articulo en la revista", "me aprobaron la visa para el intercambio"],
            ["despues de tantos meses de esfuerzo", "en el primer intento", "con el respaldo de todo el equipo", "justo cuando mas lo esperaba", "con felicitaciones del asesor", "sin correcciones mayores"],
            ["y toda mi familia lo celebro conmigo", "que gran satisfaccion siento hoy", "y sali con una sonrisa enorme", "y me siento muy orgulloso del resultado", "y pudimos brindar juntos por la noticia", "y todo el esfuerzo valio la pena"],
        ),
        (
            ["mi mama salio excelente de la operacion", "mi perro se recupero por completo", "mi hermana anuncio que tendra un bebe", "volvi a abrazar a mi hermano", "cene con toda mi familia reunida", "me sorprendieron con una fiesta de cumpleaños"],
            ["despues de tanto tiempo esperando", "en casa con todos juntos", "cuando pense que nadie vendria", "despues de dos años viviendo lejos", "con un clima precioso", "entre risas y abrazos"],
            ["y el medico dijo que todo esta perfecto", "y pasamos un dia inolvidable", "y me lleno el corazon de gratitud", "y nos reimos toda la tarde recordando anecdotas", "y fue el mejor momento del mes", "y me senti muy afortunado"],
        ),
        (
            ["por fin compre la laptop que ahorre", "pague la ultima cuota del prestamo", "encontre el departamento ideal", "obtuve mi licencia de conducir", "consegui entradas para el concierto", "adopte un cachorrito hermoso"],
            ["despues de ahorrar muchos meses", "cerca de mi trabajo y a buen precio", "que llevaba años esperando", "en el primer intento", "con mucha ilusion", "para empezar esta nueva etapa"],
            ["y ya estoy libre de esa deuda", "y lleno toda la casa de energia", "y no dejo de sonreir desde temprano", "y todo salio mucho mejor de lo que imaginaba", "y hoy todo fluyo de maravilla", "y me siento super motivado"],
        ),
        (
            ["el programa compilo y paso todas las pruebas", "terminamos la tesis a tiempo", "ganamos el campeonato universitario", "logre correr mis primeros diez kilometros", "el cliente quedo encantado con el diseño", "recupere el archivo que creia perdido"],
            ["en el ultimo minuto", "despues de tantas pruebas", "en equipo desde cero", "esta mañana entrenando", "antes de la presentacion final", "sin perder ningun dato"],
            ["y nos felicitaron delante de todos", "y nos recomendaron con otra empresa", "y mejore mi marca personal", "y nos dieron un reconocimiento especial", "y que alivio tan bonito verlo funcionando", "y quede muy satisfecho con lo logrado"],
        ),
    ],
    "neutral": [
        (
            ["fui al supermercado a comprar frutas", "deje mi telefono cargando en la mesa", "pague el recibo de luz por la aplicacion", "ordene mis cuadernos por curso", "lave la ropa y prepare la comida", "compre un cuaderno y dos lapiceros"],
            ["antes de salir de casa", "en la tienda de la esquina", "al volver de la universidad", "durante la tarde", "despues del almuerzo", "en el primer cajon del escritorio"],
            ["y regrese a las siete", "para tener todo listo mañana", "y luego segui con mis pendientes", "como hago todas las semanas", "antes de cerrar la oficina", "y guarde el comprobante en la carpeta"],
        ),
        (
            ["la clase de estadistica empieza a las ocho", "la reunion fue reprogramada para el jueves", "el autobus pasa cada quince minutos", "el banco atiende hasta las seis", "la farmacia abre todos los dias a las siete", "la lista de asistencia se cierra en cinco minutos"],
            ["en el aula tres del pabellon b", "por la avenida principal", "en la sede central", "segun el cronograma del ciclo", "en horario regular", "en la plataforma virtual"],
            ["segun el aviso publicado", "para todos los inscritos", "como figura en el calendario", "y dura aproximadamente una hora", "sin cambios adicionales", "de lunes a viernes"],
        ),
        (
            ["envie el correo con el informe adjunto", "actualice la version de python del equipo", "revise los apuntes de la semana doce", "hice una copia de seguridad de los archivos", "configure la impresora en red", "estoy comparando dos modelos de audifonos"],
            ["en formato pdf", "en el laboratorio de computo", "en un disco externo", "para organizar el indice", "en la carpeta de datos", "antes de decidir cual comprar"],
            ["y tiene diez paginas en total", "con tres graficos y dos tablas", "y corre en el puerto cinco mil", "mientras termina la descarga", "para la sesion de mañana", "con las cinco columnas requeridas"],
        ),
        (
            ["no estoy molesto, solo estoy concentrado", "no estoy triste, solo estoy pensando en la tarea", "no estoy asustado, solo leo las instrucciones", "no tengo clases durante la tarde", "no recuerdo si guarde la carpeta en documentos", "no tengo nada pendiente por ahora"],
            ["terminando esta tabla de datos", "ordenando mis horarios del mes", "con calma antes de empezar", "asi que ire a devolver dos libros", "o en el escritorio de la laptop", "esperando la siguiente clase"],
            ["y luego continuare leyendo el manual", "para dejar el trabajo organizado", "antes de salir al mediodia", "sin ningun apuro en particular", "como parte de la rutina diaria", "hasta que empiece la sesion"],
        ),
    ],
}

FRASES_EXCLUIDAS = {
    "no me fue del todo bien",
    "no me fue del todo bien en el examen",
    "no puedo estar mas feliz con esta noticia",
    "no me lo puedo creer al fin aprobe el curso",
    "no tengo clases mañana y luego ire al banco",
    "no se si la reunion empieza a las cuatro o a las cinco",
    "no me preocupa en absoluto todo salio genial",
    "no estoy triste solo estoy pensando en mis planes",
    "no estoy enojado solo cansado",
    "bueno supongo que asi son las cosas",
    "no se que va a pasar mañana tengo el pecho apretado",
    "no dormi nada pensando en la presentacion de mañana",
    "me da mucha rabia que no cumplan lo acordado",
    "hoy me siento vacio y sin ganas de salir",
    "que bueno que pudimos terminar el proyecto a tiempo",
    "no esperaba aprobar pero saque veinte",
}


def cargar_textos_prohibidos():
    # reune textos de train y validacion para impedir duplicados o fugas
    prohibidos = set(FRASES_EXCLUIDAS)
    for ruta in (RUTA_TRAIN, RUTA_VAL_GO, RUTA_VAL_SINTETICA):
        if Path(ruta).exists():
            # normaliza cada frase existente para detectarla aunque cambien tildes o mayusculas
            for fila in cargar_csv_etiquetado(ruta):
                prohibidos.add(normalizar_texto(fila["texto"]))
    return prohibidos


def generar_cortas_clase(clase, prohibidos, rng):
    # arma variantes breves y coloquiales de chat sin repetir textos
    pool = []
    vistos = set()
    for base in FRASES_CORTAS[clase]:
        for cierre in CIERRES_CORTOS:
            candidato = f"{base.capitalize()} {cierre}".strip()
            norm = normalizar_texto(candidato)
            # acepta solo si no existe en train o validacion y no se repitio antes
            if norm and norm not in prohibidos and norm not in vistos:
                vistos.add(norm)
                pool.append(candidato)
    # mezcla el orden usando el generador con semilla fija
    rng.shuffle(pool)
    return pool


def generar_largas_clase(clase, prohibidos, rng):
    # genera combinaciones situacionales completas sin cruzar validacion
    pool = []
    vistos = set()
    for inicios, contextos, cierres in ESCENARIOS[clase]:
        for inicio in inicios:
            for contexto in contextos:
                for cierre in cierres:
                    # crea una version directa y otra empezando con expresion de tiempo
                    texto_a = f"{inicio.capitalize()} {contexto} {cierre}"
                    tiempo = rng.choice(TIEMPOS)
                    texto_b = f"{tiempo} {inicio} {contexto} {cierre}"
                    for candidato in (texto_a, texto_b):
                        norm = normalizar_texto(candidato)
                        if norm and norm not in prohibidos and norm not in vistos:
                            vistos.add(norm)
                            pool.append(candidato)
    rng.shuffle(pool)
    return pool


def construir_corpus_implicito(ruta_salida=RUTA_CORPUS_IMPLICITO):
    # escribe exactamente 2780 filas mezclando frases cortas y situacionales
    rng = random.Random(SEMILLA)
    prohibidos = cargar_textos_prohibidos()
    filas = []

    for clase in CLASES:
        # toma primero hasta 60 frases cortas de chat para esa emocion
        cortas = generar_cortas_clase(clase, prohibidos, rng)[:CORTAS_OBJETIVO]
        for texto in cortas:
            prohibidos.add(normalizar_texto(texto))

        # completa hasta llegar a 556 frases por emocion con escenarios detallados
        faltan = POR_CLASE_OBJETIVO - len(cortas)
        largas = generar_largas_clase(clase, prohibidos, rng)
        if len(largas) < faltan:
            raise RuntimeError(
                f"Faltan candidatas para {clase}: {len(largas)} < {faltan}"
            )
        seleccion = cortas + largas[:faltan]
        for idx, texto in enumerate(seleccion, start=1):
            prohibidos.add(normalizar_texto(texto))
            # asigna grupo unico y origen curado para mantener trazabilidad
            filas.append(
                {
                    "texto": texto,
                    "etiqueta": clase,
                    "grupo": f"imp_{clase}_{idx:04d}",
                    "origen": "curado_situaciones_implicitas",
                }
            )

    ruta_path = Path(ruta_salida)
    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    # guarda todo en un archivo csv nuevo con las cuatro columnas estandar
    with ruta_path.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=["texto", "etiqueta", "grupo", "origen"]
        )
        escritor.writeheader()
        escritor.writerows(filas)

    return len(filas)


if __name__ == "__main__":
    total = construir_corpus_implicito()
    print(f"Corpus implicito generado con {total} filas en: {RUTA_CORPUS_IMPLICITO}")
