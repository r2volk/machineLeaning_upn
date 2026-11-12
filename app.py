# importa el generador de permisos aleatorios de la pagina
import secrets

# importa el manejo de rutas de archivos y carpetas
from pathlib import Path

# importa el servidor las peticiones y las respuestas web
from flask import Flask, g, jsonify, render_template, request, send_file

# importa el servidor que permite reservar el puerto local
from werkzeug.serving import make_server

# importa la eleccion de respuestas redactadas
from conversacion import responder

# importa las rutas constantes y utilidades compartidas
from entrenamiento.p00_configuracion import SALIDAS, leer_json

# importa la carga y el uso del modelo entrenado
from entrenamiento.p05_inferencia import cargar_modelo, predecir

# importa las reglas y la respuesta de precaucion
from seguridad import RESPUESTA_RIESGO, detectar_riesgo

# ubica la carpeta principal de la aplicacion
RAIZ = Path(__file__).resolve().parent


# prepara el servidor sus rutas y el modelo
def crear_app():
    # crea el servidor y usa la carpeta principal para la pagina
    app = Flask(__name__, template_folder=str(RAIZ), static_folder=None)
    # limita el tamano de las solicitudes y los dominios admitidos
    app.config.update(
        # limita los bytes del cuerpo de cada solicitud
        MAX_CONTENT_LENGTH=32_768,
        TRUSTED_HOSTS=["127.0.0.1", "localhost", "[::1]"],
    )
    # intenta cargar el modelo al crear la aplicacion
    try:
        # carga el modelo una vez para reutilizarlo en las peticiones
        app.config["ARTEFACTO"] = cargar_modelo()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        # registra la ausencia del modelo si falla la carga
        app.config["ARTEFACTO"] = None
        # avisa del problema y conserva las reglas de precaucion
        print(
            f"Modelo no disponible: {error}. La orientación de seguridad sigue activa.",
            flush=True,
        )

    # prepara el permiso del codigo propio de cada pagina
    @app.before_request
    def preparar_peticion():
        # crea un permiso aleatorio para el codigo propio de la pagina
        g.nonce = secrets.token_urlsafe(24)

    # anade las cabeceras de proteccion a cada respuesta
    @app.after_request
    def proteger_respuesta(respuesta):
        # evita guardar respuestas en la cache del navegador
        respuesta.headers["Cache-Control"] = "no-store"
        # impide interpretar los archivos como otro tipo de contenido
        respuesta.headers["X-Content-Type-Options"] = "nosniff"
        # evita enviar la direccion anterior como referencia
        respuesta.headers["Referrer-Policy"] = "no-referrer"
        # restringe los recursos y permite los scripts con el permiso propio
        respuesta.headers["Content-Security-Policy"] = (
            f"default-src 'self'; script-src 'nonce-{g.get('nonce', '')}'; "
            f"style-src 'nonce-{g.get('nonce', '')}'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        # devuelve la respuesta con sus cabeceras de proteccion
        return respuesta

    # atiende la ruta que muestra la pagina principal
    @app.get("/")
    def inicio():
        # entrega la pagina e inserta el permiso de sus scripts y estilos
        return render_template("index.html", nonce=g.nonce)

    # informa si el modelo esta disponible
    @app.get("/api/health")
    def salud():
        # comprueba si hay un modelo cargado en memoria
        activo = app.config["ARTEFACTO"] is not None
        # devuelve el estado y el codigo de disponibilidad
        return jsonify(
            # informa si el servicio esta disponible
            status="ok" if activo else "degradado",
            modelo_disponible=activo,
        ), (200 if activo else 503)

    # valida el mensaje y coordina precaucion o prediccion
    @app.post("/api/chat")
    def chat():
        # comprueba que la solicitud declare contenido de tipo json
        if not request.is_json:
            # rechaza un formato de entrada diferente
            return jsonify(error="Envía un objeto JSON."), 415
        # intenta convertir el cuerpo recibido en datos
        datos = request.get_json(silent=True)
        # comprueba que exista un objeto con un mensaje de texto
        if not isinstance(datos, dict) or not isinstance(datos.get("message"), str):
            # rechaza una estructura de mensaje invalida
            return jsonify(error="El campo message debe ser texto."), 400
        # retira espacios al inicio y al final del mensaje
        texto = datos["message"].strip()
        # comprueba si el texto quedo vacio
        if not texto:
            # pide escribir un mensaje antes de continuar
            return jsonify(error="Escribe un mensaje antes de enviarlo."), 400
        # lee el indicador de precaucion enviado por la interfaz
        contexto_seguridad = datos.get("safety_context", False)
        # comprueba que el indicador sea un valor booleano
        if type(contexto_seguridad) is not bool:
            # rechaza un indicador con tipo incorrecto
            return jsonify(error="safety_context debe ser booleano."), 400

        # prioriza una senal de riesgo o una alerta previa
        if detectar_riesgo(texto) or contexto_seguridad:
            # devuelve apoyo sin consultar el modelo emocional
            return jsonify(reply=RESPUESTA_RIESGO, safety=True, analysis=None)
        # comprueba el limite de longitud para el chat normal
        if len(texto) > 1500:
            # rechaza un mensaje demasiado largo
            return jsonify(error="El mensaje admite hasta 1500 caracteres."), 400
        # lee el numero de turno o usa el primero
        turno = datos.get("turn", 0)
        # comprueba el tipo y el rango del turno
        if type(turno) is not int or not 0 <= turno <= 10000:
            # rechaza un turno invalido
            return jsonify(error="El turno debe ser un entero entre 0 y 10000."), 400
        # recupera el modelo compartido por la aplicacion
        artefacto = app.config["ARTEFACTO"]
        # comprueba si la carga del modelo fallo
        if artefacto is None:
            # informa que no puede clasificar el mensaje
            return (
                jsonify(
                    error="El modelo no está disponible. Revisa la terminal y restaura los archivos de la entrega."
                ),
                503,
            )
        # clasifica el texto con el modelo congelado
        analisis = predecir(artefacto, texto)
        # devuelve la respuesta redactada y el analisis del mensaje
        return jsonify(
            # elige o entrega la respuesta de la conversacion
            reply=responder(analisis["etiqueta"], turno),
            # indica si se activa la precaucion
            safety=False,
            # entrega el analisis o lo omite por precaucion
            analysis=analisis,
        )

    # entrega solo la evidencia ya calculada
    @app.get("/api/resultados")
    def resultados():
        # lee los resultados existentes de seleccion
        seleccion = leer_json(SALIDAS / "seleccion.json")
        # lee los resultados existentes del test final
        final = leer_json(SALIDAS / "test_final.json")
        # envia los resumenes y omite las predicciones individuales
        return jsonify(
            # identifica el modelo que produjo el resultado
            modelo=seleccion["ganador"],
            # conserva el limite elegido para aceptar etiquetas
            umbral=seleccion["umbral"],
            # conserva los resultados de todos los candidatos
            candidatos=seleccion["candidatos"],
            # conserva los resultados al variar la cantidad de entrenamiento
            curva=seleccion["curva"],
            test={k: v for k, v in final.items() if k not in {"predicciones"}},
            # informa los conteos de las particiones del corpus
            datos=leer_json(RAIZ / "entrenamiento/datos/manifiesto.json")["conteos"],
        )

    # permite descargar un documento autorizado
    @app.get("/documentos/<nombre>")
    def documento(nombre):
        # limita las descargas a rutas fijas de documentos
        permitidos = {
            "informe": RAIZ / "INFORME_RESULTADOS.md",
            "protocolo": RAIZ / "entrenamiento/PROTOCOLO.md",
        }
        # comprueba si el documento solicitado esta permitido
        if nombre not in permitidos:
            # rechaza un documento no registrado
            return jsonify(error="Documento no encontrado."), 404
        # entrega el documento como archivo descargable
        return send_file(permitidos[nombre], as_attachment=True)

    # atiende el error de una solicitud demasiado grande
    @app.errorhandler(413)
    def demasiado_grande(error):
        # devuelve un error cuando el cuerpo supera el limite permitido
        return (
            jsonify(error="La solicitud es demasiado grande. Reduce el mensaje."),
            413,
        )

    # devuelve la aplicacion con sus rutas preparadas
    return app


# inicia el servidor local
def main():
    # crea la aplicacion antes de reservar el puerto
    aplicacion = crear_app()

    # prepara la variable que contendra el servidor
    servidor = None
    # prueba los puertos locales permitidos en orden
    for puerto in range(5000, 5011):
        # intenta reservar el puerto actual
        try:
            # crea un servidor local que atiende peticiones simultaneas
            servidor = make_server("127.0.0.1", puerto, aplicacion, threaded=True)
            # detiene la busqueda cuando logra reservar un puerto
            break
        except (OSError, SystemExit):
            # prueba el siguiente puerto si el actual falla
            continue
    # comprueba si todos los puertos fallaron
    if servidor is None:
        # termina con un mensaje si no pudo iniciar
        raise SystemExit("No se encontró un puerto libre entre 5000 y 5010.")
    # imprime la direccion exacta y como detener la aplicacion
    print(
        f"\nÁNIMO · Abre esta dirección: http://127.0.0.1:{servidor.server_port}\n"
        "Prototipo académico local · Ctrl+C para detener\n",
        flush=True,
    )
    # mantiene el servidor activo hasta que se solicite detenerlo
    try:
        # atiende las peticiones del navegador
        servidor.serve_forever()
    except KeyboardInterrupt:
        # avisa cuando se detiene desde la terminal
        print("\nÁnimo se ha detenido.")
    finally:
        # libera el puerto al terminar
        servidor.server_close()


# inicia el servidor solo al ejecutar este archivo directamente
if __name__ == "__main__":
    # ejecuta el arranque de la aplicacion
    main()
