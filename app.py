"""Servidor local de Ánimo. La lógica de ML se importa exclusivamente de entrenamiento/."""
import secrets
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request, send_file
from werkzeug.serving import make_server

from conversacion import responder
from entrenamiento.p00_configuracion import SALIDAS, leer_json
from entrenamiento.p05_inferencia import cargar_modelo, predecir
from seguridad import RESPUESTA_RIESGO, detectar_riesgo

RAIZ = Path(__file__).resolve().parent


def crear_app():
    """Configura Flask, carga el modelo una vez y registra endpoints sin estado persistente."""
    app = Flask(__name__, template_folder=str(RAIZ), static_folder=None)
    app.config.update(MAX_CONTENT_LENGTH=32_768, TRUSTED_HOSTS=['127.0.0.1', 'localhost', '[::1]'])
    try:
        app.config['ARTEFACTO'] = cargar_modelo()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        app.config['ARTEFACTO'] = None
        print(f'Modelo no disponible: {error}. La orientación de seguridad sigue activa.', flush=True)

    @app.before_request
    def preparar_peticion():
        """Nonce por respuesta para permitir solo el JS/CSS propio del único HTML."""
        g.nonce = secrets.token_urlsafe(24)

    @app.after_request
    def proteger_respuesta(respuesta):
        """Evita caché de mensajes, recursos externos, framing y detección insegura de MIME."""
        respuesta.headers['Cache-Control'] = 'no-store'
        respuesta.headers['X-Content-Type-Options'] = 'nosniff'
        respuesta.headers['Referrer-Policy'] = 'no-referrer'
        respuesta.headers['Content-Security-Policy'] = (
            f"default-src 'self'; script-src 'nonce-{g.get('nonce', '')}'; "
            f"style-src 'nonce-{g.get('nonce', '')}'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        return respuesta

    @app.get('/')
    def inicio():
        """Sirve index.html desde Flask: no requiere abrir archivos manualmente."""
        return render_template('index.html', nonce=g.nonce)

    @app.get('/api/health')
    def salud():
        """Permite comprobar si el modelo está cargado sin exponer mensajes."""
        activo = app.config['ARTEFACTO'] is not None
        return jsonify(status='ok' if activo else 'degradado', modelo_disponible=activo), 200 if activo else 503

    @app.post('/api/chat')
    def chat():
        """Valida entrada; prioriza seguridad; ejecuta inferencia y respuesta redactada."""
        if not request.is_json:
            return jsonify(error='Envía un objeto JSON.'), 415
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict) or not isinstance(datos.get('message'), str):
            return jsonify(error='El campo message debe ser texto.'), 400
        texto = datos['message'].strip()
        if not texto:
            return jsonify(error='Escribe un mensaje antes de enviarlo.'), 400
        contexto_seguridad = datos.get('safety_context', False)
        if type(contexto_seguridad) is not bool:
            return jsonify(error='safety_context debe ser booleano.'), 400
        # Se revisa riesgo incluso en textos por encima del límite de conversación.
        if detectar_riesgo(texto) or contexto_seguridad:
            return jsonify(reply=RESPUESTA_RIESGO, safety=True, analysis=None)
        if len(texto) > 1500:
            return jsonify(error='El mensaje admite hasta 1500 caracteres.'), 400
        turno = datos.get('turn', 0)
        if type(turno) is not int or not 0 <= turno <= 10000:
            return jsonify(error='El turno debe ser un entero entre 0 y 10000.'), 400
        artefacto = app.config['ARTEFACTO']
        if artefacto is None:
            return jsonify(error='El modelo no está disponible. Revisa la terminal y restaura los archivos de la entrega.'), 503
        analisis = predecir(artefacto, texto)
        return jsonify(reply=responder(analisis['etiqueta'], turno), safety=False, analysis=analisis)

    @app.get('/api/resultados')
    def resultados():
        """Expone exclusivamente métricas ya guardadas: jamás evalúa desde la web."""
        seleccion = leer_json(SALIDAS / 'seleccion.json')
        final = leer_json(SALIDAS / 'test_final.json')
        return jsonify(modelo=seleccion['ganador'], umbral=seleccion['umbral'],
                       candidatos=seleccion['candidatos'], curva=seleccion['curva'],
                       test={k: v for k, v in final.items() if k not in {'predicciones'}},
                       datos=leer_json(RAIZ / 'entrenamiento/datos/manifiesto.json')['conteos'])

    @app.get('/documentos/<nombre>')
    def documento(nombre):
        """Descarga solo documentos permitidos; ninguna ruta arbitraria ni artefactos pickle."""
        permitidos = {'informe': RAIZ / 'INFORME_RESULTADOS.md',
                      'protocolo': RAIZ / 'entrenamiento/PROTOCOLO.md'}
        if nombre not in permitidos:
            return jsonify(error='Documento no encontrado.'), 404
        return send_file(permitidos[nombre], as_attachment=True)

    @app.errorhandler(413)
    def demasiado_grande(error):
        """Evita procesar cuerpos excesivos y devuelve un error legible por la interfaz."""
        return jsonify(error='La solicitud es demasiado grande. Reduce el mensaje.'), 413

    return app


def main():
    """Escoge un puerto local libre y anuncia la dirección después de reservarlo."""
    aplicacion = crear_app()
    # macOS puede ocupar 5000 con AirPlay. Se elige automáticamente otro puerto.
    servidor = None
    for puerto in range(5000, 5011):
        try:
            servidor = make_server('127.0.0.1', puerto, aplicacion, threaded=True)
            break
        except (OSError, SystemExit):
            continue
    if servidor is None:
        raise SystemExit('No se encontró un puerto libre entre 5000 y 5010.')
    print(f'\nÁNIMO · Abre esta dirección: http://127.0.0.1:{servidor.server_port}\n'
          'Prototipo académico local · Ctrl+C para detener\n', flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print('\nÁnimo se ha detenido.')
    finally:
        servidor.server_close()


if __name__ == '__main__':
    main()
