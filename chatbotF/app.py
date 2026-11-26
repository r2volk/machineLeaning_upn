# servidor web flask que sirve la interfaz y la api local del chatbot
import json
import os

import joblib
from flask import Flask, jsonify, render_template, request, send_file

from entrenamiento.p01_config import (
    CLASES,
    NOMBRES_CLASES,
    DOCUMENTACION,
    RUTA_ESTADO_MODELO,
    RUTA_MODELO_ACTIVO,
    RUTA_MODELO_BASE,
    RUTA_RESULTADOS,
)
from entrenamiento.p07_inferencia import analizar_mensaje
from entrenamiento.p09_reentrenamiento import (
    agregar_correccion,
    cargar_correcciones,
    cargar_estado_modelo,
    eliminar_correccion,
    limpiar_correcciones,
    reentrenar_con_correcciones,
    restaurar_modelo_base,
)


def cargar_modelo_disponible():
    # busca primero el modelo reentrenado por el usuario y si no existe usa el modelo base
    # ambos archivos se cargan desde disco con joblib para no reentrenar al abrir la web
    if RUTA_MODELO_ACTIVO.exists():
        return joblib.load(RUTA_MODELO_ACTIVO)
    if RUTA_MODELO_BASE.exists():
        return joblib.load(RUTA_MODELO_BASE)
    # si falta el modelo avisa que comando ejecutar para entrenarlo
    raise FileNotFoundError(
        "No existe el modelo entrenado. Ejecuta primero: python -m entrenamiento.p08_experimentos"
    )


def crear_app():
    # crea la aplicacion web de flask y deja el modelo cargado en memoria
    # usa un diccionario llamado contenedor para poder cambiar el modelo al reentrenar
    app = Flask(__name__, template_folder="templates")
    contenedor = {"modelo": cargar_modelo_disponible()}

    @app.route("/", methods=["GET"])
    def inicio():
        # muestra la pagina principal con las tres pestanas del chatbot
        return render_template("index.html")

    @app.get("/api/health")
    def estado_servidor():
        # responde al navegador si el servidor esta vivo y tiene un modelo listo
        return jsonify(modelo_disponible=contenedor["modelo"] is not None)

    @app.get("/documentos/<nombre>")
    def descargar_documento(nombre):
        # permite descargar solo el informe o la explicacion desde el laboratorio
        archivos = {"informe": "INFORME_RESULTADOS.md", "explicacion": "Explicacion.md"}
        # bloquea cualquier otro nombre para evitar descargas de archivos no permitidos
        if nombre not in archivos or not (DOCUMENTACION / archivos[nombre]).is_file():
            return jsonify(error="Documento no encontrado."), 404
        # envia el archivo markdown como descarga directa al navegador
        return send_file(DOCUMENTACION / archivos[nombre], as_attachment=True)

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        # recibe un mensaje del chat y devuelve la emocion, probabilidades o alerta de riesgo
        cuerpo = request.get_json(silent=True) or {}
        # limpia espacios del mensaje actual y del contexto del turno anterior si existe
        mensaje = (cuerpo.get("mensaje") or "").strip()
        contexto_previo = (cuerpo.get("contexto_previo") or "").strip() or None
        if not mensaje:
            # devuelve error 400 si el usuario envio una cadena vacia
            return jsonify({"error": "Escribe un mensaje para analizar."}), 400

        # ejecuta seguridad primero y luego el modelo de machine learning
        resultado = analizar_mensaje(
            contenedor["modelo"],
            mensaje,
            contexto_previo=contexto_previo,
        )
        # agrega la version activa del modelo para mostrarla en la barra superior
        estado = cargar_estado_modelo()
        resultado["version_modelo"] = estado.get("version_activa", "base")
        return jsonify(resultado)

    @app.route("/api/correcciones", methods=["GET"])
    def api_listar_correcciones():
        # entrega a la pestana aprendizaje los ejemplos guardados y el estado del modelo
        return jsonify(
            {
                "correcciones": cargar_correcciones(),
                "estado_modelo": cargar_estado_modelo(),
                "clases": CLASES,
                "nombres_clases": NOMBRES_CLASES,
            }
        )

    @app.route("/api/correcciones", methods=["POST"])
    def api_guardar_correccion():
        # guarda una frase etiquetada por el usuario bloqueando riesgo o fugas de validacion
        cuerpo = request.get_json(silent=True) or {}
        # lee texto y emocion desde el json enviado por el navegador
        texto = cuerpo.get("texto", "")
        etiqueta = cuerpo.get("etiqueta", "")
        try:
            # valida y escribe la nueva frase en el json local de correcciones
            guardado = agregar_correccion(texto, etiqueta)
        except ValueError as err:
            # si tiene riesgo, esta vacia o pertenece a validacion responde error 400
            return jsonify({"error": str(err)}), 400

        # responde 201 indicando que el ejemplo se guardo y envia la lista actualizada
        return jsonify(
            {
                "mensaje": "Ejemplo guardado localmente.",
                "correccion": guardado,
                "correcciones": cargar_correcciones(),
            }
        ), 201

    @app.route("/api/correcciones/<corr_id>", methods=["DELETE"])
    def api_eliminar_correccion(corr_id):
        # borra una sola frase guardada usando su identificador unico
        eliminado = eliminar_correccion(corr_id)
        if not eliminado:
            # avisa con 404 si ese identificador no existe en el archivo
            return jsonify({"error": "No se encontro esa correccion."}), 404
        return jsonify(
            {
                "mensaje": "Corrección eliminada.",
                "correcciones": cargar_correcciones(),
            }
        )

    @app.route("/api/correcciones", methods=["DELETE"])
    def api_limpiar_correcciones():
        # borra toda la lista local de ejemplos guardados por el usuario
        limpiar_correcciones()
        return jsonify(
            {
                "mensaje": "Se eliminaron todas las correcciones locales.",
                "correcciones": [],
            }
        )

    @app.route("/api/reentrenar", methods=["POST"])
    def api_reentrenar():
        # entrena una nueva version sumando las correcciones seguras y mide validacion
        try:
            nuevo_modelo, nuevo_estado = reentrenar_con_correcciones()
        except ValueError as err:
            # si no hay correcciones guardadas avisa con error 400
            return jsonify({"error": str(err)}), 400

        # reemplaza el modelo en memoria para que el chat use la nueva version al instante
        contenedor["modelo"] = nuevo_modelo
        return jsonify(
            {
                "mensaje": f"Modelo actualizado a {nuevo_estado['version_activa']}.",
                "estado_modelo": nuevo_estado,
            }
        )

    @app.route("/api/restaurar", methods=["POST"])
    def api_restaurar():
        # vuelve al modelo base congelado en disco y tambien en memoria
        try:
            modelo_base, estado_base = restaurar_modelo_base()
        except FileNotFoundError as err:
            # si falta el modelo base en disco avisa con error 500
            return jsonify({"error": str(err)}), 500

        # vuelve a dejar activo el modelo base original en la app web
        contenedor["modelo"] = modelo_base
        return jsonify(
            {
                "mensaje": "Se restauró el modelo base correctamente.",
                "estado_modelo": estado_base,
            }
        )

    @app.route("/api/laboratorio", methods=["GET"])
    def api_laboratorio():
        # lee el json de resultados de experimentos para llenar las tablas del laboratorio
        if not RUTA_RESULTADOS.exists():
            # si aun no se corrio p08_experimentos devuelve error 404
            return jsonify(
                {"error": "No existen resultados calculados en modelos/resultados_experimentos.json."}
            ), 404

        # abre el reporte json y lo envia junto al estado activo del modelo
        with RUTA_RESULTADOS.open("r", encoding="utf-8") as archivo:
            resultados = json.load(archivo)

        return jsonify(
            {
                "experimentos": resultados,
                "estado_modelo": cargar_estado_modelo(RUTA_ESTADO_MODELO),
                "nombres_clases": NOMBRES_CLASES,
            }
        )

    return app


def main():
    # lee host y puerto opcionales y arranca el servidor local en 127.0.0.1:5000
    host = os.environ.get("HOST", "127.0.0.1")
    puerto = int(os.environ.get("PORT", "5000"))
    url = f"http://{host}:{puerto}"

    # construye la aplicacion y muestra en consola la direccion para abrir el navegador
    app = crear_app()
    print("=" * 60)
    print("Chatbot académico de clasificación emocional (Proyecto ML)")
    print(f"Abre esta URL en tu navegador: {url}")
    print("=" * 60, flush=True)

    # corre sin reloader para no cargar el modelo dos veces en memoria
    app.run(host=host, port=puerto, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
