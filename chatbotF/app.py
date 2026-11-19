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
    # carga el modelo activo o usa el modelo base si todavia no hay reentrenado
    if RUTA_MODELO_ACTIVO.exists():
        return joblib.load(RUTA_MODELO_ACTIVO)
    if RUTA_MODELO_BASE.exists():
        return joblib.load(RUTA_MODELO_BASE)
    raise FileNotFoundError(
        "No existe el modelo entrenado. Ejecuta primero: python -m entrenamiento.p08_experimentos"
    )


def crear_app():
    # crea la instancia de flask y mantiene el modelo cargado en memoria
    app = Flask(__name__, template_folder="templates")
    contenedor = {"modelo": cargar_modelo_disponible()}

    @app.route("/", methods=["GET"])
    def inicio():
        # entrega la interfaz completa en un unico archivo index.html
        return render_template("index.html")

    @app.get("/api/health")
    def estado_servidor():
        # informa si el modelo esta cargado en memoria
        return jsonify(modelo_disponible=contenedor["modelo"] is not None)

    @app.get("/documentos/<nombre>")
    def descargar_documento(nombre):
        # restringe las descargas a los documentos que muestra el laboratorio
        archivos = {"informe": "INFORME_RESULTADOS.md", "explicacion": "Explicacion.md"}
        if nombre not in archivos or not (DOCUMENTACION / archivos[nombre]).is_file():
            return jsonify(error="Documento no encontrado."), 404
        return send_file(DOCUMENTACION / archivos[nombre], as_attachment=True)

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        # analiza el mensaje priorizando seguridad antes de clasificar emociones
        cuerpo = request.get_json(silent=True) or {}
        mensaje = (cuerpo.get("mensaje") or "").strip()
        if not mensaje:
            return jsonify({"error": "Escribe un mensaje para analizar."}), 400

        resultado = analizar_mensaje(contenedor["modelo"], mensaje)
        estado = cargar_estado_modelo()
        resultado["version_modelo"] = estado.get("version_activa", "base")
        return jsonify(resultado)

    @app.route("/api/correcciones", methods=["GET"])
    def api_listar_correcciones():
        # devuelve los ejemplos locales guardados y el estado actual del modelo
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
        # guarda una correccion voluntaria bloqueando mensajes de riesgo o validacion
        cuerpo = request.get_json(silent=True) or {}
        texto = cuerpo.get("texto", "")
        etiqueta = cuerpo.get("etiqueta", "")
        try:
            guardado = agregar_correccion(texto, etiqueta)
        except ValueError as err:
            return jsonify({"error": str(err)}), 400

        return jsonify(
            {
                "mensaje": "Ejemplo guardado localmente.",
                "correccion": guardado,
                "correcciones": cargar_correcciones(),
            }
        ), 201

    @app.route("/api/correcciones/<corr_id>", methods=["DELETE"])
    def api_eliminar_correccion(corr_id):
        # borra una correccion puntual de la lista local
        eliminado = eliminar_correccion(corr_id)
        if not eliminado:
            return jsonify({"error": "No se encontro esa correccion."}), 404
        return jsonify(
            {
                "mensaje": "Corrección eliminada.",
                "correcciones": cargar_correcciones(),
            }
        )

    @app.route("/api/correcciones", methods=["DELETE"])
    def api_limpiar_correcciones():
        # vacia por completo el almacenamiento local de correcciones
        limpiar_correcciones()
        return jsonify(
            {
                "mensaje": "Se eliminaron todas las correcciones locales.",
                "correcciones": [],
            }
        )

    @app.route("/api/reentrenar", methods=["POST"])
    def api_reentrenar():
        # incorpora las correcciones seguras en una nueva version y mide validacion
        try:
            nuevo_modelo, nuevo_estado = reentrenar_con_correcciones()
        except ValueError as err:
            return jsonify({"error": str(err)}), 400

        contenedor["modelo"] = nuevo_modelo
        return jsonify(
            {
                "mensaje": f"Modelo actualizado a {nuevo_estado['version_activa']}.",
                "estado_modelo": nuevo_estado,
            }
        )

    @app.route("/api/restaurar", methods=["POST"])
    def api_restaurar():
        # vuelve al modelo base congelado y actualiza el modelo cargado en memoria
        try:
            modelo_base, estado_base = restaurar_modelo_base()
        except FileNotFoundError as err:
            return jsonify({"error": str(err)}), 500

        contenedor["modelo"] = modelo_base
        return jsonify(
            {
                "mensaje": "Se restauró el modelo base correctamente.",
                "estado_modelo": estado_base,
            }
        )

    @app.route("/api/laboratorio", methods=["GET"])
    def api_laboratorio():
        # devuelve las metricas calculadas en experimentos y el estado activo
        if not RUTA_RESULTADOS.exists():
            return jsonify(
                {"error": "No existen resultados calculados en modelos/resultados_experimentos.json."}
            ), 404

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
    # inicia el servidor local e imprime la direccion exacta para abrir en clase
    host = os.environ.get("HOST", "127.0.0.1")
    puerto = int(os.environ.get("PORT", "5000"))
    url = f"http://{host}:{puerto}"

    app = crear_app()
    print("=" * 60)
    print("Chatbot académico de clasificación emocional (Proyecto ML)")
    print(f"Abre esta URL en tu navegador: {url}")
    print("=" * 60, flush=True)

    app.run(host=host, port=puerto, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
