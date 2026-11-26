# interfaz de consola para conversar con el modelo sin servidor web
import joblib

from entrenamiento.p01_config import (
    NOMBRES_CLASES,
    RUTA_MODELO_ACTIVO,
    RUTA_MODELO_BASE,
)
from entrenamiento.p07_inferencia import analizar_mensaje


def cargar_modelo():
    # carga el modelo activo si existe o usa el modelo base congelado
    if RUTA_MODELO_ACTIVO.exists():
        # si reentrenaste en la web, carga primero esa copia activa
        return joblib.load(RUTA_MODELO_ACTIVO)
    if RUTA_MODELO_BASE.exists():
        # si no hay version reentrenada, usa el modelo base original
        return joblib.load(RUTA_MODELO_BASE)
    # si no existe ningun archivo joblib, avisa que script debes correr primero
    raise FileNotFoundError(
        "No existe el modelo entrenado. Ejecuta primero: python -m entrenamiento.p08_experimentos"
    )


def formatear_barra(probabilidad, ancho=20):
    # dibuja una barra de texto proporcional al porcentaje de la emocion
    # calcula cuantos simbolos # corresponden sobre 20 espacios totales
    lleno = round(probabilidad * ancho)
    return "#" * lleno + "-" * (ancho - lleno)


def mostrar_resultado(resultado):
    # imprime primero la respuesta del chatbot, luego la emocion y un separador
    print()
    if resultado["es_riesgo"]:
        # si detecto riesgo, muestra ayuda humana y corta antes de mostrar porcentajes
        print(f"Chatbot: {resultado['respuesta']}")
        print("\n[ATENCION PRIORITARIA - SEGURIDAD]")
        print("-" * 60+ "\n")
        return

    # imprime primero el texto empatico que responde el chatbot
    print(f"Chatbot: {resultado['respuesta']}\n")

    # muestra el encabezado segun si clasifico o pidio mas contexto
    if resultado["estado"] == "clasificado":
        # agrega un aviso si combino este mensaje con el turno anterior
        extra = " (con contexto previo)" if resultado.get("uso_contexto") else ""
        print(f"Emocion detectada: {resultado['nombre_etiqueta']}{extra}")
    elif resultado["estado"] == "pedir_contexto":
        # indica que prefirio no adivinar por baja confianza o empate
        print("Estado: Pide mas contexto (senal baja o cercana entre emociones)")
    else:
        # ocurre si solo escribieron signos, numeros o emojis sin palabras
        print("Estado: Sin texto suficiente")

    # lista las tres emociones que mas resaltan con su porcentaje
    probs = resultado.get("probabilidades") or {}
    if probs:
        print("Emociones que mas resaltan:")
        # toma solo los 3 primeros elementos porque ya vienen ordenados de mayor a menor
        for clase, valor in list(probs.items())[:3]:
            nombre = NOMBRES_CLASES.get(clase, clase)
            barra = formatear_barra(valor)
            print(f"  {nombre:<24} [{barra}] {valor * 100:5.1f}%")

    # muestra las palabras o subpalabras que mas influyeron en la decision
    terminos = resultado.get("terminos_clave") or []
    if terminos:
        # une hasta 4 palabras clave separadas por comas para mostrar evidencia
        lista_terminos = ", ".join(item["termino"] for item in terminos[:4])
        print(f"Terminos clave: {lista_terminos}")

    # linea separadora para distinguir visualmente cada turno de chat
    print("-" * 60 +"\n")

def main():
    # inicia el bucle de conversacion en terminal manteniendo contexto de un turno
    modelo = cargar_modelo()
    # empieza sin memoria previa hasta que el bot pida mas contexto
    contexto_previo = None

    print("\n------------ Escribe 'salir' para terminar -------------\n")

    # bucle infinito que escucha un mensaje tras otro hasta que decidas salir
    while True:
        try:
            # lee lo que escribe el usuario y quita espacios al inicio o final
            mensaje = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            # captura ctrl c o cierre de consola para terminar sin error
            print("\nSesion finalizada.")
            break

        if not mensaje:
            # si solo presionaron enter sin escribir, vuelve a pedir entrada
            continue
        if mensaje.lower() in {"salir", "exit", "quit"}:
            # permite cerrar el programa escribiendo salir, exit o quit
            print("Sesion finalizada.")
            break

        # analiza el mensaje usando contexto previo solo si el turno anterior lo pidio
        resultado = analizar_mensaje(
            modelo,
            mensaje,
            contexto_previo=contexto_previo,
        )
        # muestra en pantalla la respuesta, la emocion y las barras
        mostrar_resultado(resultado)

        # guarda el mensaje un turno si pide contexto y lo limpia al clasificar
        if not resultado["es_riesgo"] and resultado["estado"] == "pedir_contexto":
            contexto_previo = mensaje
        else:
            # borra el recuerdo apenas clasifica para no mezclar temas nuevos
            contexto_previo = None


if __name__ == "__main__":
    main()
