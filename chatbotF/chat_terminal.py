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
        return joblib.load(RUTA_MODELO_ACTIVO)
    if RUTA_MODELO_BASE.exists():
        return joblib.load(RUTA_MODELO_BASE)
    raise FileNotFoundError(
        "No existe el modelo entrenado. Ejecuta primero: python -m entrenamiento.p08_experimentos"
    )


def formatear_barra(probabilidad, ancho=20):
    # dibuja una barra de texto proporcional al porcentaje de la emocion
    lleno = round(probabilidad * ancho)
    return "#" * lleno + "-" * (ancho - lleno)


def mostrar_resultado(resultado):
    # imprime primero la respuesta del chatbot, luego la emocion y un separador
    print()
    if resultado["es_riesgo"]:
        print(f"Chatbot: {resultado['respuesta']}")
        print("\n[ATENCION PRIORITARIA - SEGURIDAD]")
        print("-" * 60+ "\n")
        return

    print(f"Chatbot: {resultado['respuesta']}\n")

    # muestra el encabezado segun si clasifico o pidio mas contexto
    if resultado["estado"] == "clasificado":
        extra = " (con contexto previo)" if resultado.get("uso_contexto") else ""
        print(f"Emocion detectada: {resultado['nombre_etiqueta']}{extra}")
    elif resultado["estado"] == "pedir_contexto":
        print("Estado: Pide mas contexto (senal baja o cercana entre emociones)")
    else:
        print("Estado: Sin texto suficiente")

    # lista las tres emociones que mas resaltan con su porcentaje
    probs = resultado.get("probabilidades") or {}
    if probs:
        print("Emociones que mas resaltan:")
        for clase, valor in list(probs.items())[:3]:
            nombre = NOMBRES_CLASES.get(clase, clase)
            barra = formatear_barra(valor)
            print(f"  {nombre:<24} [{barra}] {valor * 100:5.1f}%")

    # muestra las palabras o subpalabras que mas influyeron en la decision
    terminos = resultado.get("terminos_clave") or []
    if terminos:
        lista_terminos = ", ".join(item["termino"] for item in terminos[:4])
        print(f"Terminos clave: {lista_terminos}")

    print("-" * 60 +"\n")

def main():
    # inicia el bucle de conversacion en terminal manteniendo contexto de un turno
    modelo = cargar_modelo()
    contexto_previo = None

    print("\n------------ Escribe 'salir' para terminar -------------\n")

    while True:
        try:
            mensaje = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSesion finalizada.")
            break

        if not mensaje:
            continue
        if mensaje.lower() in {"salir", "exit", "quit"}:
            print("Sesion finalizada.")
            break

        # analiza el mensaje usando contexto previo solo si el turno anterior lo pidio
        resultado = analizar_mensaje(
            modelo,
            mensaje,
            contexto_previo=contexto_previo,
        )
        mostrar_resultado(resultado)

        # guarda el mensaje un turno si pide contexto y lo limpia al clasificar
        if not resultado["es_riesgo"] and resultado["estado"] == "pedir_contexto":
            contexto_previo = mensaje
        else:
            contexto_previo = None


if __name__ == "__main__":
    main()
