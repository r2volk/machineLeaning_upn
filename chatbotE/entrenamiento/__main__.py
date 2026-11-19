# importa la lectura de opciones desde la terminal
import argparse


# ejecuta la etapa solicitada desde la terminal
def main():
    # prepara la lectura de opciones desde la terminal
    parser = argparse.ArgumentParser(description="Ejecuta una etapa del experimento")
    # define las etapas permitidas del experimento
    parser.add_argument(
        "etapa", choices=["preparar", "comparar", "evaluar", "informe", "todo"]
    )
    # obtiene la etapa solicitada
    etapa = parser.parse_args().etapa
    # comprueba si corresponde preparar datos
    if etapa in {"preparar", "todo"}:
        # importa la lectura limpieza y verificacion de datos
        from .p01_datos import preparar

        # crea o verifica las particiones
        preparar()
    # comprueba si corresponde comparar modelos
    if etapa in {"comparar", "todo"}:
        # importa la etapa de seleccion con validacion
        from .p03_comparacion import comparar

        # entrena y selecciona o reutiliza la seleccion guardada
        comparar()
    # comprueba si corresponde evaluar el ganador
    if etapa in {"evaluar", "todo"}:
        # importa la etapa de evaluacion unica del test
        from .p04_evaluacion import evaluar_una_vez

        # evalua una vez o devuelve el resultado existente
        evaluar_una_vez()
    # comprueba si corresponde generar graficos y errores
    if etapa in {"informe", "todo"}:
        # importa la generacion de graficos y errores
        from .p06_informe import generar_informe

        # genera las figuras y el archivo de errores
        generar_informe()
    # informa que termino la etapa solicitada
    print(f"Etapa {etapa} completada. Resultados: entrenamiento/resultados/")


# ejecuta las opciones solo al iniciar el paquete como programa
if __name__ == "__main__":
    # procesa la etapa solicitada en la terminal
    main()
