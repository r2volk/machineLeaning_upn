"""Orquestador explícito: preparar → comparar → evaluar una vez → publicar informe."""
import argparse


def main():
    """Selecciona una etapa; repetir informe no vuelve a ejecutar el test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('etapa', choices=['preparar', 'comparar', 'evaluar', 'informe', 'todo'])
    etapa = parser.parse_args().etapa
    if etapa in {'preparar', 'todo'}:
        from .p01_datos import preparar
        preparar()
    if etapa in {'comparar', 'todo'}:
        from .p03_comparacion import comparar
        comparar()
    if etapa in {'evaluar', 'todo'}:
        from .p04_evaluacion import evaluar_una_vez
        evaluar_una_vez()
    if etapa in {'informe', 'todo'}:
        from .p06_informe import generar_informe
        generar_informe()
    print(f'Etapa {etapa} completada. Resultados: entrenamiento/resultados/')


if __name__ == '__main__':
    main()
