# compara bigramas y trigramas sin volver a evaluar el test
import csv
from time import perf_counter

import joblib
import numpy as np
from sklearn.base import clone

from .p00_configuracion import CLASES, DATOS_COMPARTIDOS, MARGEN, SALIDAS, guardar_json, huella, leer_json
from .p01_datos import normalizar, xy
from .p02_modelos import mascara_aceptacion, metricas, selectivas
from .p07_datos_ampliados import DESTINO, leer_csv
from .p08_comparacion_ampliada import MODELO_AMPLIADO, RESULTADOS

DIAGNOSTICO = DATOS_COMPARTIDOS / "diagnostico_E" / "diagnostico.csv"
SALIDA = SALIDAS / "contexto" / "comparacion.json"
PARTICIONES = ("train.csv", "validacion_go.csv", "validacion_sintetica.csv")


# lee los casos escritos para explorar negaciones y expresiones indirectas
def leer_diagnostico(particiones):
    with DIAGNOSTICO.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))

    textos_corpus = {
        normalizar(fila["texto"])
        for particion in particiones.values()
        for fila in particion
    }
    textos = [normalizar(fila["texto"]) for fila in filas]
    if len(set(textos)) != len(textos) or textos_corpus.intersection(textos):
        raise ValueError("Hay textos repetidos entre el diagnóstico y los corpus usados.")
    return filas


# usa las mismas condiciones de abstencion que la aplicacion
def puntuar(modelo, textos, umbral):
    vectores = modelo["vectorizador"].transform(textos)
    puntuaciones = modelo["clasificador"].predict_proba(vectores)
    predichas = modelo.classes_[puntuaciones.argmax(axis=1)]
    aceptadas = mascara_aceptacion(
        puntuaciones, umbral, vectores.getnnz(axis=1) > 0
    )
    return predichas, aceptadas, puntuaciones


# mide por separado el clasificador y las respuestas que muestra el chat
def medir_validacion(modelo, filas, umbral):
    textos, reales = xy(filas)
    predichas, aceptadas, _ = puntuar(modelo, textos, umbral)
    return {
        "n": len(filas),
        "metricas": metricas(reales, predichas),
        "chat": selectivas(reales, predichas, aceptadas),
    }


# cuenta aciertos por tipo y pares resueltos con sus dos frases correctas
def medir_diagnostico(modelo, filas, umbral):
    textos = [fila["texto"] for fila in filas]
    predichas, aceptadas, puntuaciones = puntuar(modelo, textos, umbral)
    analizador = modelo["vectorizador"].build_analyzer()
    vocabulario = modelo["vectorizador"].vocabulary_
    casos = []
    for fila, predicha, aceptada, scores in zip(
        filas, predichas, aceptadas, puntuaciones
    ):
        casos.append({
            **fila,
            "candidata": str(predicha),
            "chat": str(predicha) if aceptada else "incierto",
            "scores": dict(zip(modelo.classes_.tolist(), scores.tolist())),
            "trigramas_conocidos": sorted({
                termino
                for termino in analizador(fila["texto"])
                if termino.count(" ") == 2 and termino in vocabulario
            }),
        })

    etiquetados = [caso for caso in casos if caso["etiqueta"]]
    grupos = {}
    for caso in etiquetados:
        grupos.setdefault(caso["grupo"], []).append(caso)

    por_tipo = {}
    for tipo in ("negacion", "indirecta"):
        subset = [caso for caso in etiquetados if caso["tipo"] == tipo]
        por_tipo[tipo] = {
            "total": len(subset),
            "aciertos_chat": sum(caso["chat"] == caso["etiqueta"] for caso in subset),
        }

    neutrales = [caso for caso in etiquetados if caso["etiqueta"] == "neutral"]
    return {
        "n": len(etiquetados),
        "aciertos_chat": sum(caso["chat"] == caso["etiqueta"] for caso in etiquetados),
        "abstenciones": sum(caso["chat"] == "incierto" for caso in etiquetados),
        "pares_totales": len(grupos),
        "pares_correctos": sum(
            all(caso["chat"] == caso["etiqueta"] for caso in grupo)
            for grupo in grupos.values()
        ),
        "controles_neutrales": len(neutrales),
        "controles_neutrales_con_emocion": sum(caso["chat"] in CLASES
                                         and caso["chat"] != "neutral" for caso in neutrales),
        "por_tipo": por_tipo,
        "casos": casos,
    }


# reporta el filtro acordado para recomendar una evaluacion posterior
def revisar_candidato(base, candidato):
    go_base = base["validacion_go"]["metricas"]["macro_f1"]
    go_candidato = candidato["validacion_go"]["metricas"]["macro_f1"]
    a, b = base["diagnostico"], candidato["diagnostico"]
    return {
        "macro_f1_go_igual_o_mejor": go_candidato >= go_base,
        "mas_aciertos_en_diagnostico": b["aciertos_chat"] > a["aciertos_chat"],
        "pares_correctos_igual_o_mejor": b["pares_correctos"] >= a["pares_correctos"],
        "sin_mas_emociones_en_controles_neutrales": (
            b["controles_neutrales_con_emocion"] <= a["controles_neutrales_con_emocion"]
        ),
    }


# verifica el modelo de referencia y ajusta solo el candidato nuevo
def comparar():
    seleccion = leer_json(RESULTADOS / "seleccion.json")
    if huella(MODELO_AMPLIADO) != seleccion["modelo_sha256"]:
        raise RuntimeError("El modelo de referencia no coincide con la selección.")
    huellas = {nombre: huella(DESTINO / nombre) for nombre in PARTICIONES}
    if huellas != seleccion["datos_sha256"]:
        raise RuntimeError("Los datos preparados no coinciden con la selección.")

    particiones = {nombre: leer_csv(nombre) for nombre in PARTICIONES}
    diagnostico = leer_diagnostico(particiones)
    artefacto = joblib.load(MODELO_AMPLIADO)
    base = artefacto["pipeline"]
    umbral = artefacto["umbral"]
    candidato = clone(base).set_params(vectorizador__ngram_range=(1, 3))
    inicio = perf_counter()
    candidato.fit(*xy(particiones["train.csv"]))
    segundos = perf_counter() - inicio

    resultados = {}
    for nombre, modelo in (("bigramas", base), ("trigramas", candidato)):
        resultados[nombre] = {
            "rango": modelo["vectorizador"].ngram_range,
            "vocabulario": len(modelo["vectorizador"].vocabulary_),
            "validacion_go": medir_validacion(modelo, particiones["validacion_go.csv"], umbral),
            "validacion_sintetica": medir_validacion(
                modelo, particiones["validacion_sintetica.csv"], umbral
            ),
            "diagnostico": medir_diagnostico(modelo, diagnostico, umbral),
        }

    # comprueba que la referencia reproduce la metrica guardada
    anterior = next(fila for fila in seleccion["candidatos"]
                    if fila["nombre"] == seleccion["ganador"])
    for particion in ("validacion_go", "validacion_sintetica"):
        original = anterior[particion]["macro_f1"]
        comparado = resultados["bigramas"][particion]["metricas"]["macro_f1"]
        if not np.isclose(original, comparado):
            raise RuntimeError("La referencia no reproduce el macro F1 guardado.")

    guardar_json(SALIDA, {
        "modelo_base_sha256": seleccion["modelo_sha256"],
        "datos_sha256": huellas,
        "diagnostico_sha256": huella(DIAGNOSTICO),
        "train_n": len(particiones["train.csv"]),
        "umbral": umbral,
        "margen": MARGEN,
        "segundos_ajuste_trigramas": segundos,
        "resultados": resultados,
        "criterios": revisar_candidato(resultados["bigramas"], resultados["trigramas"]),
    })
    print(f"Comparación de validación guardada en {SALIDA}")


if __name__ == "__main__":
    comparar()
