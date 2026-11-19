# calculo de metricas de clasificacion, matriz de confusion y abstencion
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from entrenamiento.p01_config import (
    CLASES,
    UMBRAL_MARGEN_MINIMO,
    UMBRAL_PROB_MINIMA,
)


def redondear(valor):
    # devuelve el valor numerico redondeado a cuatro decimales para reportes
    return round(float(valor), 4)


def evaluar_predicciones(y_real, y_pred, etiquetas=CLASES):
    # calcula exactitud, macro-f1, precision, recall, f1 por clase y matriz
    exactitud = accuracy_score(y_real, y_pred)
    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_real,
        y_pred,
        labels=etiquetas,
        zero_division=0,
    )

    por_clase = {}
    for indice, clase in enumerate(etiquetas):
        por_clase[clase] = {
            "precision": redondear(precision[indice]),
            "recall": redondear(recall[indice]),
            "f1": redondear(f1[indice]),
            "soporte": int(soporte[indice]),
        }

    matriz = confusion_matrix(y_real, y_pred, labels=etiquetas).tolist()

    return {
        "total": len(y_real),
        "exactitud": redondear(exactitud),
        "macro_precision": redondear(np.mean(precision)),
        "macro_recall": redondear(np.mean(recall)),
        "macro_f1": redondear(f1_score(y_real, y_pred, labels=etiquetas, average="macro", zero_division=0)),
        "por_clase": por_clase,
        "clases": list(etiquetas),
        "matriz_confusion": matriz,
    }


def evaluar_abstencion(
    modelo,
    textos_limpios,
    y_real,
    umbral_prob=UMBRAL_PROB_MINIMA,
    umbral_margen=UMBRAL_MARGEN_MINIMO,
):
    # mide cuantos mensajes etiqueta el modelo y cuantos de esos acierta
    probs = modelo.predict_proba(textos_limpios)
    clases_modelo = np.array(modelo.classes_)
    orden = np.argsort(probs, axis=1)[:, ::-1]

    top1_indices = orden[:, 0]
    top2_indices = orden[:, 1]
    p1 = probs[np.arange(len(probs)), top1_indices]
    p2 = probs[np.arange(len(probs)), top2_indices]
    margenes = p1 - p2
    predicciones = clases_modelo[top1_indices]

    mascara = (p1 >= umbral_prob) & (margenes >= umbral_margen)
    total = len(y_real)
    etiquetados = int(np.sum(mascara))
    abstenidos = total - etiquetados

    if etiquetados == 0:
        return {
            "umbral_prob": umbral_prob,
            "umbral_margen": umbral_margen,
            "total": total,
            "etiquetados": 0,
            "abstenidos": abstenidos,
            "cobertura": 0.0,
            "aciertos_en_etiquetados": 0,
            "exactitud_selectiva": 0.0,
            "macro_f1_selectivo": 0.0,
        }

    y_real_arr = np.array(y_real)
    y_sel_real = y_real_arr[mascara]
    y_sel_pred = predicciones[mascara]
    aciertos = int(np.sum(y_sel_real == y_sel_pred))

    clases_presentes = [c for c in CLASES if c in set(y_sel_real)]
    f1_sel = f1_score(
        y_sel_real,
        y_sel_pred,
        labels=clases_presentes,
        average="macro",
        zero_division=0,
    )

    return {
        "umbral_prob": umbral_prob,
        "umbral_margen": umbral_margen,
        "total": total,
        "etiquetados": etiquetados,
        "abstenidos": abstenidos,
        "cobertura": redondear(etiquetados / total),
        "aciertos_en_etiquetados": aciertos,
        "exactitud_selectiva": redondear(aciertos / etiquetados),
        "macro_f1_selectivo": redondear(f1_sel),
    }


def evaluar_modelo_en_particiones(modelo, x_go, y_go, x_sin, y_sin):
    # evalua un modelo entrenado en validacion go, sintetica y combinada
    pred_go = modelo.predict(x_go)
    pred_sin = modelo.predict(x_sin)

    metricas_go = evaluar_predicciones(y_go, pred_go)
    metricas_sin = evaluar_predicciones(y_sin, pred_sin)
    metricas_comb = evaluar_predicciones(
        list(y_go) + list(y_sin),
        list(pred_go) + list(pred_sin),
    )

    promedio_macro_f1 = redondear(
        (metricas_go["macro_f1"] + metricas_sin["macro_f1"]) / 2.0
    )

    resultado = {
        "validacion_go": metricas_go,
        "validacion_sintetica": metricas_sin,
        "validacion_combinada": metricas_comb,
        "promedio_macro_f1": promedio_macro_f1,
    }

    if hasattr(modelo, "predict_proba"):
        resultado["abstencion"] = {
            "validacion_go": evaluar_abstencion(modelo, x_go, y_go),
            "validacion_sintetica": evaluar_abstencion(modelo, x_sin, y_sin),
            "validacion_combinada": evaluar_abstencion(
                modelo,
                list(x_go) + list(x_sin),
                list(y_go) + list(y_sin),
            ),
        }

    return resultado
