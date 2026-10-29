from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


COLUMNAS_CATEGORICAS = ["store", "item"]

COLUMNAS_NUMERICAS = [
    "day_index",
    "day_of_week",
    "month",
    "lag_7",
    "lag_14",
    "media_7_dias",
    "media_28_dias",
]

COLUMNAS_ENTRADA = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS


def preparar_variables(datos):
    """Prepara calendario e historial de ventas como entradas del modelo."""
    variables = datos.copy()

    variables["day_index"] = (
        variables["date"] - variables["date"].min()
    ).dt.days
    variables["day_of_week"] = variables["date"].dt.dayofweek
    variables["month"] = variables["date"].dt.month

    return variables[COLUMNAS_ENTRADA]


def crear_modelos():
    """Construye regresión lineal y Random Forest con el mismo preprocesamiento."""
    def crear_preparacion():
        return ColumnTransformer(
            transformers=[
                (
                    "categorias",
                    OneHotEncoder(handle_unknown="ignore"),
                    COLUMNAS_CATEGORICAS,
                ),
                ("numericas", "passthrough", COLUMNAS_NUMERICAS),
            ]
        )

    return {
        "Regresión lineal": make_pipeline(
            crear_preparacion(),
            LinearRegression(),
        ),
        "Random Forest": make_pipeline(
            crear_preparacion(),
            RandomForestRegressor(
                n_estimators=20,
                max_depth=12,
                min_samples_leaf=10,
                max_samples=0.10,
                n_jobs=2,
                random_state=42,
            ),
        ),
    }