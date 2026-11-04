
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from app.config import BUSINESS_ID
from app.data.sales_repository import get_weekly_sales
from app.ml.sales_preprocessing import prepare_weekly_sales
from app.ml.sales_features import build_features
from sklearn.tree import export_text

from app.ml.sales_preprocessing import prepare_weekly_sales, TEST_START


#variables de entrada para el entrenamiento
FEATURE_COLUMNS = [
    "product_id",
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_lag_4",
    "sales_avg_4",
    "sales_change",
]


def train_regressor(branch_id: int):
    """Entrena y evalúa un árbol de regresión de ventas semanales."""

    # obtiene la data, y los procesa 
    df = get_weekly_sales(BUSINESS_ID, branch_id=branch_id)
    prepared = prepare_weekly_sales(df)


    first_sales = (
    df.groupby(["branch_id", "product_id"])["week_start"]
    .min()
    .rename("first_sale_week")
    )

    check = prepared.join(
        first_sales,
        on=["branch_id", "product_id"]
    )

    artificial_zeros = (
        (check["units_sold"] == 0)
        & (check["week_start"] < check["first_sale_week"])
    )

    print(
        "Ceros anteriores a la primera venta:",
        artificial_zeros.sum()
    )

    data = build_features(prepared)

    test_start = TEST_START

    train = data[data["week_start"] < test_start] #con esta data se entrena el modelo
    test = data[data["week_start"] >= test_start] #con esta data se evalua el modelo

    if train.empty or test.empty:
        raise ValueError("No hay suficientes datos para entrenar y evaluar.")

    
    X_train = train[FEATURE_COLUMNS] #variables de entrada
    y_train = train["units_sold"] #variable reales

    X_test = test[FEATURE_COLUMNS] #variables de entrada
    y_test = test["units_sold"] #ventas reales

    print(
    "Porcentaje de semanas sin ventas (train):",
    round((train["units_sold"] == 0).mean() * 100, 2),
    "%"
    )

    print(
        "Porcentaje de semanas sin ventas (test):",
        round((test["units_sold"] == 0).mean() * 100, 2),
        "%"
    )

    
    model = DecisionTreeRegressor(
        max_depth=8, #Profundida maxima
        min_samples_leaf=20, #minimo de muestras por hoja
        random_state=42,
    )

    model.fit(X_train, y_train) #ejecucion del algoritmo de entrenamiento

    # Mostrar las reglas aprendidas por el árbol.
    tree_rules = export_text(
        model,
        feature_names=FEATURE_COLUMNS,
        decimals=2,
        max_depth=3,
    )

    # 5. Predecir las ventas de las semanas de prueba.
    predictions = model.predict(X_test)

    # 6. Evaluar el error del árbol.
    mae = mean_absolute_error(y_test, predictions)

    # Modelo base: predecir las mismas ventas de la semana anterior.
    baseline_mae = mean_absolute_error(
        y_test,
        test["sales_lag_1"],
    )

    print(f"Ejemplos de entrenamiento: {len(train)}")
    print(f"Ejemplos de prueba: {len(test)}")
    print(f"Inicio del período de prueba: {test_start}")
    print(f"Profundidad del árbol: {model.get_depth()}")
    print(f"Cantidad de hojas: {model.get_n_leaves()}")
    print(f"MAE del árbol: {mae:.2f}")
    print(f"MAE del modelo base: {baseline_mae:.2f}")

    # Mostrar algunas predicciones frente a las ventas reales.
    results = test[
        ["branch_id", "product_id", "week_start", "units_sold"]
    ].copy()

    results["predicted_units"] = predictions

    print("\nPrimeras 10 predicciones:")
    print(results.head(10).to_string(index=False))

    return model


if __name__ == "__main__":
    while True:
        try:
            branch_id = int(input("Ingresa la sucursal a analizar (44, 45, 47, 3, 46): "))
            break
        except ValueError:
            print("Por favor, ingresa un número entero.")

    train_regressor(branch_id)
