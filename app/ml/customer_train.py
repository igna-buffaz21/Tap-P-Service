
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from app.config import BUSINESS_ID
from app.data.customer_repository import get_customer_purchases
from app.ml.customer_features import build_customer_features


FEATURE_COLUMNS = [
    "days_since_last_purchase",
    "purchases_last_30d",
    "purchases_last_90d",
    "avg_days_between_purchases",
    "total_spent_last_90d",
]


def train_classifier():
    """Entrena un clasificador para predecir el regreso de clientes."""

    # 1. Obtener las compras y construir las variables.
    purchases = get_customer_purchases(BUSINESS_ID)
    data = build_customer_features(purchases)

    # 2. Separar cronológicamente entrenamiento y prueba.
    dates = sorted(data["evaluation_date"].unique())
    split_index = int(len(dates) * 0.8)
    test_start = dates[split_index]

    # Las etiquetas de entrenamiento deben conocerse
    # antes del comienzo del período de prueba.
    train = data[
        data["evaluation_date"] + 
        __import__("pandas").Timedelta(days=30) <= test_start
    ]

    test = data[data["evaluation_date"] >= test_start]

    if train.empty or test.empty:
        raise ValueError("No hay suficientes datos para evaluar.")

    # 3. Separar entradas (X) y resultados reales (y).
    X_train = train[FEATURE_COLUMNS]
    y_train = train["will_return_30d"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["will_return_30d"]

    if y_train.nunique() < 2:
        raise ValueError("Se necesitan ejemplos de ambas clases.")

    # 4. Crear y entrenar el clasificador.
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Probabilidad estimada de que el cliente NO regrese.
    no_return_index = list(model.classes_).index(0)

    prob_no_return = model.predict_proba(X_test)[:, no_return_index]

    # Umbral de clasificación: 40 %
    threshold = 0.40

    # 0 = No regresa | 1 = Regresa
    predictions = (prob_no_return < threshold).astype(int)

    # 6. Evaluar, tomando 0 = no regresa como clase de interés.
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=0, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label=0, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label=0, zero_division=0
    )

    print(f"Ejemplos de entrenamiento: {len(train)}")
    print(f"Ejemplos de prueba: {len(test)}")
    print(f"Inicio del período de prueba: {test_start}")
    print(f"Profundidad del árbol: {model.get_depth()}")
    print(f"Cantidad de hojas: {model.get_n_leaves()}")

    print("\nMÉTRICAS DEL CLASIFICADOR")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (no regresa): {precision:.4f}")
    print(f"Recall (no regresa):    {recall:.4f}")
    print(f"F1-score (no regresa):  {f1:.4f}")

    print("\nMATRIZ DE CONFUSIÓN")
    print("Filas: valores reales | Columnas: predicciones")
    print("Orden de clases: [0 = no regresa, 1 = regresa]")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # 7. Comparar con el modelo base: siempre predice "regresa".
    baseline = [1] * len(y_test)

    print("\nMODELO BASE (siempre predice que regresa)")
    print(f"Accuracy: {accuracy_score(y_test, baseline):.4f}")
    print(
        "F1-score (no regresa): "
        f"{f1_score(y_test, baseline, pos_label=0, zero_division=0):.4f}"
    )

    # 8. Mostrar ejemplos y probabilidades estimadas.
    results = test[
        ["customer_id", "evaluation_date", "will_return_30d"]
    ].copy()

    results["predicted_return"] = predictions

    # Probabilidad estimada de que NO regrese.
    no_return_index = list(model.classes_).index(0)
    results["prob_no_return"] = model.predict_proba(X_test)[
        :, no_return_index
    ]

    print("\nPRIMERAS 10 PREDICCIONES")
    print(results.head(10).to_string(index=False))

    return model


if __name__ == "__main__":
    train_classifier()
