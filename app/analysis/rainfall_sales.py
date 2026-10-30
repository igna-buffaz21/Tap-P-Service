
import pandas as pd

from app.database import get_connection


BRANCH_ID = 44

MIN_TRAIN_DAYS = 300
MIN_TEST_DAYS = 180
MIN_COVERAGE = 0.75
MIN_CORRELATION = 0.10


def load_correlations() -> pd.DataFrame:
    """Calcula la correlación entre lluvia y ventas por producto y período."""

    query = """
        WITH daily_data AS (
            SELECT
                hs.product_id,
                hs.sale_date,
                hs.units_sold,
                hw.precipitation_sum AS rainfall_mm,
                CASE
                    WHEN hs.sale_date < DATE '2016-01-01'
                        THEN 'train'
                    ELSE 'test'
                END AS period
            FROM historical_sale hs
            JOIN branch b
                ON b.id = hs.branch_id
            JOIN historical_weather hw
                ON hw.location_id = b.weather_location_id
               AND hw.weather_date = hs.sale_date
            WHERE hs.branch_id = %s
              AND hs.sale_date BETWEEN
                  DATE '2013-01-01' AND DATE '2017-08-15'
              AND hw.precipitation_sum IS NOT NULL
        )
        SELECT
            product_id,
            period,
            COUNT(*) AS days,
            MIN(sale_date) AS first_date,
            MAX(sale_date) AS last_date,
            CORR(
                units_sold::double precision,
                rainfall_mm::double precision
            ) AS correlation
        FROM daily_data
        GROUP BY product_id, period
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (BRANCH_ID,))
            rows = cur.fetchall()
            columns = [column.name for column in cur.description]

    return pd.DataFrame(rows, columns=columns)


def analyze_rainfall_sales():
    """Busca productos con patrones estables en ambos períodos."""

    df = load_correlations()

    if df.empty:
        raise ValueError("No se encontraron datos.")

    df["first_date"] = pd.to_datetime(df["first_date"])
    df["last_date"] = pd.to_datetime(df["last_date"])

    # Calculamos qué proporción de días tiene registros.
    df["period_days"] = (
        df["last_date"] - df["first_date"]
    ).dt.days + 1

    df["coverage"] = df["days"] / df["period_days"]

    # Filtramos productos con suficiente historial.
    df = df[
        (df["coverage"] >= MIN_COVERAGE)
        & (
            ((df["period"] == "train")
             & (df["days"] >= MIN_TRAIN_DAYS))
            |
            ((df["period"] == "test")
             & (df["days"] >= MIN_TEST_DAYS))
        )
    ]

    # Colocamos entrenamiento y prueba en la misma fila.
    result = df.pivot(
        index="product_id",
        columns="period",
        values=["correlation", "days"],
    )

    result.columns = [
        f"{metric}_{period}"
        for metric, period in result.columns
    ]

    result = result.dropna().reset_index()

    print(f"Productos con historial suficiente: {len(result)}")

    # Exigimos correlación relevante en ambos períodos.
    stable = result[
        (result["correlation_train"].abs() >= MIN_CORRELATION)
        & (result["correlation_test"].abs() >= MIN_CORRELATION)
        & (
            result["correlation_train"]
            * result["correlation_test"] > 0
        )
    ].copy()

    # Ordenamos por la correlación más débil de ambos períodos.
    stable["strength"] = stable[
        ["correlation_train", "correlation_test"]
    ].abs().min(axis=1)

    stable = stable.sort_values(
        "strength",
        ascending=False,
    )

    print(f"Productos con asociación estable: {len(stable)}")

    print("\nTOP 20 PRODUCTOS:")
    print(
        stable[
            [
                "product_id",
                "days_train",
                "days_test",
                "correlation_train",
                "correlation_test",
            ]
        ].head(20).round(4).to_string(index=False)
    )


if __name__ == "__main__":
    analyze_rainfall_sales()
