
import numpy as np
import pandas as pd


def build_customer_features(
    df: pd.DataFrame,
    horizon_days: int = 30,
    history_days: int = 90,
) -> pd.DataFrame:
    """Genera ejemplos históricos para predecir el regreso de clientes."""

    required = {
        "customer_id",
        "purchase_date",
        "total_amount",
    }

    if not required.issubset(df.columns):
        raise ValueError("Faltan columnas requeridas.")

    if df.empty:
        raise ValueError("No hay compras para procesar.")

    if horizon_days <= 0 or history_days <= 0:
        raise ValueError("Los períodos deben ser positivos.")

    data = df.copy()
    data["purchase_date"] = pd.to_datetime(data["purchase_date"])
    data["total_amount"] = pd.to_numeric(data["total_amount"])

    data = data.sort_values(["customer_id", "purchase_date"])

    # Fechas de evaluación, cada 30 días.
    first_date = data["purchase_date"].min().normalize()
    last_date = data["purchase_date"].max().normalize()

    evaluation_dates = pd.date_range(
        start=first_date + pd.Timedelta(days=history_days),
        end=last_date - pd.Timedelta(days=horizon_days),
        freq=f"{horizon_days}D",
    )

    records = []

    for customer_id, purchases in data.groupby("customer_id"):
        dates = purchases["purchase_date"].to_numpy(
            dtype="datetime64[ns]"
        )
        amounts = purchases["total_amount"].to_numpy(dtype=float)

        # Permite calcular gastos históricos sin sumar repetidamente.
        cumulative_spent = np.concatenate(
            ([0.0], np.cumsum(amounts))
        )

        for evaluation_date in evaluation_dates:
            current = np.datetime64(evaluation_date)
            start_30 = current - np.timedelta64(30, "D")
            start_90 = current - np.timedelta64(90, "D")
            future_end = current + np.timedelta64(
                horizon_days, "D"
            )

            # Las compras anteriores a la evaluación.
            idx_current = np.searchsorted(dates, current)
            if idx_current == 0:
                continue

            idx_30 = np.searchsorted(dates, start_30)
            idx_90 = np.searchsorted(dates, start_90)

            # Compras históricas.
            purchases_30 = idx_current - idx_30
            purchases_90 = idx_current - idx_90

            last_purchase = dates[idx_current - 1]
            days_since_last = (
                (current - last_purchase) / np.timedelta64(1, "D")
            )

            # Intervalo promedio entre compras anteriores.
            historical_dates = dates[:idx_current]

            if len(historical_dates) >= 2:
                intervals = np.diff(historical_dates)
                avg_interval = np.mean(
                    intervals / np.timedelta64(1, "D")
                )
            else:
                avg_interval = 0.0

            spent_90 = (
                cumulative_spent[idx_current]
                - cumulative_spent[idx_90]
            )

            # Variable objetivo: ¿compró en los próximos 30 días?
            idx_future = np.searchsorted(dates, future_end)
            will_return = int(idx_future > idx_current)

            records.append({
                "customer_id": customer_id,
                "evaluation_date": evaluation_date,
                "days_since_last_purchase": float(days_since_last),
                "purchases_last_30d": purchases_30,
                "purchases_last_90d": purchases_90,
                "avg_days_between_purchases": float(avg_interval),
                "total_spent_last_90d": float(spent_90),
                "will_return_30d": will_return,
            })

    if not records:
        raise ValueError("No se pudieron generar ejemplos.")

    return pd.DataFrame(records).sort_values(
        ["evaluation_date", "customer_id"]
    ).reset_index(drop=True)
