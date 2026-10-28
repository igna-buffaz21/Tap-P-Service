import pandas as pd


def prepare_weekly_sales(
    df: pd.DataFrame, #df = respuesta de la base de datos
    min_active_weeks: int = 40, #número mínimo de semanas activas que debe tener un producto para ser incluido
    train_ratio: float = 0.8, #porcentaje de datos que vamos a usar para entrenar el modelo, el resto se usa para testearlo
) -> pd.DataFrame:
    """Prepara series semanales continuas por producto y sucursal."""

    if df.empty:
        raise ValueError("No hay datos de ventas para procesar.")

    required = {"branch_id", "product_id", "week_start", "units_sold"}

    if not required.issubset(df.columns):
        raise ValueError("Faltan columnas requeridas.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio debe estar entre 0 y 1.")

    data = df.copy()
    data["week_start"] = pd.to_datetime(data["week_start"])
    data["units_sold"] = pd.to_numeric(data["units_sold"])

    # El dataset reconstruido abarca del 01/01/2023 al 11/12/2024.
    # Conservamos únicamente semanas completas (lunes a domingo).
    first_week = pd.Timestamp("2023-01-02")
    last_week = pd.Timestamp("2024-12-02")

    data = data[
        data["week_start"].between(first_week, last_week) #aca filtramos las semanas que queremos conservar, entre el 01/01/2023 y el 11/12/2024
    ]

    weeks = pd.date_range(first_week, last_week, freq="W-MON") #aca generamos todas las semanas entre el 01/01/2023 y el 11/12/2024, con frecuencia semanal, empezando los lunes

    # La selección se hace usando solamente el período de entrenamiento.
    train_weeks = int(len(weeks) * train_ratio)
    train_end = weeks[train_weeks - 1]

    activity = (
        data[data["week_start"] <= train_end]
        .groupby(["branch_id", "product_id"])["week_start"]
        .nunique()
    )

    selected = activity[activity >= min_active_weeks].index

    if len(selected) == 0:
        raise ValueError("Ningún producto cumple el criterio de actividad.")

    # Filtramos las combinaciones seleccionadas.
    data = data.set_index(["branch_id", "product_id"])
    data = data.loc[data.index.isin(selected)].reset_index()

    # Generamos todas las combinaciones seleccionadas y semanas.
    full_index = pd.MultiIndex.from_product(
        [
            sorted(data["branch_id"].unique()),
            sorted(data["product_id"].unique()),
            weeks,
        ],
        names=["branch_id", "product_id", "week_start"],
    )

    # Evitamos crear combinaciones de producto y sucursal inexistentes.
    full_index = full_index[
        full_index.droplevel("week_start").isin(selected)
    ]

    result = (
        data.set_index(["branch_id", "product_id", "week_start"])
        ["units_sold"]
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    result["units_sold"] = result["units_sold"].astype(float)

    return result