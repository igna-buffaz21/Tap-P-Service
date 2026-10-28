import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye variables históricas para predecir la demanda semanal."""

    required = {"branch_id", "product_id", "week_start", "units_sold"}

    if not required.issubset(df.columns):
        raise ValueError("Faltan columnas requeridas.")

    data = df.sort_values(
        ["branch_id", "product_id", "week_start"]
    ).copy()

    # Agrupamos por producto y sucursal para no mezclar historiales.
    groups = data.groupby(["branch_id", "product_id"])["units_sold"]

    #ejemplo de data:
    # branch_id | product_id | week_start | units_sold
    # Grupo 1-------------------------------------------
    # 1         | 1          | 2023-01-02  | 10
    # 1         | 1          | 2023-01-09  | 15
    # 1         | 1          | 2023-01-16  | 20
    # Grupo 2-------------------------------------------
    # 1         | 2          | 2023-01-02  | 5
    # 1         | 2          | 2023-01-09  | 10
    # 1         | 2          | 2023-01-16  | 15
    # Grupo 3-------------------------------------------
    # 1         | 3          | 2023-01-02  | 0


    for i in range(1, 5):
        data[f"sales_lag_{i}"] = groups.shift(i)

    #ejemplo de data:
    # sales_lag_1 = ventas de la semana anterior a la semana actual
    # branch_id | product_id | week_start | units_sold | sales_lag_1 | sales_lag_2 | sales_lag_3 | sales_lag_4
    # Grupo 1-------------------------------------------
    # 1         | 1          | 2023-01-02  | 10        | NaN         | NaN         | NaN         | NaN
    # 1         | 1          | 2023-01-09  | 15        | 10          | NaN         | NaN         | NaN
    # 1         | 1          | 2023-01-16  | 20        | 15          | 10          | NaN         | NaN
    # Grupo 2-------------------------------------------
    # 1         | 2          | 2023-01-02  | 5          | NaN         | NaN         | NaN         | NaN
    # 1         | 2          | 2023-01-09  | 10         | 5           | NaN         | NaN         | NaN
    # 1         | 2          | 2023-01-16  | 15         | 10          | 5           | NaN         | NaN


    #aca se agrega una nueva columna al dataframe llamada "sales_avg_4" que contiene el promedio de ventas de las últimas 4 semanas
    data["sales_avg_4"] = (
        data[
            ["sales_lag_1", "sales_lag_2",
             "sales_lag_3", "sales_lag_4"]
        ].mean(axis=1)
    )

    # Variación entre las dos últimas semanas.
    data["sales_change"] = (
        data["sales_lag_1"] - data["sales_lag_2"]
    )

    # Eliminamos las primeras cuatro semanas de cada serie por producto y sucursal, ya que no tienen suficiente historial.
    data = data.dropna(
        subset=[f"sales_lag_{i}" for i in range(1, 5)]
    )

    # branch_id | product_id | week_start | units_sold | lag_1 | lag_2 | lag_3 | lag_4 | sales_avg_4 | sales_change
    #
    # Grupo 1 -----------------------------------------------------------------------------------------------------
    # 1         | 1          | 2023-01-02 | 10         | NaN   | NaN   | NaN   | NaN   | NaN         | NaN
    # 1         | 1          | 2023-01-09 | 15         | 10    | NaN   | NaN   | NaN   | NaN         | NaN
    # 1         | 1          | 2023-01-16 | 20         | 15    | 10    | NaN   | NaN   | 12.50       | 5
    # 1         | 1          | 2023-01-23 | 12         | 20    | 15    | 10    | NaN   | 15.00       | 5
    # 1         | 1          | 2023-01-30 | 18         | 12    | 20    | 15    | 10    | 14.25       | -8
    # 1         | 1          | 2023-02-06 | 25         | 18    | 12    | 20    | 15    | 16.25       | 6
    # 1         | 1          | 2023-02-13 | 22         | 25    | 18    | 12    | 20    | 18.75       | 7
    #
    # Grupo 2 -----------------------------------------------------------------------------------------------------
    # 1         | 2          | 2023-01-02 | 5          | NaN   | NaN   | NaN   | NaN   | NaN         | NaN
    # 1         | 2          | 2023-01-09 | 10         | 5     | NaN   | NaN   | NaN   | NaN         | NaN
    # 1         | 2          | 2023-01-16 | 15         | 10    | 5     | NaN   | NaN   | 7.50        | 5
    # 1         | 2          | 2023-01-23 | 12         | 15    | 10    | 5     | NaN   | 10.00       | 5
    # 1         | 2          | 2023-01-30 | 20         | 12    | 15    | 10    | 5     | 10.50       | -3
    # 1         | 2          | 2023-02-06 | 18         | 20    | 12    | 15    | 10    | 14.25       | 8
    # 1         | 2          | 2023-02-13 | 25         | 18    | 20    | 12    | 15    | 16.25       | -2

    return data.reset_index(drop=True)