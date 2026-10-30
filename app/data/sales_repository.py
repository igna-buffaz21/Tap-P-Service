
import pandas as pd

from app.database import get_connection
from app.ml.preprocessing import TEST_START


def get_weekly_sales(
    business_id: int,
    branch_id: int = 44,
    product_limit: int = 100,
) -> pd.DataFrame:
    """Obtiene ventas semanales de los productos más vendidos de una sucursal."""

    if product_limit <= 0:
        raise ValueError("product_limit debe ser mayor que cero.")

    query = """
        SELECT
            hs.branch_id,
            hs.product_id,
            DATE_TRUNC('week', hs.sale_date)::date AS week_start,
            SUM(hs.units_sold) AS units_sold
        FROM historical_sale hs
        WHERE hs.branch_id = %s
        GROUP BY
            hs.branch_id,
            hs.product_id,
            DATE_TRUNC('week', hs.sale_date)::date
        ORDER BY
            hs.product_id,
            week_start;
    """

    # Fecha de corte para seleccionar productos sin utilizar
    # información del período de evaluación.
    selection_end_date = TEST_START.date()

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Verificar que la sucursal pertenezca al negocio.
            cur.execute(
                """
                SELECT 1
                FROM branch
                WHERE id = %s AND business_id = %s
                """,
                (branch_id, business_id),
            )

            if cur.fetchone() is None:
                raise ValueError(
                    "La sucursal no pertenece al negocio indicado."
                )

            cur.execute(query, (branch_id,))

            rows = cur.fetchall()
            columns = [
                column.name
                for column in cur.description
            ]

    df = pd.DataFrame(rows, columns=columns)

    df["week_start"] = pd.to_datetime(df["week_start"])
    df["units_sold"] = pd.to_numeric(df["units_sold"])

    return df


#branch_id	product_id	week_start	units_sold
#44	        21	        2012-12-31	621.776
#44	        21	        2013-01-07	584.844
#44	        21	        2013-01-14	510.356
#44	        21	        2013-01-21	427.762
#44	        21	        2013-01-28	532.197