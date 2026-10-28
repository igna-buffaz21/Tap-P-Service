import pandas as pd

from app.database import get_connection


def get_weekly_sales(business_id: int) -> pd.DataFrame:
    """Obtiene las unidades vendidas por producto, sucursal y semana."""

    query = """
        SELECT
            s.branch_id,
            si.product_id,
            DATE_TRUNC(
                'week',
                s.created_at AT TIME ZONE
                'America/Argentina/Buenos_Aires'
            )::date AS week_start,
            SUM(si.quantity) AS units_sold
        FROM sale s
        INNER JOIN sale_item si ON si.sale_id = s.id
        WHERE s.business_id = %s
          AND s.status = 'COMPLETED'
        GROUP BY
            s.branch_id,
            si.product_id,
            week_start
        ORDER BY
            s.branch_id,
            si.product_id,
            week_start
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (business_id,))
            rows = cur.fetchall()
            columns = [column.name for column in cur.description]

    df = pd.DataFrame(rows, columns=columns)

    df["week_start"] = pd.to_datetime(df["week_start"])
    df["units_sold"] = pd.to_numeric(df["units_sold"])

    return df