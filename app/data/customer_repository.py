
import pandas as pd

from app.database import get_connection_tap2


def get_customer_purchases(business_id: int) -> pd.DataFrame:
    """Obtiene el historial de compras de clientes de un negocio."""

    query = """
        SELECT
            s.customer_id,
            s.branch_id,
            s.created_at AT TIME ZONE
                'America/Argentina/Buenos_Aires' AS purchase_date,
            s.total_amount
        FROM sale s
        WHERE s.business_id = %s
          AND s.status = 'COMPLETED'
          AND s.customer_id IS NOT NULL
        ORDER BY s.customer_id, s.created_at
    """

    with get_connection_tap2() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (business_id,))
            rows = cur.fetchall()
            columns = [column.name for column in cur.description]

    df = pd.DataFrame(rows, columns=columns)

    if not df.empty:
        df["purchase_date"] = pd.to_datetime(df["purchase_date"])
        df["total_amount"] = pd.to_numeric(df["total_amount"])

    return df
