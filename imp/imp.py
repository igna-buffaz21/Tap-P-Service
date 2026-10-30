
import csv
from datetime import date
from pathlib import Path

from app.config import BUSINESS_ID
from app.database import get_connection


CSV_PATH = Path.home() / (
    "Descargas/favorita-grocery-sales-forecasting/train.csv"
)

BATCH_SIZE = 100_000
START_DATE = date(2015, 9, 27)


def load_mappings(cur):
    """Relaciona los identificadores de Favorita con los IDs internos."""

    cur.execute(
        """
        SELECT external_reference, id
        FROM branch
        WHERE business_id = %s
        """,
        (BUSINESS_ID,),
    )
    branches = {str(ref): id_ for ref, id_ in cur.fetchall()}

    cur.execute(
        """
        SELECT sku, id
        FROM product
        WHERE business_id = %s
        """,
        (BUSINESS_ID,),
    )
    products = {str(sku): id_ for sku, id_ in cur.fetchall()}

    return branches, products


def import_sales():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    with get_connection() as conn:
        with conn.cursor() as cur:
            branches, products = load_mappings(cur)

            if not branches or not products:
                raise ValueError("Faltan sucursales o productos importados.")

            # Permitir continuar la carga sin duplicar el período pendiente.
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM historical_sale AS sale
                    JOIN branch ON branch.id = sale.branch_id
                    WHERE branch.business_id = %s
                      AND sale.sale_date >= %s
                )
                """,
                (BUSINESS_ID, START_DATE),
            )
            if cur.fetchone()[0]:
                raise ValueError(
                    f"Ya hay ventas del negocio desde {START_DATE}. "
                    "No se continuará la importación para evitar duplicados."
                )

        total = 0

        with CSV_PATH.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            iterator = iter(reader)

            while True:
                batch = []

                while len(batch) < BATCH_SIZE:
                    row = next(iterator, None)
                    if row is None:
                        break

                    if date.fromisoformat(row["date"]) < START_DATE:
                        continue

                    branch_id = branches.get(row["store_nbr"])
                    product_id = products.get(row["item_nbr"])

                    if branch_id is None or product_id is None:
                        raise ValueError(
                            "Referencia desconocida: "
                            f"store={row['store_nbr']}, "
                            f"item={row['item_nbr']}"
                        )

                    promotion = row["onpromotion"].strip().lower()
                    if promotion not in ("", "true", "false"):
                        raise ValueError(
                            f"Valor inválido de onpromotion: {promotion}"
                        )

                    batch.append((
                        branch_id,
                        product_id,
                        row["date"],
                        row["unit_sales"],
                        None if promotion == "" else promotion == "true",
                    ))

                if not batch:
                    break

                with conn.cursor() as cur:
                    with cur.copy(
                        """
                        COPY historical_sale (
                            branch_id,
                            product_id,
                            sale_date,
                            units_sold,
                            onpromotion
                        )
                        FROM STDIN
                        """
                    ) as copy:
                        for record in batch:
                            copy.write_row(record)

                conn.commit()
                total += len(batch)

                if total % 1_000_000 == 0:
                    print(f"Importados: {total:,}", flush=True)

                if len(batch) < BATCH_SIZE:
                    break

    print(f"Importación finalizada: {total:,} registros.")


if __name__ == "__main__":
    import_sales()
