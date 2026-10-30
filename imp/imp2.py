
from datetime import date

import requests

from app.database import get_connection


API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = date(2013, 1, 1)
END_DATE = date(2017, 8, 15)

TIMEZONE = "America/Guayaquil"


def import_historical_weather(location_id: int):
    """Descarga y almacena el clima histórico de una ubicación."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT latitude, longitude
                FROM weather_location
                WHERE id = %s
                """,
                (location_id,),
            )

            location = cur.fetchone()

            if location is None:
                raise ValueError("La ubicación meteorológica no existe.")

            latitude, longitude = location

        params = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "start_date": START_DATE.isoformat(),
            "end_date": END_DATE.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": TIMEZONE,
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=120,
        )
        response.raise_for_status()

        daily = response.json()["daily"]

        dates = daily["time"]
        temperatures = daily["temperature_2m_mean"]
        precipitations = daily["precipitation_sum"]

        if not (
            len(dates) == len(temperatures) == len(precipitations)
        ):
            raise ValueError("La respuesta meteorológica es inconsistente.")

        records = list(
            zip(dates, temperatures, precipitations)
        )

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO historical_weather (
                    location_id,
                    weather_date,
                    temperature_mean,
                    precipitation_sum
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (location_id, weather_date)
                DO UPDATE SET
                    temperature_mean = EXCLUDED.temperature_mean,
                    precipitation_sum = EXCLUDED.precipitation_sum
                """,
                [
                    (location_id, weather_date, temperature, precipitation)
                    for weather_date, temperature, precipitation in records
                ],
            )

        conn.commit()

    print(f"Clima importado: {len(records)} días.")


if __name__ == "__main__":
    import_historical_weather(location_id=1)
