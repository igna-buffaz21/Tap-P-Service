import psycopg

from app.config import DATABASE_URL_TAP1
from app.config import DATABASE_URL_TAP2


def get_connection_tap1():
    return psycopg.connect(DATABASE_URL_TAP1)

def get_connection_tap2():
    return psycopg.connect(DATABASE_URL_TAP2)