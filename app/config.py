import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL_TAP1 = os.environ["DATABASE_URL_TAP1"]
DATABASE_URL_TAP2 = os.environ["DATABASE_URL_TAP2"]
BUSINESS_ID = int(os.getenv("BUSINESS_ID", "1"))