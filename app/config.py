import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
BUSINESS_ID = int(os.getenv("BUSINESS_ID", "1"))