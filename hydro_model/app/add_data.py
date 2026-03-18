from dataclasses import dataclass
import os
from io import StringIO

import requests
import psycopg

@dataclass
class DBEnv():
    POSTGRES_DB: str = os.getenv('POSTGRES_DB')
    POSTGRES_USER: str = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD: str = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_HOST: str = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT: str = os.getenv('POSTGRES_PORT')

class DBUpdater():
    def __init__(self, dataset_id: str, export_ext: str = "csv"):
        self.dataset_id = dataset_id
        self.export_ext = export_ext
        self.url = f"https://data.townofcary.org/api/explore/v2.1/catalog/datasets/{self.dataset_id}/exports/{self.export_ext}"
        self.db_env = DBEnv()


    def get_online_data(self):
        self.data = requests.get(self.url).text.splitlines()

    def get_raingage_data(self):
        return
    
    def get_stormdrain_data(self):
        return

    def preprocess_data(self):
        # This function will find the data for the desired sensors and organize
        # it according to the rainfall_data schema.
        self.data = self.data

    def update_db(self):
        
        csv_data = "\n".join(self.data)

        with psycopg.connect(f"dbname={self.db_env.POSTGRES_DB} user={self.db_env.POSTGRES_USER} host={self.db_env.POSTGRES_HOST}") as conn:
            with conn.cursor() as cur:
                buffer = StringIO(csv_data)

                cur.copy(
                    "COPY rainfall_data FROM STDIN WITH (FORMAT CSV, HEADER TRUE)",
                    buffer
                )
            conn.commit()