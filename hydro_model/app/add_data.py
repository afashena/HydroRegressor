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
    def __init__(self, export_ext: str = "csv"):
        self.dataset_id : str | None = None
        self.export_ext = export_ext
        self.db_env = DBEnv()


    def get_online_data(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.url = f"https://data.townofcary.org/api/explore/v2.1/catalog/datasets/{self.dataset_id}/exports/{self.export_ext}"
        self.data = requests.get(self.url).text.splitlines()

    def preprocess_data(self):
        
        for i, data_item in enumerate(self.data):
            if i == 0:
                continue
            prefix, separator, suffix = data_item.rpartition(';')
            string_geopoint = f"\"{suffix}\""
            self.data[i] = prefix + separator + string_geopoint
        
        csv_data = "\n".join(self.data)
        self.data = str.replace(csv_data, ";", ",")

    def write_data(self, table_name: str):

        buffer = StringIO(self.data)
        print("Made buffer")

        with psycopg.connect(f"dbname={self.db_env.POSTGRES_DB} user={self.db_env.POSTGRES_USER} password={self.db_env.POSTGRES_PASSWORD} host={self.db_env.POSTGRES_HOST} port={self.db_env.POSTGRES_PORT}") as conn:
            with conn.cursor() as cur:
                cur.execute("""
                        SELECT string_agg(column_name, ', ' ORDER BY ordinal_position)
                        FROM information_schema.columns
                        WHERE table_name = %s
                        AND table_schema = 'public'
                    """, (table_name,))

                columns = cur.fetchone()[0]

                cur.execute(f"""
                    CREATE TEMP TABLE temp_{table_name}
                    (LIKE {table_name})
                    ON COMMIT DROP
                """)

                # 3. COPY into temp table
                copy_sql = f"""
                    COPY temp_{table_name} ({columns})
                    FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
                """

                with cur.copy(copy_sql) as copy:
                    copy.write(buffer.read())

                print("Copied into temp table")

                # 4. Insert into real table, ignore duplicates
                cur.execute(f"""
                    INSERT INTO {table_name}
                    SELECT * FROM temp_{table_name}
                    ON CONFLICT (unique_record_id) DO NOTHING
                """)
                
                print("Copied")
                # sanity check
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                print(f"Rows inserted into {table_name}:", cur.fetchone()[0])
            conn.commit()
            print("Committed.")

    def get_data(self, dataset_id: str, table_name: str):
        self.get_online_data(dataset_id)
        self.preprocess_data()
        self.write_data(table_name)

    def update_db(self):
        self.get_data(dataset_id="rain-gages", table_name="rain_gage_data")
        self.get_data(dataset_id="stream-gages", table_name="stream_gage_data")
        print("Updated DB.")



if __name__ == "__main__":
    db_updater = DBUpdater()
    db_updater.update_db()
    print("Done!")