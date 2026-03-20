from dataclasses import dataclass
import os
from io import StringIO
from typing import Literal

import requests
import psycopg
from datetime import datetime

@dataclass
class DBEnv():
    POSTGRES_DB: str = os.getenv('POSTGRES_DB')
    POSTGRES_USER: str = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD: str = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_HOST: str = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT: str = os.getenv('POSTGRES_PORT')

class DBUpdater():
    def __init__(self, conn_info: str, export_ext: str = "csv"):
        self.dataset_id : str | None = None
        self.export_ext = export_ext
        self.conn_info = conn_info


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

    def get_table_columns(self, table_name: str, cur: psycopg.Cursor) -> str:
        cur.execute("""
                SELECT string_agg(column_name, ', ' ORDER BY ordinal_position)
                FROM information_schema.columns
                WHERE table_name = %s
                AND table_schema = 'public'
            """, (table_name,))

        columns = cur.fetchone()[0]
        return columns
    
    def add_data_to_temp_table(self, table_name: str, columns: str, cur: psycopg.Cursor, buffer: StringIO):
        
        # make temp table
        cur.execute(f"""
            CREATE TEMP TABLE temp_{table_name}
            (LIKE {table_name})
            ON COMMIT DROP
        """)

        # COPY into temp table
        copy_sql = f"""
            COPY temp_{table_name} ({columns})
            FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
        """

        with cur.copy(copy_sql) as copy:
            copy.write(buffer.read())

        print("Copied into temp table")

    def write_data(self, table_name: str):

        buffer = StringIO(self.data)

        with psycopg.connect(self.conn_info) as conn:
            with conn.cursor() as cur:
                
                # get column names of table
                columns = self.get_table_columns(table_name, cur)

                # Write to temp table
                self.add_data_to_temp_table(table_name, columns, cur, buffer)

                # Insert into real table, ignore duplicates
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

class DBQuerier():
    def __init__(self, conn_info: str):
        self.conn_info = conn_info

    def get_data_from_range(self, table_name: str, site_id: int, timestamp: datetime | list[datetime] = datetime.now(), ineq: Literal["before", "after"] = "before"):
        
        # in the case where you want data from either before or after a single timestamp
        if ineq == "before":
            ineq_symbol = "<="
        else:
            ineq_symbol = ">="

        if isinstance(timestamp, list):
            sql = f"""
                SELECT *
                FROM {table_name}
                WHERE site_id = {site_id}
                AND collect_date BETWEEN '{timestamp[0]}' AND '{timestamp[1]}'
                ORDER BY collect_date"""
        else:
            sql = f"""
                SELECT *
                FROM {table_name}
                WHERE site_id = {site_id}
                AND collect_date {ineq_symbol} '{timestamp}'
                ORDER BY collect_date"""
            
        with psycopg.connect(self.conn_info) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                data = cur.fetchall()

        return data

class DB():
    def __init__(self):
        self.db_env = DBEnv()
        self.conn_info = f"dbname={self.db_env.POSTGRES_DB} user={self.db_env.POSTGRES_USER} password={self.db_env.POSTGRES_PASSWORD} host={self.db_env.POSTGRES_HOST} port={self.db_env.POSTGRES_PORT}"
        self.updater = DBUpdater(conn_info=self.conn_info)
        self.querier = DBQuerier(conn_info=self.conn_info)

if __name__ == "__main__":
    db = DB()
    #db.updater.update_db()
    db.querier.get_data_from_range(table_name="rain_gage_data", site_id=35771767874133)
    print("Done!")