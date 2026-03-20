import pandas as pd
import streamlit as st

from app.db_utils.database import DB
from app.UI.utils import parse_site_ids
import psycopg
from datetime import datetime
from typing import Literal
import subprocess

db = DB()

st.title("🌧️ Hydroregressor Dashboard 🎛️")

rain_site_id = st.sidebar.text_area("Enter the Rain Gage Site IDs: ", value="")
st.sidebar.write("Selected value:", rain_site_id)

stream_site_id = st.sidebar.text_area("Enter the Stream Gage Site IDs: ", value="")
st.sidebar.write("Selected value:", stream_site_id)

if rain_site_id != "":

     # Parse given site IDs
     rain_site_ids = parse_site_ids(rain_site_id)
     stream_site_ids = parse_site_ids(stream_site_id)

     # Get data from the DB (normal first, then try cache)
     rain_gage_data = db.querier.get_data_from_range(table_name="rain_gage_data", site_id=rain_site_ids[0])

     # Plot data
     with psycopg.connect(db.updater.conn_info) as conn:
            with conn.cursor() as cur:
               all_columns = db.updater.get_table_columns(table_name="rain_gage_data", cur=cur).split(sep=", ")
               print(all_columns)
     chart_data = pd.DataFrame(rain_gage_data, columns=all_columns)
     st.line_chart(chart_data.set_index("collect_date")["rain_amount"])

if __name__ == "__main__":
    subprocess.run(["streamlit", "run", "app/UI/main.py"])