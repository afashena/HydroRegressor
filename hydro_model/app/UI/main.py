import pandas as pd
import streamlit as st
import altair as alt

from app.db_utils.database import DB
from app.UI.utils import parse_site_ids
import psycopg
from datetime import datetime
from typing import Literal
import subprocess

db = DB()

st.title("🌧️ Hydroregressor Dashboard 🎛️")

with st.expander(label="Data Visualization", icon=":material/insert_chart:"):

    rain_site_id = st.text_area("Enter the Rain Gage Site IDs: ", value="")
    st.write("Selected value:", rain_site_id)

    stream_site_id = st.text_area("Enter the Stream Gage Site IDs: ", value="")
    st.write("Selected value:", stream_site_id)


    if rain_site_id != "":

        # Parse given site IDs
        rain_site_ids = parse_site_ids(rain_site_id)
        #stream_site_ids = parse_site_ids(stream_site_id)

        # Get data from the DB (normal first, then try cache)
        rain_gage_data = db.querier.get_data_from_range(table_name="rain_gage_data", site_id=rain_site_ids[0])

        # Plot data
        with psycopg.connect(db.updater.conn_info) as conn:
                with conn.cursor() as cur:
                    all_columns = db.updater.get_table_columns(table_name="rain_gage_data", cur=cur).split(sep=", ")
                    print(all_columns)
        chart_data = pd.DataFrame(rain_gage_data, columns=all_columns)

        # make an altair chart
        hover = alt.selection_point(
            fields=["collect_date"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        lines = (
            alt.Chart(chart_data, title=f"Rain Amount (in) at Site {rain_site_ids[0]}")
            .mark_line()
            .encode(
                x="collect_date",
                y="rain_amount",
            )
        )

        points = lines.transform_filter(hover).mark_circle(size=65)

        tooltips = (
                alt.Chart(chart_data)
                .mark_rule()
                .encode(
                    # x="yearmonthdate(collect_date)",
                    # y="rain_amount",
                    opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                    tooltip=[
                        alt.Tooltip("collect_date", title="Date"),
                        alt.Tooltip("rain_amount", title="Rain Amount (inches)"),
                    ],
                )
                .add_params(hover)
            )

        data_layer = lines + points + tooltips
        st.altair_chart(data_layer, use_container_width=True)

with st.expander(label="Database", icon=":material/database:"):
    st.write('''
        Here you can do database things.
    ''')
    st.button("Update DB", on_click=db.updater.update_db)

if __name__ == "__main__":
    subprocess.run(["streamlit", "run", "app/UI/main.py"])