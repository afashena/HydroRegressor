import pandas as pd
import streamlit as st
import altair as alt

from app.db_utils.database import DB
from app.UI.utils import parse_site_ids
import psycopg
from datetime import datetime
from typing import Literal
import subprocess

def get_data(table_name: str, site_ids: str, db: DB) -> pd.DataFrame:

    # Get data from the DB (normal first, then try cache)
    data = db.querier.get_data_from_range(table_name=table_name, site_id=site_ids[0])

    # Plot data
    with psycopg.connect(db.updater.conn_info) as conn:
            with conn.cursor() as cur:
                all_columns = db.updater.get_table_columns(table_name=table_name, cur=cur).split(sep=", ")
                print(all_columns)
    return pd.DataFrame(data, columns=all_columns)

def make_plot(chart_data: pd.DataFrame, site_ids: list[str], title: str, y_axis: str):
    # make an altair chart
    hover = alt.selection_point(
        fields=["collect_date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(chart_data, title=f"{title} (in) at Site {site_ids[0]}")
        .mark_line()
        .encode(
            x="collect_date",
            y=y_axis,
        )
    )

    points = lines.transform_filter(hover).mark_circle(size=65)

    tooltips = (
            alt.Chart(chart_data)
            .mark_rule()
            .encode(
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("collect_date", title="Date"),
                    alt.Tooltip(y_axis, title=f"{title} (inches)"),
                ],
            )
            .add_params(hover)
        )

    data_layer = lines + points + tooltips
    st.altair_chart(data_layer, use_container_width=True)

def plot_data(input_site_ids: str, db: DB, table_name: str, title: str, y_axis: str):
    if input_site_ids != "":
        # Parse given site IDs
        site_ids = parse_site_ids(input_site_ids)

        chart_data = get_data(table_name=table_name, site_ids=site_ids, db=db)

        # make an altair chart
        make_plot(chart_data=chart_data, site_ids=site_ids, title=title, y_axis=y_axis)

db = DB()

st.title("🌧️ Hydroregressor Dashboard 🎛️")

with st.expander(label="Forecast", icon=":material/emoji_objects:"):
    a = 5
    time_hrzn = st.text_input("Enter the forecast time horizon (in hours): ", value="5")
    st.write("Selected value:", time_hrzn)

with st.expander(label="Data Visualization", icon=":material/insert_chart:"):

    rain_site_id = st.text_area("Enter the Rain Gage Site IDs: ", value="35771767874133")
    st.write("Selected value:", rain_site_id)

    stream_site_id = st.text_area("Enter the Stream Gage Site IDs: ", value="")
    st.write("Selected value:", stream_site_id)


    plot_data(input_site_ids=rain_site_id, db=db, table_name="rain_gage_data", title="Rain Amount", y_axis="rain_amount")
    #plot_data(input_site_ids=stream_site_id, db=db, table_name="stream_gage_data", title="Stream Height", y_axis="stream_height")

with st.expander(label="Database", icon=":material/database:"):
    st.write('''
        Here you can do database things.
    ''')
    st.button("Update DB", on_click=db.updater.update_db)

if __name__ == "__main__":
    subprocess.run(["streamlit", "run", "app/UI/main.py"])