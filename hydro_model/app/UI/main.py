from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

from app.db_utils.database import DB
from app.UI.utils import parse_site_ids
from app.config import Config
import psycopg
from datetime import datetime
from typing import Literal
import subprocess

from app.train_narx import test_forecast

config_path = Path(__file__).parent.parent / "config.json"
print(config_path)
config = Config.model_validate_json(config_path.read_text())

### HELPER METHODS ###

def get_data(table_name: str, site_ids: list[str], db: DB, timestamp: datetime = datetime.now(), ineq: Literal["before", "after"] = "before") -> list[pd.DataFrame]:

    df_list = []

    for site_id in site_ids:
        print(f"Getting data for site ID {site_id} from table {table_name}...")

        # Get data from the DB (normal first, then try cache later on)
        data = db.querier.get_data_from_range(table_name=table_name, site_id=site_id, timestamp=timestamp, ineq=ineq)

        # Plot data
        with psycopg.connect(db.updater.conn_info) as conn:
                with conn.cursor() as cur:
                    all_columns = db.updater.get_table_columns(table_name=table_name, cur=cur).split(sep=", ")
                    print(all_columns)

        df = pd.DataFrame(data, columns=all_columns)
        df_list += [df]
    return df_list

def make_plot(chart_data: list[pd.DataFrame], site_ids: list[str], title: str, y_axis: str):
    # Concatenate the DataFrames and add a site_id column
    combined_data = pd.concat(
        [df.assign(site_id=site_id) for df, site_id in zip(chart_data, site_ids)],
        ignore_index=True
    )

    # Create an Altair chart
    hover = alt.selection_point(
        fields=["collect_date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(combined_data, title=title)
        .mark_line()
        .encode(
            x="collect_date",
            y=y_axis,
            color="site_id:N",  # Use site_id for color encoding
        )
    )

    points = lines.transform_filter(hover).mark_circle(size=65)

    tooltips = (
        alt.Chart(combined_data)
        .mark_rule()
        .encode(
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("collect_date", title="Date"),
                alt.Tooltip(y_axis, title=f"{title} (inches)"),
                alt.Tooltip("site_id", title="Site ID"),
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


####### UI SCRIPT #######

db = DB()

st.title("🌧️ Hydroregressor Dashboard 🎛️")

with st.expander(label="Forecast", icon=":material/emoji_objects:"):
    rain_site_id = st.text_area("Enter the Rain Gage Site IDs for Forecast: ", value="35771767874133\n35785007877408")
    st.write("Selected value:", rain_site_id)

    stream_site_id = st.text_area("Enter the Stream Gage Site IDs for Forecast: ", value="35776827875152")
    st.write("Selected value:", stream_site_id)

    start_date = st.datetime_input("Enter the forecast start date and time: ", value=datetime(2026, 3, 1, 5, 36))
    st.write("Selected value:", start_date)
    time_hrzn = st.text_input("Enter the forecast time horizon (in hours): ", value="5")
    st.write("Selected value:", time_hrzn)

    # get data for forecast plot, taking into account the lag length of the model (i.e. the amount of recent history needed to make a forecast)
    extra_time_X = config.X_lag * config.sample_time
    extra_time_y = config.y_lag * config.sample_time
    start_date_X = start_date - pd.Timedelta(minutes=extra_time_X)
    start_date_y = start_date - pd.Timedelta(minutes=extra_time_y)

    end_date = start_date + pd.Timedelta(hours=int(time_hrzn))

    # Parse given site IDs
    rain_site_ids = parse_site_ids(rain_site_id)
    stream_site_ids = parse_site_ids(stream_site_id)

    rain_sensor_data = get_data(table_name="rain_gage_data", site_ids=rain_site_ids, db=db, timestamp=[start_date_X, end_date], ineq="after")
    stream_sensor_data = get_data(table_name="stream_gage_data", site_ids=stream_site_ids, db=db, timestamp=[start_date_y, end_date], ineq="after")

    print(stream_sensor_data[0].head())
    print(rain_sensor_data[0].head())

    # forecast predictions and plot them alongside the recent history data
    empty = False
    if not stream_sensor_data[0].empty:
        for df in rain_sensor_data:
            if df.empty:
                empty = True
        if not empty:
            print("Doing forecast")
            y_pred, y_true, mse = test_forecast(rain_sensor_data, stream_sensor_data, config)
            make_plot(chart_data=[pd.DataFrame(y_true), pd.DataFrame(y_pred)], site_ids=f"Stream sensor {stream_site_ids[0]}", title=f"Forecast of {time_hrzn} hours", y_axis="Rain level (inches)")


    
    # plot


with st.expander(label="Data Visualization", icon=":material/insert_chart:"):

    rain_site_id = st.text_area("Enter the Rain Gage Site IDs: ", value="35771767874133")
    st.write("Selected value:", rain_site_id)

    stream_site_id = st.text_area("Enter the Stream Gage Site IDs: ", value="35776827875152")
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