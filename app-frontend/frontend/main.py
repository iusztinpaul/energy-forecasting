import requests

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# TODO: Load this from a config file
API_URL = "http://172.17.0.1:8001/api/v1"

st.title("Energy Consumption")

# Input area
area_response = requests.get(f"{API_URL}/area_values/")
json_area_response = area_response.json()

area = st.selectbox(
    label="Denmark is divided in two price areas, or bidding zones,\
        divided by the Great Belt. DK1 (shown as 1) is west of the Great Belt \
            and DK2 (shown as 2) is east of the Great Belt.",
    options=(json_area_response.get("values")),
)

# Input consumer_type
consumer_type_response = requests.get(f"{API_URL}/consumer_type_values/")
json_consumer_type_response = consumer_type_response.json()

consumer_type = st.selectbox(
    label="The consumer type is the Industry Code DE35 which is owned\
          and maintained by Danish Energy, a non-commercial lobby\
              organization for Danish energy compa-nies. \
                The code is used by Danish energy companies.",
    options=(json_consumer_type_response.get("values")),
)

input_data = {"area": area, "consumer_type": consumer_type}

# Check both area and consumer type have values listed
if area and consumer_type:
    response = requests.get(
        f"{API_URL}/predictions/{area}/{consumer_type}", verify=False
    )

    json_response = response.json()

    datetime_utc = json_response.get("datetime_utc")
    energy_consumption = json_response.get("energy_consumption")
    pred_datetime_utc = json_response.get("preds_datetime_utc")
    pred_energy_consumption = json_response.get("preds_energy_consumption")

    train_df = pd.DataFrame(
        list(zip(datetime_utc, energy_consumption)),
        columns=["datetime_utc", "energy_consumption"],
    )

    preds_df = pd.DataFrame(
        list(zip(pred_datetime_utc, pred_energy_consumption)),
        columns=["datetime_utc", "energy_consumption"],
    )

    train_df["datetime_utc"] = pd.to_datetime(train_df["datetime_utc"], unit="h")
    preds_df["datetime_utc"] = pd.to_datetime(preds_df["datetime_utc"], unit="h")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=train_df["datetime_utc"],
                y=train_df["energy_consumption"],
                line=dict(color="blue"),
                name="Observations",
                hovertemplate="<br>".join(
                    ["Datetime: %{x}", "Energy Consumption: %{y} kWh"]
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Energy Consumption per DE35 Industry Code per Hour",
            font=dict(family="Arial", size=16),
        ),
        showlegend=True,
    )

    fig.update_xaxes(title_text="Datetime UTC")
    fig.update_yaxes(title_text="Total Consumption")

    fig.add_scatter(
        x=preds_df["datetime_utc"],
        y=preds_df["energy_consumption"],
        name="Predictions",
        line=dict(color="red"),
        hovertemplate="<br>".join(["Datetime: %{x}", "Total Consumption: %{y} kWh"]),
    )

    st.plotly_chart(fig)
