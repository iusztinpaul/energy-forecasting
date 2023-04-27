import requests

import pandas as pd
import plotly.graph_objects as go

from settings import API_URL


def build_data_plot(area: int, consumer_type: int):
    """
    Build plotly graph for data.
    """

    # Get predictions from API.
    response = requests.get(
        API_URL / "predictions" / f"{area}" / f"{consumer_type}", verify=False
    )
    json_response = response.json()

    datetime_utc = json_response.get("datetime_utc")
    energy_consumption = json_response.get("energy_consumption")
    pred_datetime_utc = json_response.get("preds_datetime_utc")
    pred_energy_consumption = json_response.get("preds_energy_consumption")

    # Prepare data for plotting.
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

    # Create plot.
    fig = go.Figure()
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
        x=train_df["datetime_utc"],
        y=train_df["energy_consumption"],
        name="Observations",
        line=dict(color="#C4B6B6"),
        hovertemplate="<br>".join(["Datetime: %{x}", "Energy Consumption: %{y} kWh"]),
    )
    fig.add_scatter(
        x=preds_df["datetime_utc"],
        y=preds_df["energy_consumption"],
        name="Predictions",
        line=dict(color="#FFC703"),
        hovertemplate="<br>".join(["Datetime: %{x}", "Total Consumption: %{y} kWh"]),
    )

    return fig
