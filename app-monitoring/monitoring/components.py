import requests

import pandas as pd
import plotly.graph_objects as go

from settings import API_URL


def build_metrics_plot():
    """
    Build plotly graph for metrics.
    """

    response = requests.get(API_URL / "monitoring" / "metrics", verify=False)
    json_response = response.json()

    datetime_utc = json_response.get("datetime_utc", [])
    mape = json_response.get("mape", [])

    # Prepare data for plotting.
    metrics_df = pd.DataFrame(
        list(zip(datetime_utc, mape)),
        columns=["datetime_utc", "mape"],
    )
    metrics_df["datetime_utc"] = pd.to_datetime(metrics_df["datetime_utc"], unit="h")

    # Create plot.
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text="Predictions vs. Observations | Aggregated Metrics",
            font=dict(family="Arial", size=16),
        ),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Datetime UTC")
    fig.update_yaxes(title_text="MAPE")
    fig.add_scatter(
        x=metrics_df["datetime_utc"],
        y=metrics_df["mape"],
        name="MAPE",
        line=dict(color="#C4B6B6"),
        hovertemplate="<br>".join(["Datetime UTC: %{x}", "MAPE: %{y} kWh"]),
    )

    return fig


def build_data_plot(area: int, consumer_type: int):
    """
    Build plotly graph for data.
    """

    # Get predictions from API.
    response = requests.get(
        API_URL / "monitoring" / "values" / f"{area}" / f"{consumer_type}", verify=False
    )
    json_response = response.json()

    y_monitoring_datetime_utc = json_response.get("y_monitoring_datetime_utc", [])
    y_monitoring_energy_consumption = json_response.get(
        "y_monitoring_energy_consumption", []
    )
    predictions_monitoring_datetime_utc = json_response.get(
        "predictions_monitoring_datetime_utc", []
    )
    predictions_monitoring_energy_consumptionc = json_response.get(
        "predictions_monitoring_energy_consumptionc", []
    )

    # Prepare data for plotting.
    train_df = pd.DataFrame(
        list(zip(y_monitoring_datetime_utc, y_monitoring_energy_consumption)),
        columns=["datetime_utc", "energy_consumption"],
    )
    preds_df = pd.DataFrame(
        list(
            zip(
                predictions_monitoring_datetime_utc,
                predictions_monitoring_energy_consumptionc,
            )
        ),
        columns=["datetime_utc", "energy_consumption"],
    )

    train_df["datetime_utc"] = pd.to_datetime(train_df["datetime_utc"], unit="h")
    preds_df["datetime_utc"] = pd.to_datetime(preds_df["datetime_utc"], unit="h")

    # Create plot.
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text="Predictions vs. Observations | Energy Consumption",
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
        hovertemplate="<br>".join(
            ["Datetime UTC: %{x}", "Energy Consumption: %{y} kWh"]
        ),
    )
    fig.add_scatter(
        x=preds_df["datetime_utc"],
        y=preds_df["energy_consumption"],
        name="Predictions",
        line=dict(color="#FFC703"),
        hovertemplate="<br>".join(
            ["Datetime UTC: %{x}", "Total Consumption: %{y} kWh"]
        ),
    )

    return fig
