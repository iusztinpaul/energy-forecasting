import requests

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


API_URL = "http://172.17.0.1:8001/api/v1"

st.title("Energy Consumption")

# Input area
area_response = requests.get(f"{API_URL}/area_values/")
json_area_response = area_response.json()

area = st.selectbox(
    label= "Denmark is divided in two price areas, or bidding zones,\
        divided by the Great Belt. DK1 (shown as 1) is west of the Great Belt \
            and DK2 (shown as 2) is east of the Great Belt.",
    options=(
    json_area_response.get("values")
    )
)

# Input consumer_type
consumer_type_response = requests.get(f"{API_URL}/consumer_type_values/")
json_consumer_type_response = consumer_type_response.json()

consumer_type = st.selectbox(
    label="The consumer type is the Industry Code DE35 which is owned\
          and maintained by Danish Energy, a non-commercial lobby\
              organization for Danish energy compa-nies. \
                The code is used by Danish energy companies.",
    options=(
    json_consumer_type_response.get("values")
    )
)

input_data = {
    "area": area,
    "consumer_type": consumer_type
}

# Check both area and consumer type have values listed
if st.button("Get Predictions"): 

    response = requests.get(
        f"{API_URL}/predictions/{area}/{consumer_type}", 
        verify=False
        )
    
    json_response = response.json()

    datetime_utc = json_response.get("datetime_utc")
    energy_consumption = json_response.get("energy_consumption")
    pred_datetime_utc = json_response.get("preds_datetime_utc")
    pred_energy_consumption = json_response.get("preds_energy_consumption")

    train_df = pd.DataFrame(list(
        zip(
            datetime_utc, energy_consumption
        )
        ),
        columns=["datetime_utc", "energy_consumption"]
        )
    
    preds_df = pd.DataFrame(list(
        zip(
            pred_datetime_utc, pred_energy_consumption
        )
        ),
        columns=["datetime_utc", "energy_consumption"]
        )

    train_df["datetime_utc"] = pd.to_datetime(train_df["datetime_utc"], unit="h")
    train_df.set_index("datetime_utc", inplace=True)

    preds_df["datetime_utc"] = pd.to_datetime(preds_df["datetime_utc"], unit="h")
    preds_df.set_index("datetime_utc", inplace=True)

    fig, ax = plt.subplots()
    ax.plot(train_df.index, train_df["energy_consumption"], color="blue", label="current")
    ax.plot(preds_df.index, preds_df["energy_consumption"], color="red", label="prediction")
    st.pyplot(fig) 