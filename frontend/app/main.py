import requests

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("Energy Consumption")

# Input area
area_response = requests.get(f"http://localhost:8001/api/v1/area_values/")
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
consumer_type_response = requests.get(f"http://localhost:8001/api/v1/consumer_type_values/")
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


# Check both area and consumer type have values listed
if area and consumer_type: 

    response = requests.get(f"http://host.docker.internal:8001/api/v1/get_predictions/")
    json_response = response.json()

    datetime_utc = json_response.get("datetime_utc")
    area_ = json_response.get("area")
    consumer_type_ = json_response.get("consumer_type")
    energy_consumption = json_response.get("energy_consumption")

    pred_datetime_utc = json_response.get("preds_datetime_utc")
    pred_area_ = json_response.get("preds_area")
    pred_consumer_type_ = json_response.get("preds_consumer_type")
    pred_energy_consumption = json_response.get("preds_energy_consumption")

    train_df = pd.DataFrame(list(
        zip(
        datetime_utc, area_, consumer_type_, energy_consumption
        )
        ),
        columns=["datetime_utc", "area", "consumer_type", "energy_consumption"]
        )
    
    preds_df = pd.DataFrame(list(
        zip(
        pred_datetime_utc, pred_area_, pred_consumer_type_, pred_energy_consumption
        )
        ),
        columns=["datetime_utc", "area", "consumer_type", "energy_consumption"]
        )

    # Get specific columns 
    train_area_and_consumer_df = train_df.loc[
        (train_df["area"] == area) & (train_df["consumer_type"] == consumer_type) 
    ].copy() 

    pred_area_and_consumer_df = preds_df.loc[
        (preds_df["area"] == area) & (preds_df["consumer_type"] == consumer_type) 
    ].copy() 


    train_area_and_consumer_df["datetime_utc"] = pd.to_datetime(train_area_and_consumer_df["datetime_utc"], unit="h")
    train_area_and_consumer_df.set_index("datetime_utc", inplace=True)

    pred_area_and_consumer_df["datetime_utc"] = pd.to_datetime(pred_area_and_consumer_df["datetime_utc"], unit="h")
    pred_area_and_consumer_df.set_index("datetime_utc", inplace=True)


    fig, ax = plt.subplots()
    ax.plot(train_area_and_consumer_df.index, train_area_and_consumer_df["energy_consumption"], color="blue", label="current")
    ax.plot(pred_area_and_consumer_df.index, pred_area_and_consumer_df["energy_consumption"], color="red", label="prediction")
    st.pyplot(fig) 