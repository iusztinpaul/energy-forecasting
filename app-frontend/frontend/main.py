import requests

import streamlit as st

from settings import API_URL, TITLE
from components import build_data_plot


st.set_page_config(page_title=TITLE)
st.title(TITLE)


# Create dropdown for area selection.
area_response = requests.get(API_URL / "area_values")

area = st.selectbox(
    label="Denmark is divided in two price areas, or bidding zones,\
        divided by the Great Belt. DK1 (shown as 1) is west of the Great Belt \
            and DK2 (shown as 2) is east of the Great Belt.",
    options=area_response.json().get("values", []),
)

# Create dropdown for consumer type selection.
consumer_type_response = requests.get(API_URL / "consumer_type_values")

consumer_type = st.selectbox(
    label="The consumer type is the Industry Code DE35 which is owned \
          and maintained by Danish Energy, a non-commercial lobby \
              organization for Danish energy companies. \
                The code is used by Danish energy companies.",
    options=consumer_type_response.json().get("values", []),
)


# Check if both area and consumer type have values listed, then create plot for data.
if area and consumer_type:
    st.plotly_chart(build_data_plot(area, consumer_type))
