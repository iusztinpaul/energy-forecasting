"""
NOTE: Scratchpad blocks are used only for experimentation and testing out code.
The code written here will not be executed as part of the pipeline.
"""
from mage_ai.data_preparation.variable_manager import get_variable

import matplotlib.pyplot as plt


df = get_variable('energy_consumption', 'remove_extra_data', 'output_0')

df = df[(df["area"] == "DK1") & (df["consumer_type"] == "111")]

fig, ax = plt.subplots(figsize=(20, 6))
ax = df.plot(x="datetime_utc", y="energy_consumption", ax=ax)
df.plot(x="datetime_utc", y="energy_consumption_rolling_average_days_1", ax=ax)
df.plot(x="datetime_utc", y="energy_consumption_lagged_hours_22", ax=ax)