"""
NOTE: Scratchpad blocks are used only for experimentation and testing out code.
The code written here will not be executed as part of the pipeline.
"""
from mage_ai.data_preparation.variable_manager import get_variable

import matplotlib.pyplot as plt


df = get_variable('energy_consumption', 'remove_extra_data', 'output_0')

df = df[(df["Area"] == "DK1") & (df["ConsumerType"] == "111")]

fig, ax = plt.subplots(figsize=(20, 6))
ax = df.plot(x="UTCDatetime", y="EnergyConsumption", ax=ax)
df.plot(x="UTCDatetime", y="EnergyConsumptionRollingAverageDays1", ax=ax)
df.plot(x="UTCDatetime", y="EnergyConsumptionLaggedHours25", ax=ax)
df.plot(x="UTCDatetime", y="EnergyConsumptionFutureHours25", ax=ax)

df[["EnergyConsumption", "EnergyConsumptionLaggedHours1", "EnergyConsumptionFutureHours1"]].head(100)
