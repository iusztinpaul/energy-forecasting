"""
NOTE: Scratchpad blocks are used only for experimentation and testing out code.
The code written here will not be executed as part of the pipeline.
"""
from mage_ai.data_preparation.variable_manager import get_variable


df = get_variable('energy_consumption', 'compute_rolling_average', 'output_0')

df = df[(df["PriceArea"] == "DK1") & (df["ConsumerType_DE35"] == "111")]

df.plot(y="TotalCon")

df.head(n=10)
