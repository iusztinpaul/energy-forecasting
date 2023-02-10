from mage_ai.data_preparation.variable_manager import get_variable


df = get_variable('energy_consumption', 'compute_future_values', 'output_0')

df = df[(df["Area"] == "DK1") & (df["ConsumerType"] == "111")]

df[["UTCDatetime", "EnergyConsumption", "EnergyConsumptionFutureHours1", "EnergyConsumptionFutureHours2", "EnergyConsumptionFutureHours24"]].tail(n=24)