"""
NOTE: Scratchpad blocks are used only for experimentation and testing out code.
The code written here will not be executed as part of the pipeline.
"""
from mage_ai.data_preparation.variable_manager import get_variable


df = get_variable('energy_consumption', 'create_splits', 'output_0')

print(f'Area: {len(df["area"].drop_duplicates())}')
print(f'consumer_type: {len(df["consumer_type"].drop_duplicates())}')
