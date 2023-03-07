
import data_loader as dl 
from data_validation import expectation_suite_energy_consumption 
from data_exporter import energy_feature_group, feature_descriptions
import preprocessing as pp

def exectue_pipeline():
    # Ingest data
    data = dl.load_api_data()

    # Clean columns
    data = pp.rename_columns(data)

    # Cast columns 
    data = pp.cast_columns(data)

    # Standardize categorical data
    data = pp.standardize_categorical_data(data)

    # Export to feature store
    energy_feature_group.insert(data)

    for description in feature_descriptions:
        energy_feature_group.update_feature_description(
        description["name"], 
        description["description"]
    )

    # Perform data validation
    energy_feature_group.save_expectation_suite(
        expectation_suite=expectation_suite_energy_consumption,
        validation_ingestion_policy="STRICT"
        )

if __name__ == "__main__": 
    exectue_pipeline()
