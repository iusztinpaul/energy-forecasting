import cleaning
import extract
import load
import validation


def main():
    # Ingest data
    data = extract.from_api()

    # Clean columns
    data = cleaning.rename_columns(data)

    # Cast columns
    data = cleaning.cast_columns(data)

    # Standardize categorical data
    data = cleaning.standardize_categorical_data(data)

    # Export to feature store
    feature_group = load.to_feature_store(data)

    # Perform data validation
    feature_group.save_expectation_suite(
        expectation_suite=validation.expectation_suite_energy_consumption,
        validation_ingestion_policy="STRICT",
    )


if __name__ == "__main__":
    main()
