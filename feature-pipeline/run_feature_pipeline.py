import pandas as pd

import cleaning
import extract
import load
import utils
import validation


def main() -> dict:
    data, metadata = extract.from_api()

    data = transform(data)

    load.to_feature_store(
        data,
        validation_expectation_suite=validation.expectation_suite_energy_consumption
    )

    utils.save_json(metadata)

    return metadata


def transform(data: pd.DataFrame):
    # Clean columns
    data = cleaning.rename_columns(data)

    # Cast columns
    data = cleaning.cast_columns(data)

    # Standardize categorical data
    data = cleaning.standardize_categorical_data(data)

    return data


if __name__ == "__main__":
    main()
