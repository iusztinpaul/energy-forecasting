import joblib
import pandas as pd


def save_model(model, model_path: str):
    """
    Template for saving a model.

    Args:
        model: Trained model.
        model_path: Path to save the model.
    """

    joblib.dump(model, model_path)


def load_data_from_parquet(data_path: str) -> pd.DataFrame:
    """
    Template for loading data from a parquet file.

    Args:
        data_path: Path to the parquet file.

    Returns: Dataframe with the data.
    """

    return pd.read_parquet(data_path)
