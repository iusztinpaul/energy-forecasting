import logging
import joblib
import lightgbm as lgb
import pandas as pd

import preprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(data_path: str, model_path: str) -> None:
    """
    Template for training a model.

    Args:
        data_path: Path to the training data.
        model_path: Path to save the trained model.
    """
    # load data
    df = load_data_from_parquet(data_path)

    train_df, test_df = preprocess.split_data(df)
    train_df, test_df = preprocess.encode_categorical(train_df, test_df)

    # train model
    model = train_lgbm_model(train_df, target="energy_consumption_future_hours_0")

    # evaluate model
    rmse = evaluate_model(model, train_df, target="energy_consumption_future_hours_0")
    print(f"Train RMSE: {rmse}")
    print(f"Train Mean Energy Consumption: {train_df['energy_consumption_future_hours_0'].mean()}")

    # evaluate model
    rmse = evaluate_model(model, test_df, target="energy_consumption_future_hours_0")
    print(f"Test RMSE: {rmse}")
    print(f"Test Mean Energy Consumption: {test_df['energy_consumption_future_hours_0'].mean()}")

    # save model
    save_model(model, model_path)


def load_data_from_parquet(data_path: str) -> pd.DataFrame:
    """
    Template for loading data from a parquet file.

    Args:
        data_path: Path to the parquet file.

    Returns: Dataframe with the data.
    """

    return pd.read_parquet(data_path)


# write a function that trains a lightgbm model on a dataframe
def train_lgbm_model(df: pd.DataFrame, target: str) -> lgb.LGBMRegressor:
    """
   Function that is training a LGBM model.

    Args:
        df: Dataframe containing the training data.
        target: Name of the target column.

    Returns: Trained LightGBM model
    """

    model = lgb.LGBMRegressor(
        n_estimators=500
    )

    feature_columns = list(set(df.columns) - set([target, "datetime_utc"]))
    model.fit(
        X=df[feature_columns],
        y=df[target]
    )

    return model


# write a function that evaluates the model with rmse
def evaluate_model(model, df: pd.DataFrame, target: str):
    """
    Template for evaluating a model.

    Args:
        model: Trained model.
        df: Dataframe containing the evaluation data.
        target: Name of the target column.

    Returns: RMSE
    """
    from sklearn.metrics import mean_squared_error

    feature_columns = list(set(df.columns) - set([target, "datetime_utc"]))
    y_pred = model.predict(df[feature_columns])
    y_true = df[target]

    return mean_squared_error(y_true, y_pred, squared=False)


# write function that saves the model using joblib
def save_model(model, model_path: str):
    """
    Template for saving a model.

    Args:
        model: Trained model.
        model_path: Path to save the model.
    """

    joblib.dump(model, model_path)


if __name__ == "__main__":
    train(
        "/home/iusztin/Documents/projects/energy-forecasting/energy_consumption_data.parquet",
        "/home/iusztin/Documents/projects/energy-forecasting/energy_consumption_model.pkl"
    )