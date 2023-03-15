import logging
from pathlib import Path
from typing import Union, Optional

import joblib
import pandas as pd

import settings
import wandb


def save_model(model, model_path: Union[str, Path]):
    """
    Template for saving a model.

    Args:
        model: Trained model.
        model_path: Path to save the model.
    """

    joblib.dump(model, model_path)


def load_model(model_path: Union[str, Path]):
    """
    Template for loading a model.

    Args:
        model_path: Path to the model.

    Returns: Loaded model.
    """

    return joblib.load(model_path)


def load_data_from_parquet(data_path: str) -> pd.DataFrame:
    """
    Template for loading data from a parquet file.

    Args:
        data_path: Path to the parquet file.

    Returns: Dataframe with the data.
    """

    return pd.read_parquet(data_path)


def get_logger(name: str) -> logging.Logger:
    """
    Template for getting a logger.

    Args:
        name: Name of the logger.

    Returns: Logger.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)

    return logger


def init_wandb_run(
    name: str,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    add_timestamp_to_name: bool = False,
    run_id: Optional[str] = None,
    resume: Optional[str] = None,
    reinit: bool = False,
    project: str = settings.CREDENTIALS["WANDB_PROJECT"],
    entity: str = settings.CREDENTIALS["WANDB_ENTITY"],
):
    if add_timestamp_to_name:
        name = f"{name}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        id=run_id,
        reinit=reinit,
        resume=resume
    )

    return run
