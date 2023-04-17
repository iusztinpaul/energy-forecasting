import logging
import json
import joblib
import pandas as pd
import wandb

from pathlib import Path
from typing import Union, Optional


from training_pipeline import settings


def save_json(data: dict, file_name: str, save_dir: str = settings.OUTPUT_DIR):
    """
    Save a dictionary as a JSON file.

    Args:
        data: data to save.
        file_name: Name of the JSON file.
        save_dir: Directory to save the JSON file.

    Returns: None
    """

    data_path = Path(save_dir) / file_name
    with open(data_path, "w") as f:
        json.dump(data, f)


def load_json(file_name: str, save_dir: str = settings.OUTPUT_DIR) -> dict:
    """
    Load a JSON file.

    Args:
        file_name: Name of the JSON file.
        save_dir: Directory of the JSON file.

    Returns: Dictionary with the data.
    """

    data_path = Path(save_dir) / file_name
    with open(data_path, "r") as f:
        return json.load(f)


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
    project: str = settings.SETTINGS["WANDB_PROJECT"],
    entity: str = settings.SETTINGS["WANDB_ENTITY"],
):
    """Wrapper over the wandb.init function."""

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
        resume=resume,
    )

    return run


def check_if_artifact_exists(
    artifact_name: str,
    project: str = settings.SETTINGS["WANDB_PROJECT"],
    entity: str = settings.SETTINGS["WANDB_ENTITY"],
) -> bool:
    """Utiliy function that checks if a W&B artifact exists."""

    try:
        get_artifact(artifact_name, project, entity)

        return True
    except wandb.errors.CommError:
        return False


def get_artifact(
    artifact_name: str,
    project: str = settings.SETTINGS["WANDB_PROJECT"],
    entity: str = settings.SETTINGS["WANDB_ENTITY"],
) -> wandb.Artifact:
    """Get the latest version of a W&B artifact."""

    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}:latest")

    return artifact
