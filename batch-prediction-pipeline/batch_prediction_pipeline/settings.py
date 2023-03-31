import os
import warnings
from pathlib import Path
from typing import Union

from dotenv import load_dotenv


warnings.filterwarnings(action="ignore", category=FutureWarning, module="sktime")


def load_env_vars(root_dir: Union[str, Path]) -> dict:
    """
    Load environment variables from .env.default and .env files.

    Args:
        root_dir: Root directory of the .env files.

    Returns:
        Dictionary with the environment variables.
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    load_dotenv(dotenv_path=root_dir / ".env.default")
    load_dotenv(dotenv_path=root_dir / ".env", override=True)

    return dict(os.environ)


# TODO: Find how to properly inject the root dit.
# ROOT_DIR = Path(".")
ROOT_DIR = Path("/opt/airflow/dags")
CREDENTIALS = load_env_vars(root_dir=ROOT_DIR)
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
