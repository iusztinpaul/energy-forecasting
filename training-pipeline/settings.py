import warnings
from pathlib import Path

import matplotlib

import utils

warnings.filterwarnings(action="ignore", category=FutureWarning, module="sktime")
matplotlib.use("Agg")

CREDENTIALS = utils.load_env_vars(root_dir="..")
# TODO: Change output dir with a tmp dir that is deleted at the end of the training script.
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
