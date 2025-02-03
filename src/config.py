import yaml
from pathlib import Path

# Automatically find the config file in the correct location
CONFIG_PATH = Path(__file__).parent.parent / "configs/config.yml"
CNN_HYPERPARAMETERS_PATH = Path(__file__).parent.parent / "configs/cnn_sweep_config.yml"


if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}. Please create it from `config.example.yml`."
    )
if not CNN_HYPERPARAMETERS_PATH.exists():
    raise FileNotFoundError(f"Hyperparameter configuration file not found: {CNN_HYPERPARAMETERS_PATH}. Please create it from `config.example.yml`.")


# Load the general configuration
with open(CONFIG_PATH, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# Load the CNN hyperparameter configuration
with open(CNN_HYPERPARAMETERS_PATH, "r") as ymlfile:
    cnn_sweep_cfg = yaml.safe_load(ymlfile)