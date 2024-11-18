from pathlib import Path
from bioprocess_mlops.utils import SNV, SavitzkyGolayFilter

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SECRETS_FILE_PATH = Path("config/secrets.yaml")
TRUSTED_SKOPS_OBJ = [SNV, SavitzkyGolayFilter]