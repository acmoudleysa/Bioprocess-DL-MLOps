import logging
import logging.config
from datetime import datetime
import os
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

env_file = find_dotenv()
load_dotenv()

CONFIG_DIR = "config"
LOG_DIR = "logs"


def setup_logging():
    log_configs = {
        "dev": "logging.dev.ini",
        "prod": "logging.prod.ini"
    }
    config = log_configs.get(os.environ["ENV"], "logging.dev.ini")
    config_path = Path(__file__).resolve().parents[3] / CONFIG_DIR / config
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": f"{LOG_DIR}/{timestamp}.log"}
    )


if __name__ == "__main__":
    from pathlib import Path
    print(Path(__file__).resolve().parents[3])