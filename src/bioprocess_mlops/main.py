import logging
import logging.config
from datetime import datetime
import os
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from bioprocess_mlops.config.config import ConfigurationManager
from bioprocess_mlops.components import (DataIngestion,
                                         DataTransformation,
                                         ModelTrainer)


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
    config_path = Path(__file__).resolve().parents[2] / CONFIG_DIR / config
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": f"{LOG_DIR}/{timestamp}.log"}
    )


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    configuration = ConfigurationManager()
    data_ingestion = DataIngestion(configuration.get_data_config(),
                                   configuration.get_split_config())
    data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation(
        configuration.get_preprocessing_config())
    data_transformation.create_preprocessor_object()

    model_train = ModelTrainer(configuration.get_model_config(),
                               configuration.get_data_config(),
                               configuration.get_preprocessing_config())
    model_train.initiate_model_training()

