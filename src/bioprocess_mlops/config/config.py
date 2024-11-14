from dataclasses import dataclass
from bioprocess_mlops.utils import load_yaml
from bioprocess_mlops.constants import (CONFIG_FILE_PATH,
                                        PARAMS_FILE_PATH)
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str


@dataclass(frozen=True)
class ModelConfig:
    type: str
    trained_model_path: str
    preprocesser_path: str


class ConfigurationManager:
    def __init__(self,
                 config_path: str = CONFIG_FILE_PATH,
                 params_path: str = PARAMS_FILE_PATH):
        self.config = load_yaml(config_path)
        self.params = load_yaml(params_path)

    def get_data_config(self) -> DataConfig:
        data_config = self.config["data_paths"]
        return DataConfig(
            raw_data_path=data_config["raw_data"],
            train_data_path=data_config["train_data"],
            test_data_path=data_config["test_data"]
        )

    def get_model_config(self) -> ModelConfig:
        return ModelConfig(
            type=self.config["ML_MODEL"]["type"],
            trained_model_path=self.config["model_object"]
            ["trained_model_path"],
            preprocesser_path=self.config["preprocessing_object"]
            ["preprocesser_path"]
        )

    def get_split_config(self) -> Dict[str, Any]:
        """Get train-test split parameters"""
        return self.config["split"]

    def get_model_params(self):
        raise NotImplementedError("Not implemented yet. Come back later")


if __name__ == "__main__":
    configuration = ConfigurationManager()
    print(configuration.config)
