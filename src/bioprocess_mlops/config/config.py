from dataclasses import dataclass
from bioprocess_mlops.utils import load_yaml
from bioprocess_mlops.constants import (CONFIG_FILE_PATH,
                                        PARAMS_FILE_PATH,
                                        SECRETS_FILE_PATH)
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str


@dataclass(frozen=True)
class TrainTestSplitConfig:
    test_size: float
    random_seed: int


@dataclass(frozen=True)
class ModelConfig:
    type: str
    trained_model_path: str    


@dataclass
class MLflowConfig:
    active_status: bool
    experiment_name: str
    run_name: str
    uri: str


@dataclass
class PreprocessingConfig:
    steps: Dict[str, Dict[str, Any]]
    pp_template_path: str
    pp_fitted_path: str


class ConfigurationManager:
    def __init__(self,
                 config_path: str = CONFIG_FILE_PATH,
                 params_path: str = PARAMS_FILE_PATH,
                 secrets_path: str = SECRETS_FILE_PATH):
        self.config = load_yaml(config_path)
        self.params = load_yaml(params_path)
        self.secrets_path = load_yaml(secrets_path)

    def get_data_config(self) -> DataConfig:
        data_config = self.config["data_paths"]
        return DataConfig(
            raw_data_path=data_config["raw_data"],
            train_data_path=data_config["train_data"],
            test_data_path=data_config["test_data"]
        )

    def get_mlflow_config(self) -> MLflowConfig:
        """Get MLflow related configurations"""
        mlflow_config = self.config["mlflow"]
        return MLflowConfig(
            experiment_name=mlflow_config["experiment_name"],
            run_name=mlflow_config["run_name"],
            active_status=mlflow_config["active"],
            uri=self.secrets_path['mlflow_uri']
        )

    def get_model_config(self) -> ModelConfig:
        model_config = self.config["model"]
        return ModelConfig(
            type=model_config["type"],
            trained_model_path=model_config['artifact']
            ["trained_model_path"]
        )

    def get_split_config(self) -> TrainTestSplitConfig:
        """Get train-test split parameters"""
        return TrainTestSplitConfig(
            test_size=self.config['split']['test_size'],
            random_seed=self.config['split']['random_state']
        )

    def get_preprocessing_config(self) -> PreprocessingConfig:
        pp_config = self.config['preprocessing']
        return PreprocessingConfig(
            steps=pp_config['methods'],
            pp_template_path=pp_config['artifact']
            ['preprocesser_template_path'],
            pp_fitted_path=pp_config['artifact']
            ['fitted_preprocessor_path']
        )

    def get_model_params(self):
        raise NotImplementedError("Not implemented yet. Come back later")


if __name__ == "__main__":
    configuration = ConfigurationManager()
    print(configuration.get_mlflow_config().active_status)
