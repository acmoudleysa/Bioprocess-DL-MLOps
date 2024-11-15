import logging
from bioprocess_mlops.components import (DataIngestion,
                                         DataTransformation,
                                         ModelTrainer,
                                         ModelEvaluation)
from bioprocess_mlops.config import ConfigurationManager

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def start_data_ingestion(self):
        data_ingestion = DataIngestion(
            self.config.get_data_config,
            self.config.get_split_config
        )
        data_ingestion.initiate_data_ingestion()

    def start_data_transformation(self):
        data_transformation = DataTransformation(
            self.config.get_preprocessing_config
        )
        data_transformation.create_preprocessor_object()

    def start_model_training(self):
        model_train = ModelTrainer(
            self.config.get_model_config,
            self.config.get_data_config,
            self.config.get_preprocessing_config
        )
        model_train.initiate_model_training()

    def start_model_evaluation(self):
        model_evaluate = ModelEvaluation(
            self.config.get_model_config,
            self.config.get_data_config,
            self.config.get_preprocessing_config,
            self.config.get_mlflow_config
        )
        model_evaluate.initiate_model_evaluation()
