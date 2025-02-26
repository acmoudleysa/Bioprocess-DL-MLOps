from bioprocess_mlops.pipeline.training_pipeline import TrainingPipeline
from bioprocess_mlops.utils import setup_logging
import logging


def main():
    setup_logging()
    _ = logging.getLogger(__name__)
    training = TrainingPipeline()
    training.start_data_ingestion()
    training.start_data_transformation()
    training.start_model_training()
    training.start_model_evaluation()


if __name__ == "__main__":
    main()
