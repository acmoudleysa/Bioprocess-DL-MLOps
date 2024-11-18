import logging
import pandas as pd
import skops.io as sio
from bioprocess_mlops.config.config import (ModelConfig,
                                            DataConfig,
                                            PreprocessingConfig,
                                            MLflowConfig)
import numpy as np
from numpy.typing import NDArray
import mlflow
from bioprocess_mlops.utils import Metrics
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluation:
    def __init__(self,
                 model_config: ModelConfig,
                 data_config: DataConfig,
                 preprocessing_config: PreprocessingConfig,
                 mlflow_config: MLflowConfig):
        self.model_config = model_config
        self.data_config = data_config
        self.preprocessing_config = preprocessing_config
        self.mlflow_config = mlflow_config

    def preprocess_test_data(self) -> NDArray:
        try:
            logger.info("Reading test data")
            test_df = pd.read_csv(
                self.data_config.test_data_path
            )

            target_column = test_df.columns[-1]
            input_feature_test_df = test_df.drop(columns=[target_column])

            target_feature_test_df = test_df[target_column]

            logger.info("Preprocessing test data")
            pp_obj = sio.load(
                self.preprocessing_config.artifacts_path[
                    'fitted_preprocessor_path']
            )
            input_feature_test_arr = pp_obj.transform(input_feature_test_df)  # noqa E51
            test_arr = np.c_[input_feature_test_arr,
                             target_feature_test_df.to_numpy()]
            return test_arr

        except Exception:
            logger.exception("Error during preprocessing "
                             "the test data")
            raise

    def initiate_model_evaluation(self):
        try:
            logger.info("Model Evaluation started")
            test_arr = self.preprocess_test_data()
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            model = sio.load(
                self.model_config.trained_model_path
            )
            logger.info("Loaded Pretrained-model.")
            predictions = model.predict(X_test)
            metrics = Metrics(
                        metric_type=self.model_config.type,
                        y_true=y_test,
                        y_pred=predictions
                    ).results

            logger.info("Prediction completed!")
            if self.mlflow_config.active_status:
                logger.info("Logging to MLflow")
                mlflow.set_experiment(
                    self.mlflow_config.experiment_name
                )
                run_time = datetime.now().strftime("%Y%m%d-%H-%M-%S")
                with mlflow.start_run(run_name=f"{self.mlflow_config.run_name}-"   # noqa E51
                                      f"{run_time}"):
                    for key, val in metrics.items():
                        mlflow.log_metric(f"Test_{key}", val)

                    tracking_uri = self.mlflow_config.uri
                    input_example = np.array(X_test[0]).reshape(1, -1)

                    if (tracking_uri is not None) and tracking_uri != "file":
                        mlflow.set_tracking_uri(
                            tracking_uri
                        )
                        mlflow.sklearn.log_model(model, "model",
                                                 input_example=input_example)
                    else:
                        mlflow.sklearn.log_model(
                            model,
                            "model",
                            input_example=input_example
                        )
            else:
                logger.info(f"Metrics: {metrics}")

        except Exception:
            logger.error("Error during model evaluation")
            raise
