import logging
import skops.io as sio
import pandas as pd
from sklearn.linear_model import LinearRegression
from bioprocess_mlops.config.config import (ModelConfig,
                                            DataConfig,
                                            PreprocessingConfig)
import numpy as np

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self,
                 model_config: ModelConfig,
                 data_config: DataConfig,
                 preprocessing_config: PreprocessingConfig):
        self.model_config = model_config
        self.data_config = data_config
        self.preprocessing_config = preprocessing_config

    def initiate_preprocessing(self) -> tuple:
        try:
            train_df = pd.read_csv(self.data_config.train_data_path)
            test_df = pd.read_csv(self.data_config.test_data_path)
            logger.info("Loaded train and test data.")

            # Modify as necessary
            target_column = train_df.columns[-1]
            input_feature_train_df = train_df.drop(columns=[target_column])
            input_feature_test_df = test_df.drop(columns=[target_column])

            target_feature_train_df = train_df[target_column]
            target_feature_test_df = test_df[target_column]

            preprocessing_obj = sio.load(
                self.preprocessing_config.preprocesser_path
            )

            # Apply transformations if pipeline has steps
            if preprocessing_obj.steps:
                logger.info("Applying preprocessing transformations...")
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)  # noqa E51
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  # noqa E51
            else:
                logger.info("No preprocessing steps enabled - using raw data")
                input_feature_train_arr = input_feature_train_df.values
                input_feature_test_arr = input_feature_test_df.values

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  # noqa E51
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # noqa E51
            logger.info("Preprocessing done!")
            return train_arr, test_arr

        except Exception:
            logger.error("Error during preprocessing the data.")
            raise

    def initiate_model_training(self):
        try:
            logger.info("Model training started!")
            train_arr, test_arr = self.initiate_preprocessing()
            
            model = LinearRegression()
            logger.info(f"Model training using {model.__class__.__name__}")
            model.fit(train_arr[:, :-1], train_arr[:, -1])

            logger.info("Model training completed!")

            sio.dump(model, self.model_config.trained_model_path)

            logger.info("Model successfully saved!")
        except Exception:
            logger.error("Error during model training")
            raise
