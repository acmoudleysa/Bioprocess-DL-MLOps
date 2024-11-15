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
            logger.info("Loaded train data.")

            target_column = train_df.columns[-1]
            input_feature_train_df = train_df.drop(columns=[target_column])

            target_feature_train_df = train_df[target_column]

            preprocessing_obj = sio.load(
                self.preprocessing_config.pp_template_path
            )

            logger.info("Fitting the preprocessing pipeline")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)  # noqa E51
            sio.dump(preprocessing_obj,
                     self.preprocessing_config.pp_fitted_path)

            logging.info("Fitted preprocessor is saved!")
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]  # noqa E51
            return train_arr

        except Exception:
            logger.error("Error during preprocessing the data.")
            raise

    def initiate_model_training(self):
        try:
            logger.info("Model training started!")
            train_arr = self.initiate_preprocessing()

            model = LinearRegression()
            logger.info(f"Model training using {model.__class__.__name__}")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            model.fit(X_train, y_train)

            logger.info("Model training completed!")

            sio.dump(model, self.model_config.trained_model_path)

            logger.info("Model successfully saved!")
        except Exception:
            logger.error("Error during model training")
            raise
