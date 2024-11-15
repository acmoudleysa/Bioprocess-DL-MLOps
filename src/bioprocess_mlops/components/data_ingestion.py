import pandas as pd
# import polars as pl
import logging
from bioprocess_mlops.config.config import (DataConfig,
                                            TrainTestSplitConfig)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self,
                 data_config: DataConfig,
                 split_config: TrainTestSplitConfig):
        self.data_config = data_config
        self.split_config = split_config

    def initiate_data_ingestion(self):
        logger.info(f"Reading {self.data_config.raw_data_path}")
        try:
            data = pd.read_csv(self.data_config.raw_data_path)
            logger.info("Reading complete")

            logger.info("Performing train-test split")
            train_data, test_data = train_test_split(
                data,
                test_size=self.split_config.test_size,
                random_state=self.split_config.random_seed
            )
            logger.info("Train-Test split complete.")
            train_data.to_csv(self.data_config.train_data_path,
                              index=False)
            test_data.to_csv(self.data_config.test_data_path,
                             index=False)

            logger.info("Data Ingestion completed.")
        except Exception:
            logger.exception("Error during Data Ingestion")
            raise
