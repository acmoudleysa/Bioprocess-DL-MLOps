import logging
from typing import Any, Dict
import yaml  # type: ignore
from dataclasses import dataclass, field
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    roc_auc_score,
    r2_score,
    recall_score
)
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    """
    Logging colored formatter, adapted from 
    https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
    """  # noqa E51

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def load_yaml(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath) as file:
            data = yaml.safe_load(file)
            logger.info(f"{filepath.name} loaded.")
            return data
    except Exception:
        logger.error(f"Error loading {filepath}")
        raise


class SavitzkyGolayFilter:
    ...


class SNV(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return X

@dataclass
class Metrics:
    metric_type: Literal["classification", "regression"]
    y_true: NDArray
    y_pred: NDArray
    results: dict = field(init=False)

    def __post_init__(self):
        logger.info(f"Calculating {self.metric_type} metrics.")
        self.results = (self.classification()
                        if self.metric_type == "classification"
                        else self.regression())

    def classification(self):
        return {
            'accuracy': accuracy_score(self.y_true,
                                       self.y_pred),
            'recall': recall_score(self.y_true,
                                   self.y_pred,
                                   average='binary'),
            'precision': precision_score(self.y_true,
                                         self.y_pred,
                                         average='binary'),
            'roc_auc': roc_auc_score(self.y_true,
                                     self.y_pred)
        }

    def regression(self):
        return {
            'rmse': root_mean_squared_error(self.y_true,
                                            self.y_pred),
            'mae': mean_absolute_error(self.y_true,
                                       self.y_pred),
            'r2': r2_score(self.y_true,
                           self.y_pred)
        }


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
