import logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import skops.io as sio

from bioprocess_mlops.config.config import PreprocessingConfig
from bioprocess_mlops.utils import SavitzkyGolayFilter, SNV

logger = logging.getLogger(__name__)


class DataTransformation:
    def __init__(self,
                 preprocessing_config: PreprocessingConfig):
        self.preprocessing_config = preprocessing_config

    def get_transformer_object(self) -> Pipeline:
        try:
            preprocessing_steps = []
            steps_config = self.preprocessing_config.steps

            if steps_config.get('sg_smooth', {}).get('enabled', False):
                sg_params = steps_config['sg_smooth']['params']
                preprocessing_steps.append(
                    ('smoothing', SavitzkyGolayFilter(
                        window_length=sg_params['window_length'],
                        polyorder=sg_params['polyorder'],
                        deriv=sg_params['deriv']
                    ))
                    )
                logger.info("Added SG-smoothing to the pipeline")

            if steps_config.get('snv', {}).get('enabled', False):
                preprocessing_steps.append(('snv', SNV()))
                logger.info("Added SNV to pipeline")

            if steps_config.get('standard_scaler', {}).get('enabled', False):
                preprocessing_steps.append(('scaler', StandardScaler()))
                logger.info("Added StandardScaler to pipeline")

            if not preprocessing_steps:
                logger.warning("No preprocessing steps enabled - "
                               "returning empty pipeline")

            preprocessor = Pipeline(preprocessing_steps)

            return preprocessor

        except Exception:
            logger.exception("Error in getting the preprocessor object!")
            raise

    def create_preprocessor_object(self):
        try:
            preprocessing_obj = self.get_transformer_object()
            logger.info(f"Created preprocessing pipeline with "
                        f"{len(preprocessing_obj.steps)} steps")

            # Save preprocessing object if there are steps
            preprocessor_path = self.preprocessing_config.preprocesser_path
            if preprocessing_obj.steps:
                logger.info(f"Saving preprocessing object at {preprocessor_path}")  # noqa E51
                sio.dump(preprocessing_obj, preprocessor_path)

        except Exception:
            logger.error("Error in creating preprocessor object")
            raise
