import logging
from sklearn.preprocessing import StandardScaler, FunctionTransformer
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
                preprocessing_steps.append(('passthrough',
                                            FunctionTransformer(func=None,
                                                                validate=False)))  # noqa E51
            preprocessor = Pipeline(preprocessing_steps)
            logging.debug(preprocessor.steps)
            return preprocessor

        except Exception:
            logger.exception("Error in getting the preprocessor object!")
            raise

    def create_preprocessor_object(self):
        try:
            pp_template = self.get_transformer_object()
            logger.info(f"Created preprocessing pipeline with "
                        f"{len(pp_template.steps)} steps")

            # Save preprocessing object if there are steps
            preprocessor_template_path = self.preprocessing_config.pp_template_path  # noqa E51
            if pp_template.steps:
                logger.info(f"Saving preprocessing object at {preprocessor_template_path}")  # noqa E51
                sio.dump(pp_template, preprocessor_template_path)

        except Exception:
            logger.error("Error in creating preprocessor pipeline template")
            raise
