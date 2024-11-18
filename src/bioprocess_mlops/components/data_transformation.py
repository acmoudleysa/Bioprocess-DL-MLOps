import logging
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import skops.io as sio
from typing import List, Tuple, Any

from bioprocess_mlops.config.config import PreprocessingConfig
from bioprocess_mlops.utils import SavitzkyGolayFilter, SNV

logger = logging.getLogger(__name__)


class DataTransformation:
    def __init__(self,
                 preprocessing_config: PreprocessingConfig):
        self.preprocessing_config = preprocessing_config

    def _validate_order_config(self) -> bool:
        enabled_steps = {
            name for name, config in self.preprocessing_config.steps.items()
            if config['enabled']
        }
        order_steps = self.preprocessing_config.order

        if ((len(enabled_steps) != len(order_steps)) or (enabled_steps !=
                                                         set(order_steps))):
            raise ValueError(
                f"Mismatch in configuration. Enabled steps: {enabled_steps}, "
                f"Order specified: {self.preprocessing_config.order}"
                )

    def get_transformer_object(self) -> Pipeline:
        try:
            self._validate_order_config()
            preprocessing_steps: List[Tuple[str, Any]] = []
            steps_config = self.preprocessing_config.steps

            for step_name in self.preprocessing_config.order:
                config = steps_config[step_name]
                if config['enabled']:
                    if step_name == 'sg_smooth':
                        sg_params = config['params']
                        preprocessing_steps.append(
                            ('smoothing', SavitzkyGolayFilter(
                                window_length=sg_params['window_length'],
                                polyorder=sg_params['polyorder'],
                                deriv=sg_params['deriv']
                            ))
                        )
                        logger.info("Added SG-smoothing to the pipeline")

                    elif step_name == 'snv':
                        preprocessing_steps.append(('snv', SNV()))
                        logger.info("Added SNV to pipeline")

                    elif step_name == 'standard_scaler':
                        preprocessing_steps.append(
                            ('scaler', StandardScaler()))
                        logger.info("Added StandardScaler to pipeline")

                    else:
                        logger.warning(f"{step_name} has not be added yet")

            if not preprocessing_steps:
                logger.warning("No preprocessing steps enabled - returning "
                               "passthrough pipeline")
                preprocessing_steps.append(
                    ('passthrough', FunctionTransformer(func=None,
                                                        validate=False))
                )
            preprocessor = Pipeline(preprocessing_steps)
            logger.debug(f"Pipeline steps: "
                         f"{[step[0]for step in preprocessor.steps]}")
            return preprocessor

        except Exception:
            logger.exception("Error in getting the preprocessor object!")
            raise

    def create_preprocessor_object(self):
        try:
            pp_template = self.get_transformer_object()
            logger.info(f"Created preprocessing pipeline with "
                        f"{len(pp_template.steps)} steps")
            preprocessor_template_path = self.preprocessing_config.artifacts_path['preprocesser_template_path']  # noqa E51
            logger.info(f"Saving preprocessing object at {preprocessor_template_path}")  # noqa E51
            sio.dump(pp_template, preprocessor_template_path)

        except Exception:
            logger.error("Error in creating preprocessor pipeline template")
            raise


if __name__ == "__main__":
    from bioprocess_mlops.config import ConfigurationManager
    cfm = ConfigurationManager()
    dtf = DataTransformation(
        cfm.get_preprocessing_config)
    dtf.get_transformer_object()
