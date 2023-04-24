import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import read_data, process_target
from src.entities.predict_pipeline_params import (
    PredictPipelineParams,
    PredictPipelineParamsSchema,
)
from src.models import (
    predict_model,
    predict_proba_model,
    deserialize_model,
)

logger = logging.getLogger(__name__)


def predict_pipeline(prediction_pipeline_params: PredictPipelineParams):
    """
    The pipeline to load trained model and make predictions on given data.
    :param prediction_pipeline_params: prediction parameters
    :return: nothing
    """
    logger.info(f'Start prediction pipeline with params {prediction_pipeline_params}.')

    logger.info('Reading data...')
    features = read_data(prediction_pipeline_params.input_data_path)
    logger.info(f"data.shape is {features.shape}.")

    logger.info('Loading model...')
    target_transformer, pipeline = deserialize_model(prediction_pipeline_params.model_path)

    logger.info('Processing the target...')
    features, target, target_transformer = process_target(features, prediction_pipeline_params.feature_params, target_transformer)
    logger.info(f'features.shape is {features.shape}.')

    logger.info('Making predictions and calculating scores on the provided data...')
    predicts = predict_model(pipeline, features)
    predict_probes = predict_proba_model(pipeline, features)

    logger.info('Saving predictions and predicted scores...')
    features[prediction_pipeline_params.feature_params.target_col] = target
    features['predictions'] = predicts
    features['probabilities'] = predict_probes
    features.to_csv(prediction_pipeline_params.output_data_path, sep=';')
    logger.info(f'Output data saved to {prediction_pipeline_params.output_data_path}.')

    logger.info('Done.')


@hydra.main(config_path='../configs', config_name='predict_config.yaml')
def predict_pipeline_command(config: DictConfig):
    """
    Loads prediction parameters from config file and starts the prediction process.
    :param config: training configuration
    :return: nothing
    """
    os.chdir(hydra.utils.to_absolute_path('.'))
    schema = PredictPipelineParamsSchema()
    params = schema.load(config)
    logger.info(f'Prediction config:\n{OmegaConf.to_yaml(params)}')
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()
