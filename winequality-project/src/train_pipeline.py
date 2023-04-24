import json
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import read_data, split_train_val_data, process_target
from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    TrainingPipelineParamsSchema,
)
from src.features.build_features import (
    build_transformer,
    make_features,
)
from src.models import (
    train_model,
    serialize_model,
    predict_proba_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    """
    The pipeline to transform data, train and evaluate model and store the artifacts.
    :param training_pipeline_params: training parameters
    :return: nothing
    """
    logger.info(f'Start train pipeline with params {training_pipeline_params}.')

    logger.info('Reading train data...')
    features = read_data(training_pipeline_params.input_data_path)
    logger.info(f'features.shape is {features.shape}.')
    logger.info(features.head())

    logger.info('Processing the target...')
    features, target, target_transformer = process_target(features, training_pipeline_params.feature_params)
    logger.info(f'features.shape is {features.shape}.')

    logger.info('Splitting data...')
    train_features, val_features, train_target, val_target = split_train_val_data(
        features, target, training_pipeline_params.splitting_params
    )
    logger.info(f'train_df.shape is {train_features.shape}.')
    logger.info(f'val_df.shape is {val_features.shape}.')

    logger.info(f'Training target distribution:\n{train_target.value_counts()}')
    logger.info(f'Validation target distribution:\n{val_target.value_counts()}')

    logger.info('Building transformer...')
    transformer = build_transformer(training_pipeline_params.feature_params)
    logger.info('Fitting transformer...')
    transformer.fit(train_features)

    logger.info('Preparing train data...')
    train_features = make_features(
        transformer, train_features, training_pipeline_params.feature_params
    )

    logger.info('Training model...')
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    logger.info('Preparing validation data...')
    val_features = make_features(
        transformer, val_features, training_pipeline_params.feature_params
    )

    logger.info('Predicting scores on the validation data...')
    predict_pbs = predict_proba_model(model, val_features)

    logger.info('Evaluating model...')
    metrics = evaluate_model(predict_pbs, val_target)
    logger.info(f'Metrics are {metrics}.')

    logger.info('Saving metrics...')
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    logger.info('Saving training pipeline...')
    path_to_model = serialize_model(
        model,
        training_pipeline_params.output_model_path,
        target_transformer,
        transformer
    )

    logger.info('Done.')
    return path_to_model, metrics


@hydra.main(config_path='../configs', config_name='train_config.yaml')
def train_pipeline_command(config: DictConfig):
    """
    Loads training parameters from config file and starts training process.
    :param config: training configuration
    :return: nothing
    """
    os.chdir(hydra.utils.to_absolute_path('.'))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(config)
    logger.info(f'Training config:\n{OmegaConf.to_yaml(params)}')
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
