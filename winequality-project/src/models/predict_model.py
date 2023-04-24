import logging
import pickle
import sys
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def predict_model(
    model: Union[Pipeline, RandomForestClassifier], features: pd.DataFrame
) -> np.ndarray:
    """
    Makes predictions based on model.
    :param model: the model to predict with
    :param features: the features to predict on
    :return: model predictions
    """
    predictions = model.predict(features)
    return predictions


def predict_proba_model(
    model: Union[Pipeline, RandomForestClassifier], features: pd.DataFrame
) -> np.ndarray:
    """
    Returns probability estimates for the 'Fraud' class objects.
    :param model: the model to calculate scores with
    :param features: the features to be scored
    :return: probability scores
    """
    predict_pbs = model.predict_proba(features)[:, 1]
    return predict_pbs


def deserialize_model(path: str) -> Tuple[LabelEncoder, Pipeline]:
    """
    Loads model (pipeline) from pickle file.
    :param path: path to file to load from
    :return: deserialized Pipeline
    """
    try:
        with open(path, "rb") as file:
            (target_transformer, pipeline) = pickle.load(file)
        return target_transformer, pipeline
    except FileNotFoundError as error:
        logger.error(error)
        sys.exit(0)
