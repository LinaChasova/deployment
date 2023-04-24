# -*- coding: utf-8 -*-
import logging
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from entities.feature_params import FeatureParams
from sklearn.preprocessing import LabelEncoder

from src.entities import SplittingParams, FeatureParams

logger = logging.getLogger(__name__)


def read_data(path: str) -> pd.DataFrame:
    """
        Read data from input file
        :param path: path to the input file
        :return: dataframe of features plus target variables
    """
    features = pd.read_csv(path, index_col=0)
    return features


def process_target(data: pd.DataFrame, params: FeatureParams,
    transformer: Optional[LabelEncoder] = None) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
        Making binary classificaion for the response variable.
        Dividing wine as good and bad by giving the limit for the quality
        :param data: initial dataframe of features plus target
        :param params: feature params
    """
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    data[params.target_col] = pd.cut(data[params.target_col], bins=bins, labels=group_names)

    if transformer is None:
        transformer = LabelEncoder()
    data[params.target_col] = transformer.fit_transform(data[params.target_col])

    features = data[params.numerical_features]
    target = data[params.target_col]
    return features, target, transformer


def split_train_val_data(
        data: pd.DataFrame, target: pd.Series, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into training and validation datasets.
    :param data: initial dataframe
    :param params: splitting parameters
    :return: tuple of training and validation dataframes
    """
    train_features, val_features, train_target, val_target = train_test_split(
        data, target, test_size=params.val_size,
        random_state=params.random_state, stratify=target
    )
    return train_features, val_features, train_target, val_target
