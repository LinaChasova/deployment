import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.entities.feature_params import FeatureParams
from typing import Tuple

import warnings
warnings.filterwarnings('ignore')


def build_numerical_pipeline() -> Pipeline:
    """
    Builds pipeline for numerical features processing.
    :return: pipeline class
    """
    num_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='constant')),
            ('scale', StandardScaler()),
        ]
    )
    return num_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds pipeline and processes numerical features.
    :param numerical_df: dataframe of numerical features
    :return: dataframe of the processed features
    """
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    """
    Builds transformer to process numerical features.
    :param params: configuration for feature processing
    :return: transformer class
    """
    transformer = ColumnTransformer(
        [
            (
                'numerical_pipeline',
                build_numerical_pipeline(),
                params.numerical_features,
            )
        ]
    )
    return transformer


def make_features(
        transformer: ColumnTransformer, features: pd.DataFrame, params: FeatureParams
) -> pd.DataFrame:
    """
    Processes features and removes unnecessary features.
    :param transformer: transformer to process features
    :param features: features dataframe
    :param params: configuration for feature processing
    :return: dataframe of the processed features
    """
    return pd.DataFrame(
        transformer.transform(features),
        columns=params.numerical_features
    )
