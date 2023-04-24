from pydantic import BaseModel, BaseSettings, validator
from typing import List


SERVICE_NAME = 'Winequality.ML'
SERVICE_VERSION = '0.1.0'


def to_camel(string: str) -> str:
    """
    Converts lowercase string to camelCase format.
    :param string: lowercase string
    :return: camelCase format string
    """
    return ' '.join(string.split('_'))


class TransactionModel(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    @validator('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
        'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol')
    def validate_strings(cls, parameter):
        if not isinstance(parameter, float):
            raise TypeError('features should be a float')
        return parameter

    class Config:
        alias_generator = to_camel


class TransactionResponse(BaseModel):
    risk_rating: float
    is_risk_calculated: bool

class FeatureParams(BaseSettings):
    numerical_features: List[str] = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    target_col: str = 'quality'
