import yaml

from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams


@dataclass()
class PredictPipelineParams:
    """
    Dataclass for prediction pipeline configuration.
    """
    input_data_path: str
    output_data_path: str
    model_path: str
    metric_path: str
    feature_params: FeatureParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    """
    Reads configuration file for prediction pipeline.
    :param path: path to configuration file
    :return: configuration dataclass
    """
    with open(path, 'r') as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
