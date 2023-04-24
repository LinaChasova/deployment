from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    """
    Dataclass for feature and target parameters configuration.
    """
    numerical_features: List[str]
    categorical_features: List[str]
    target_col: Optional[str]
