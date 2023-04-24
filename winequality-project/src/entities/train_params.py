from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    """
    Dataclass for model parameters configuration.
    """
    n_estimators: int
    random_state: int = field(default=1)
