from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    """
    Dataclass for data splitting configuration.
    """
    random_state: int = field(default=42)
    val_size: float = field(default=0.2)
