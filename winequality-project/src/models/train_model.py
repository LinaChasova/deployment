import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
    auc
)

from src.entities.train_params import TrainingParams


def train_model(
    train_features: pd.DataFrame, train_target: pd.Series, train_params: TrainingParams
) -> RandomForestClassifier:
    """
    Trains the model.
    :param train_features: features to train on
    :param train_target: target to train on
    :param train_params: training parameters
    :return: trained classifier model
    """
    model = RandomForestClassifier(
        random_state=train_params.random_state,
        n_estimators=train_params.n_estimators,
    )
    model.fit(train_features, train_target)
    return model


def evaluate_model(
    predicts_pbs: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    """
    Evaluates model predictions and returns the metrics.
    :param predicts_pbs: target scores predicted by model
    :param target: actual target labels
    :return: a dict of metrics in format {'metric_name': value}
    """
    pr_curve = precision_recall_curve(target, predicts_pbs)
    f1_scores = 2 * pr_curve[0] * pr_curve[1] / (pr_curve[0] + pr_curve[1])
    threshold = pr_curve[2][np.nanargmax(f1_scores)]
    predicts = (predicts_pbs >= threshold).astype('float')
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        target, predicts, average='binary'
    )

    return {
        "accuracy": accuracy_score(target, predicts),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc_score(target, predicts_pbs),
        "pr_auc": auc(pr_curve[1], pr_curve[0]),
        "confusion_matrix": confusion_matrix(target, predicts).tolist()
    }


def serialize_model(
    model: RandomForestClassifier, output: str, target_transformer: LabelEncoder, 
    transformer: Optional[ColumnTransformer] = None
) -> str:
    """
    Saves trained model (pipeline) to pickle file.
    :param transformer: data transformer to save
    :param model: trained model to save
    :param output: filename to save to
    :return: the path to pickle file
    """
    pipeline = Pipeline(
        [
            ('transformer', transformer),
            ('model', model),
        ]
    )
    with open(output, "wb") as file:
        pickle.dump((target_transformer, pipeline), file)
    return output
