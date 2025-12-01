from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_classification(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            # If model outputs probabilities for 2 classes (n,2)
            if y_proba.ndim == 2:
                y_scores = y_proba[:, 1]
            else:
                y_scores = y_proba

            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics["roc_auc"] = np.nan

    return metrics


def pretty_print_results(model_name: str, metrics: Dict[str, float]) -> None:
    print(f"\n===== Results for {model_name} =====")
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.4f}")