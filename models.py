from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator


def get_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    models = {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }

    return models