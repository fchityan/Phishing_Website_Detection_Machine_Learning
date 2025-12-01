from typing import Dict
import argparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_loader import load_data, split_features_target
from .preprocessing import get_feature_types, build_preprocessor
from .models import get_models
from .evaluation import evaluate_classification, pretty_print_results



def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end ML pipeline for phishing detection."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/phishing.db",
        help="Path to SQLite database file.",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="phishing_data",
        help="Table name in SQLite database.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()

def run_pipeline(
    db_path: str,
    table_name: str,
    test_size: float,
    random_state: int,
) -> Dict[str, Dict[str, float]]:
    print(f"Loading data from {db_path}, table '{table_name}'...")
    df = load_data(db_path, table_name)
    print(f"Loaded dataset with shape: {df.shape}")

    X, y = split_features_target(df, target_col="label")

    numeric_cols, categorical_cols = get_feature_types(X)
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = get_models(random_state=random_state)

    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_proba = None
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)
        elif hasattr(clf, "decision_function"):
            y_proba = clf.decision_function(X_test)

        metrics = evaluate_classification(y_test, y_pred, y_proba)
        pretty_print_results(name, metrics)

        results[name] = metrics

    return results


def main():
    args = parse_args()
    run_pipeline(
        db_path=args.db_path,
        table_name=args.table_name,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()