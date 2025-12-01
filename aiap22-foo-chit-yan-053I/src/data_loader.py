import sqlite3
import pandas as pd
from typing import Tuple


def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y