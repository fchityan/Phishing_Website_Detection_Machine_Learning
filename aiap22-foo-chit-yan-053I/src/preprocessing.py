from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_feature_types(X: pd.DataFrame):
    cols = [c for c in X.columns if c != "Unnamed: 0"]
    numeric_cols = X[cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X[cols].select_dtypes(exclude=np.number).columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocessor(
        num_col: List[str], cat_col: List[str]
) -> ColumnTransformer:
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                   ("scaler", StandardScaler()),])
    
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),])
    
    preproceesor = ColumnTransformer(transformers=[("num", num_pipeline, num_col), ("cat", cat_pipeline, cat_col)])

    return preproceesor