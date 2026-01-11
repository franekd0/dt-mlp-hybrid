from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DATA_PATH = Path(__file__).parents[2].resolve() / "data"


# ============================================================
# CORE HELPER
# ============================================================

def split_and_preprocess(
    df: pd.DataFrame,
    target_col: str,
    numerical_cols: Iterable[str],
    categorical_cols: Iterable[str],
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Unified split + preprocessing:
    - one shared split
    - scaler / encoder fit ONLY on train
    - no data leakage
    """

    # -------------------------
    # TARGET
    # -------------------------
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])

    # -------------------------
    # SPLIT
    # -------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val
    )

    # -------------------------
    # NUMERICAL
    # -------------------------
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_val_num   = scaler.transform(X_val[numerical_cols])
    X_test_num  = scaler.transform(X_test[numerical_cols])

    # -------------------------
    # CATEGORICAL
    # -------------------------
    if categorical_cols:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_val_cat   = encoder.transform(X_val[categorical_cols])
        X_test_cat  = encoder.transform(X_test[categorical_cols])
    else:
        X_train_cat = np.empty((len(X_train), 0))
        X_val_cat   = np.empty((len(X_val),   0))
        X_test_cat  = np.empty((len(X_test),  0))

    # -------------------------
    # CONCAT
    # -------------------------
    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_val_final   = np.hstack([X_val_num,   X_val_cat])
    X_test_final  = np.hstack([X_test_num,  X_test_cat])

    return (
        X_train_final,
        X_val_final,
        X_test_final,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# DATASET LOADERS
# ============================================================

def load_wine_dataset(test_size: float, val_size: float, random_state: int = 42):
    """
    Wine dataset (all numerical).
    """
    data = load_wine(as_frame=True)
    df = data.frame  # target already included

    return split_and_preprocess(
        df=df,
        target_col="target",
        numerical_cols=data.feature_names,
        categorical_cols=[],
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def load_cancer_dataset(test_size: float, val_size: float, random_state: int = 42):
    """
    Breast Cancer Wisconsin dataset (all numerical).
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame  # target already included

    return split_and_preprocess(
        df=df,
        target_col="target",
        numerical_cols=data.feature_names,
        categorical_cols=[],
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def load_adult_dataset(test_size: float, val_size: float, random_state: int = 42):
    """
    Adult Income dataset (mixed).
    """
    df = pd.read_csv(DATA_PATH / "adult.csv", skipinitialspace=True)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    df["income"] = (df["income"] == ">50K").astype(int)

    numerical_cols = [
        "age",
        "capital-gain", "capital-loss", "hours-per-week"
    ]

    categorical_cols = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race", "gender",
        "native-country"
    ]

    return split_and_preprocess(
        df=df,
        target_col="income",
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def load_heart_statlog_dataset(test_size: float, val_size: float, random_state: int = 42):
    """
    Statlog Heart Disease dataset (mixed).
    """
    df = pd.read_csv(DATA_PATH / "heart.dat", header=None, sep=r"\s+")
    df.columns = [
        "age", "sex", "chest_pain", "rest_bp", "serum_chol",
        "fasting_blood_sugar", "electrocardiographic",
        "max_heart_rate", "angina", "oldpeak",
        "slope", "major_vessels", "thal", "target"
    ]

    df["target"] = (df["target"] == 1).astype(int)

    numerical_cols = [
        "age", "rest_bp", "serum_chol",
        "max_heart_rate", "oldpeak", "major_vessels"
    ]

    categorical_cols = [
        "sex", "chest_pain", "fasting_blood_sugar",
        "electrocardiographic", "angina", "slope", "thal"
    ]

    return split_and_preprocess(
        df=df,
        target_col="target",
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
