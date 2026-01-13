from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, make_moons
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def get_columns(dataset: str):
    match dataset:
        case "wine": return load_wine().feature_names
        case "cancer": return load_breast_cancer().feature_names
        case _: return None


def split_and_preprocess(
    df: pd.DataFrame,
    target_col: str,
    numerical_cols: Iterable[str],
    categorical_cols: Iterable[str],
    test_size: float,
    random_state: int,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray
]:
    """
    Unified split + preprocessing (TRAIN / TEST only):
    - single split
    - scaler / encoder fit ONLY on train
    """

    y = df[target_col].values
    X_df = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_test_num  = scaler.transform(X_test[numerical_cols])

    if categorical_cols:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat  = encoder.transform(X_test[categorical_cols])
    else:
        X_train_cat = np.empty((len(X_train), 0))
        X_test_cat  = np.empty((len(X_test),  0))

    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_test_final  = np.hstack([X_test_num,  X_test_cat])

    return (
        X_train_final,
        X_test_final,
        y_train,
        y_test
    )

def load_wine_dataset(test_size: float, random_state: int = 42):
    data = load_wine(as_frame=True)
    df = data.frame

    return split_and_preprocess(
        df=df,
        target_col="target",
        numerical_cols=data.feature_names,
        categorical_cols=[],
        test_size=test_size,
        random_state=random_state
    )

def load_cancer_dataset(test_size: float, random_state: int = 42):
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    return split_and_preprocess(
        df=df,
        target_col="target",
        numerical_cols=data.feature_names,
        categorical_cols=[],
        test_size=test_size,
        random_state=random_state
    )


def load_moons_dataset(test_size: float, random_state: int = 42):
    X, y = make_moons(n_samples=10000, noise=0.2, random_state=random_state)

    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["target"] = y

    return split_and_preprocess(
        df=df,
        target_col="target",
        numerical_cols=["x1", "x2"],
        categorical_cols=[],
        test_size=test_size,
        random_state=random_state
    )