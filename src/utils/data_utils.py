from sklearn.datasets import make_moons, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def _split_and_scale(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float,
        val_size: float,
        random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train / validation / test sets and apply StandardScaler
    fitted only on training data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.

    y : np.ndarray
        Target labels.

    test_size : float
        Fraction of the full dataset used as test set.

    val_size : float
        Fraction of the remaining data (after test split)
        used as validation set.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray
    X_val   : np.ndarray
    X_test  : np.ndarray
    y_train : np.ndarray
    y_val   : np.ndarray
    y_test  : np.ndarray
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_moons_dataset(
        noise: float,
        test_size: float,
        val_size: float,
        n_samples: int,
        random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load make_moons dataset with explicit train / validation / test split
    and StandardScaler fitted only on training data.
    """

    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )

    return _split_and_scale(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def load_wine_dataset(
        test_size: float,
        val_size: float,
        random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the wine dataset with explicit train / validation / test split
    and StandardScaler fitted only on training data.
    """

    data = load_wine()
    X = data.data
    y = data.target

    return _split_and_scale(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
