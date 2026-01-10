from sklearn.datasets import make_moons, load_wine, make_circles, load_digits, load_breast_cancer
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
    Load make_moons dataset.
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


def load_circles_dataset(
        noise: float,
        test_size: float,
        val_size: float,
        n_samples: int,
        random_state: int = 42,
        factor: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load make_circles dataset (Okręgi wpisane w siebie).
    To jest 'Killer' dla zwykłego drzewa decyzyjnego.
    """
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state
    )

    return _split_and_scale(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def load_digits_dataset(
        test_size: float,
        val_size: float,
        random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load digits dataset (8x8 images flattened to 64 features).
    10 classes (numbers 0-9).
    """
    data = load_digits()
    X = data.data
    y = data.target

    return _split_and_scale(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def load_cancer_dataset(
        test_size: float,
        val_size: float,
        random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Breast Cancer Wisconsin dataset (30 features).
    Binary classification.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target

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
    Load the wine dataset.
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