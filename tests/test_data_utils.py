import numpy as np
import pytest

from src.utils.data.data_utils import (
    _split_and_scale,
    load_moons_dataset,
    load_wine_dataset
)


def test_split_and_scale_shapes():
    X = np.random.randn(1000, 5)
    y = np.random.randint(0, 2, size=1000)

    X_train, X_val, X_test, y_train, y_val, y_test = _split_and_scale(
        X=X,
        y=y,
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )

    assert X_train.shape[0] == 600
    assert X_val.shape[0] == 200
    assert X_test.shape[0] == 200

    assert y_train.shape[0] == 600
    assert y_val.shape[0] == 200
    assert y_test.shape[0] == 200


def test_split_and_scale_no_nan():
    X = np.random.randn(500, 3)
    y = np.random.randint(0, 2, size=500)

    X_train, X_val, X_test, *_ = _split_and_scale(
        X=X,
        y=y,
        test_size=0.2,
        val_size=0.25,
        random_state=0
    )

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()


def test_split_and_scale_standardization():
    X = np.random.randn(1000, 4)
    y = np.random.randint(0, 2, size=1000)

    X_train, _, _, _, _, _ = _split_and_scale(
        X=X,
        y=y,
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )

    # Mean should be ~0 after StandardScaler
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-7)


def test_load_moons_dataset_shapes():
    X_train, X_val, X_test, y_train, y_val, y_test = load_moons_dataset(
        noise=0.2,
        n_samples=2000,
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )

    assert X_train.shape == (1200, 2)
    assert X_val.shape == (400, 2)
    assert X_test.shape == (400, 2)

    assert y_train.shape == (1200,)
    assert y_val.shape == (400,)
    assert y_test.shape == (400,)


def test_load_wine_dataset_shapes():
    X_train, X_val, X_test, y_train, y_val, y_test = load_wine_dataset(
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )

    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]

    assert X_train.shape[1] == 13
    assert total == 178

    assert y_train.shape[0] == X_train.shape[0]
    assert y_val.shape[0] == X_val.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
