from src.utils import (
    load_moons_dataset,
    load_circles_dataset,
    load_wine_dataset,
    load_digits_dataset,
    load_cancer_dataset
)
from src.experiments.experiment_config import ExperimentConfig


def load_dataset(cfg: ExperimentConfig):
    if cfg.dataset_name == "moons":
        return load_moons_dataset(
            noise=cfg.noise or 0.0,
            n_samples=cfg.n_samples or 1000,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state
        )

    if cfg.dataset_name == "circles":
        return load_circles_dataset(
            noise=cfg.noise or 0.0,
            factor=cfg.factor or 0.5,
            n_samples=cfg.n_samples or 1000,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state
        )

    if cfg.dataset_name == "wine":
        return load_wine_dataset(
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state
        )

    if cfg.dataset_name == "digits":
        return load_digits_dataset(
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state
        )

    if cfg.dataset_name == "cancer":
        return load_cancer_dataset(
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state
        )

    raise ValueError(f"Unknown dataset: {cfg.dataset_name}")