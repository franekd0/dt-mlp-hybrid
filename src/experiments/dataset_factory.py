from src.utils import (
    load_wine_dataset,
    load_cancer_dataset,
    load_adult_dataset,
    load_heart_statlog_dataset,

)
from src.experiments.experiment_config import ExperimentConfig


def load_dataset(cfg: ExperimentConfig):
    if cfg.dataset_name == "wine":
        return load_wine_dataset(
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

    if cfg.dataset_name == "adult":
        return load_adult_dataset(
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state
        )

    if cfg.dataset_name == "heart":
        return load_heart_statlog_dataset(
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state
        )

    raise ValueError(f"Unknown dataset: {cfg.dataset_name}")