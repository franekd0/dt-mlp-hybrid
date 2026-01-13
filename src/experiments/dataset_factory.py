from src.utils.data import (
    load_wine_dataset,
    load_cancer_dataset,
)
from src.experiments.experiment_config import ComparisonExperimentConfig


def load_dataset(cfg: ComparisonExperimentConfig):
    if cfg.dataset_name == "wine":
        return load_wine_dataset(
            test_size=cfg.test_size,
            random_state=cfg.random_state
        )


    if cfg.dataset_name == "cancer":
        return load_cancer_dataset(
            test_size=cfg.test_size,
            random_state=cfg.random_state
        )


    raise ValueError(f"Unknown dataset: {cfg.dataset_name}")