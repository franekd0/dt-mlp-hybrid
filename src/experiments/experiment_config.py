from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    # --- META ---
    name: str
    dataset_name: str
    random_state: int = 42

    # --- DATA ---
    test_size: float = 0.2
    val_size: float = 0.2
    n_samples: Optional[int] = None

    # --- DATASET-SPECIFIC ---
    noise: Optional[float] = None
    factor: Optional[float] = None

    # --- TREE ---
    tree_type: str = "decision_tree"
    tree_max_depth: int = 5

    # --- MLP ---
    embedding_dim: int = 2
    hidden_dim: int = 16
    num_layers: int = 3
    epochs: int = 200
    lr: float = 0.005
