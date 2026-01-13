from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TreeConfig:
    type: str = "decision_tree"
    max_depth: int = 8
    n_estimators: int = 10

@dataclass
class MLPConfig:
    embedding_dim: int = 16
    hidden_dim: int = 32
    num_layers: int = 2
    epochs: int = 50
    lr: float = 0.002


@dataclass
class HybridConfig:
    tree_type: str = "decision_tree"
    tree_max_depth: int = 5
    n_estimators: int = 10

@dataclass
class ExperimentConfig:

    name: str
    dataset_name: str
    random_state: int = 42
    n_runs: int = 100

    test_size: float = 0.2
    val_size: float = 0.2
    n_samples: Optional[int] = None

    tree: TreeConfig = field(default_factory=TreeConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)