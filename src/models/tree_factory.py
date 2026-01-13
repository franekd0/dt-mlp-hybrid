from src.experiments.experiment_config import ExperimentConfig
from src.models.decision_tree_model import DecisionTreeModel
from src.models.random_forest_model import RandomForestModel
from src.models.tree_model import TreeModel


def create_tree(cfg: ExperimentConfig, mode: str) -> TreeModel:
    if mode == "tree":
        if cfg.tree.type == "decision_tree":
            return DecisionTreeModel(
                max_depth=cfg.tree.max_depth,
                random_state=cfg.random_state
            )

        if cfg.tree.type == "random_forest":
            return RandomForestModel(
                n_estimators=cfg.tree.n_estimators,
                max_depth=cfg.tree.max_depth,
                random_state=cfg.random_state
            )

    if mode == "hybrid":
        if cfg.hybrid.tree_type == "decision_tree":
            return DecisionTreeModel(
                max_depth=cfg.hybrid.tree_max_depth,
                random_state=cfg.random_state
            )

        if cfg.hybrid.tree_type == "random_forest":
            return RandomForestModel(
                n_estimators=cfg.hybrid.n_estimators,
                max_depth=cfg.hybrid.tree_max_depth,
                random_state=cfg.random_state
            )

    raise ValueError(f"Unknown tree_type: {cfg.tree.type}")
