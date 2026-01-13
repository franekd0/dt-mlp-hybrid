from src.experiments.experiment_config import ComparisonExperimentConfig
from src.models.decision_tree_model import DecisionTreeModel
from src.models.random_forest_model import RandomForestModel
from src.models.tree_model import TreeModel


def create_tree(cfg: ComparisonExperimentConfig, for_hybrid: bool) -> TreeModel:
    tree_depth = cfg.hybrid.tree_max_depth if for_hybrid else cfg.tree.max_depth
    if cfg.tree_type == "decision_tree":
        return DecisionTreeModel(
            max_depth=tree_depth,
            random_state=cfg.random_state,
            min_impurity_decrease=0.0 if for_hybrid else 0.0,
        )

    if cfg.tree_type == "random_forest":
        return RandomForestModel(
            max_depth=tree_depth,
            n_estimators=cfg.tree.n_estimators,
            random_state=cfg.random_state
        )

    raise ValueError(f"Unknown tree_type: {cfg.tree_type}")
