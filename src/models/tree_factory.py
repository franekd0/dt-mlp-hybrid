from src.models.decision_tree_model import DecisionTreeModel
# from src.models.random_forest_model import RandomForestModel


def create_tree(cfg):
    if cfg.tree_type == "decision_tree":
        return DecisionTreeModel(
            max_depth=cfg.tree_max_depth,
            random_state=cfg.random_state
        )

    # if cfg.tree_type == "random_forest":
    #     return RandomForestModel(
    #         n_estimators=cfg.n_estimators,
    #         random_state=cfg.random_state
    #     )

    raise ValueError(f"Unknown tree_type: {cfg.tree_type}")
