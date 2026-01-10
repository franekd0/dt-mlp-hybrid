from copy import deepcopy

from src.models import MLP, HybridModel, DecisionTreeModel
from src.training.mlp_training import train_mlp
from src.experiments.dataset_factory import load_dataset
from src.experiments.experiment_config import ExperimentConfig
from src.utils.time_utils import timer
import numpy as np

from src.utils.visualization_utils import compare_models_viz


@timer
def _run_tree(cfg, X_train, X_test, y_train, y_test):
    tree = DecisionTreeModel(
        max_depth=cfg.tree_max_depth,
        random_state=cfg.random_state
    )
    tree.fit(X_train, y_train)
    acc = tree.score(X_test, y_test)
    return acc, tree


@timer
def _run_mlp(cfg, X_train, X_test, y_train, y_test):
    mlp = MLP(
        input_dim=X_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        num_layers=cfg.num_layers,
        num_classes=len(set(y_train))
    )
    train_mlp(mlp, X_train, y_train, cfg.epochs, cfg.lr)
    acc = mlp.score(X_test, y_test)
    return acc, mlp


@timer
def _run_hybrid(cfg, X_train, X_test, y_train, y_test):
    hybrid = HybridModel(
        input_dim=X_train.shape[1],
        num_classes=len(set(y_train)),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        tree_max_depth=cfg.tree_max_depth,
        epochs=cfg.epochs,
        lr=cfg.lr,
        random_state=cfg.random_state
    )
    hybrid.fit(X_train, y_train)
    acc = hybrid.score(X_test, y_test)
    return acc, hybrid


def _run_experiment(cfg: ExperimentConfig, do_plots) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(cfg)

    (acc_tree, tree_model), time_tree = _run_tree(cfg, X_train, X_test, y_train, y_test)
    (acc_mlp, mlp_model), time_mlp = _run_mlp(cfg, X_train, X_test, y_train, y_test)
    (acc_hybrid, hybrid_model), time_hybrid = _run_hybrid(cfg, X_train, X_test, y_train, y_test)

    if do_plots: compare_models_viz(X_test, y_test, tree_model, mlp_model, hybrid_model)
    return {
        "experiment": cfg.name,
        "dataset": cfg.dataset_name,
        "tree": {
            "accuracy": acc_tree,
            "time": time_tree
        },
        "mlp": {
            "accuracy": acc_mlp,
            "time": time_mlp
        },
        "hybrid": {
            "accuracy": acc_hybrid,
            "time": time_hybrid
        }
    }


def run_experiment_avg(cfg: ExperimentConfig, n_runs: int = 10, do_plots: bool = False) -> dict:
    acc = {"tree": [], "mlp": [], "hybrid": []}
    time = {"tree": [], "mlp": [], "hybrid": []}

    for i in range(n_runs):
        cfg_i = deepcopy(cfg)
        cfg_i.random_state = cfg.random_state + i

        res = _run_experiment(cfg_i, do_plots)

        for model in ["tree", "mlp", "hybrid"]:
            acc[model].append(res[model]["accuracy"])
            time[model].append(res[model]["time"])

    summary = {}
    for model in ["tree", "mlp", "hybrid"]:
        summary[model] = {
            "acc_mean": float(np.mean(acc[model])),
            "acc_std": float(np.std(acc[model])),
            "time_mean": float(np.mean(time[model])),
        }

    return {
        "experiment": cfg.name,
        "dataset": cfg.dataset_name,
        "n_runs": n_runs,
        "results": summary
    }
