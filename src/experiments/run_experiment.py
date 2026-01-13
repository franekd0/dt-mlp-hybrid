from typing import Any
from numpy import ndarray

from src.experiments.experiment_config import ExperimentConfig
from src.models.tree_factory import create_tree
from src.utils.time_utils import timer
from copy import deepcopy
import numpy as np
from src.models import HybridModel, MLPTrainer, MLP
from src.experiments.dataset_factory import load_dataset


def run_model(model: Any, X_train: ndarray, y_train: ndarray, X_test: ndarray, y_test: ndarray, time: float):
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)

    return {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "gap": acc_train - acc_test,
        "time": time,
        "model": model,
    }


def get_trained_tree(cfg: ExperimentConfig, X_train: ndarray, y_train: ndarray):
    @timer
    def train():
        return tree.fit(X_train, y_train)

    tree = create_tree(cfg, "tree")

    return train()


def get_trained_mlp(cfg: ExperimentConfig, X_train: ndarray, y_train: ndarray):
    @timer
    def train():
        return [trainer.fit(X_train, y_train), trainer.loss_history]

    mlp = MLP(
        input_dim=X_train.shape[1],
        hidden_dim=cfg.mlp.hidden_dim,
        embedding_dim=cfg.mlp.embedding_dim,
        num_layers=cfg.mlp.num_layers,
        num_classes=len(set(y_train)),
    )

    trainer = MLPTrainer(
        mlp,
        lr=cfg.mlp.lr,
        epochs=cfg.mlp.epochs
    )

    return train()


def get_trained_hybrid(cfg: ExperimentConfig, X_train: ndarray, y_train: ndarray, mlp: MLP, time_from_mlp: float):
    @timer
    def train():
        return hybrid.fit(X_train, y_train)

    hybrid = HybridModel(
        tree_max_depth=cfg.hybrid.tree_max_depth,
        mlp=mlp,
        tree_model=create_tree(cfg, "hybrid")
    )

    model, time = train()

    return model, time + time_from_mlp


def run_single_experiment(cfg: ExperimentConfig):
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(cfg)

    tree_results = get_trained_tree(cfg, X_train, y_train)
    (mlp_results, mlp_loss), mlp_time = get_trained_mlp(cfg, X_train, y_train)
    hybrid_results = get_trained_hybrid(cfg, X_train, y_train, mlp_results, mlp_time)

    models = {
        "tree": tree_results,
        "mlp": (mlp_results, mlp_loss),
        "hybrid": hybrid_results

    }

    results = {}
    trained_models = {}

    for name, (model, time) in models.items():
        res = run_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            time=time
        )

        results[name] = {
            "acc_train": res["acc_train"],
            "acc_test": res["acc_test"],
            "gap": res["gap"],
            "time": res["time"]
        }
        trained_models[name] = res["model"]

    return {
        "experiment": cfg.name,
        "dataset": cfg.dataset_name,
        "results": results,
        "loss_history": mlp_loss
    }


def run_experiment_avg(cfg: ExperimentConfig):
    acc_train = {"tree": [], "mlp": [], "hybrid": []}
    acc_test = {"tree": [], "mlp": [], "hybrid": []}
    gap = {"tree": [], "mlp": [], "hybrid": []}
    time = {"tree": [], "mlp": [], "hybrid": []}

    loss_history = []

    for i in range(cfg.n_runs):
        cfg_i = deepcopy(cfg)
        cfg_i.random_state = cfg.random_state + i

        res = run_single_experiment(cfg_i)

        if i == 0:
            loss_history = res["loss_history"]

        for m in acc_test:
            acc_test[m].append(res["results"][m]["acc_test"])
            acc_train[m].append(res["results"][m]["acc_train"])
            gap[m].append(res["results"][m]["gap"])
            time[m].append(res["results"][m]["time"])

    summary = {
        m: {
            "acc_test_mean": float(np.mean(acc_test[m])),
            "acc_test_std": float(np.std(acc_test[m])),
            "acc_train_mean": float(np.mean(acc_train[m])),
            "acc_train_std": float(np.std(acc_train[m])),
            "gap_mean": float(np.mean(gap[m])),
            "gap_std": float(np.std(gap[m])),
            "time_mean": float(np.mean(time[m]))
        }
        for m in acc_test
    }

    return {
        "experiment": cfg.name,
        "dataset": cfg.dataset_name,
        "n_runs": cfg.n_runs,
        "results": summary,
        "loss_history": loss_history
    }