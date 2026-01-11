from src.utils.time_utils import timer
from copy import deepcopy
import numpy as np
from src.models import HybridModel, DecisionTreeModel, MLPTrainer
from src.experiments.dataset_factory import load_dataset
from src.utils.visualization_utils import compare_models_viz


def run_model(model, X_train, y_train, X_test, y_test):
    @timer
    def train():
        model.fit(X_train, y_train)

    time = train()[1]

    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)

    return {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "gap": acc_train - acc_test,
        "time": time,
        "model": model
    }


def run_single_experiment(cfg, do_plots=False):
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(cfg)

    models = {
        "tree": DecisionTreeModel(
            max_depth=cfg.tree_max_depth,
            random_state=cfg.random_state
        ),
        "mlp": MLPTrainer(
            input_dim=X_train.shape[1],
            hidden_dim=cfg.hidden_dim,
            embedding_dim=cfg.embedding_dim,
            num_layers=cfg.num_layers,
            num_classes=len(set(y_train)),
            lr=cfg.lr,
            epochs=cfg.epochs

        ),
        "hybrid": HybridModel(
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
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        res = run_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        results[name] = {
            "acc_train": res["acc_train"],
            "acc_test": res["acc_test"],
            "gap": res["gap"],
            "time": res["time"]
        }
        trained_models[name] = res["model"]

    if do_plots:
        compare_models_viz(
            X_test, y_test,
            trained_models["tree"],
            trained_models["mlp"],
            trained_models["hybrid"]
        )

    return {
        "experiment": cfg.name,
        "dataset": cfg.dataset_name,
        "results": results
    }


def run_experiment_avg(cfg, do_plots=False):
    acc_test = {"tree": [], "mlp": [], "hybrid": []}
    gap = {"tree": [], "mlp": [], "hybrid": []}
    time = {"tree": [], "mlp": [], "hybrid": []}

    for i in range(cfg.n_runs):
        cfg_i = deepcopy(cfg)
        cfg_i.random_state = cfg.random_state + i

        res = run_single_experiment(cfg_i, do_plots=do_plots)

        for m in acc_test:
            acc_test[m].append(res["results"][m]["acc_test"])
            gap[m].append(res["results"][m]["gap"])
            time[m].append(res["results"][m]["time"])

    summary = {
        m: {
            "acc_test_mean": float(np.mean(acc_test[m])),
            "acc_test_std": float(np.std(acc_test[m])),
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
        "results": summary
    }
