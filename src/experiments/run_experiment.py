from src.models.tree_factory import create_tree
from src.trainers.mlp_trainer import train_mlp
from src.utils.time.time_utils import timer
from copy import deepcopy
import numpy as np
from src.models import HybridModel, MLP
from src.experiments.dataset_factory import load_dataset


def _run_model(model, X_train, y_train, X_test, y_test):
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)

    return {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "gap": acc_train - acc_test,
        "model": model,
    }


def _get_trained_tree(cfg, X_train, y_train):
    @timer
    def train():
        return tree.fit(X_train, y_train)

    tree = create_tree(cfg, for_hybrid=False)

    return train()


def _get_trained_mlp(cfg, X_train, y_train):
    @timer
    def train():
        loss_history = train_mlp(mlp, cfg.mlp.lr, cfg.mlp.epochs, X_train, y_train)
        return mlp, loss_history

    mlp = MLP(
        input_dim=X_train.shape[1],
        hidden_dim=cfg.mlp.hidden_dim,
        embedding_dim=cfg.mlp.embedding_dim,
        num_layers=cfg.mlp.num_layers,
        num_classes=len(set(y_train)),
    )

    return train()


def _get_trained_hybrid(cfg, X_train, y_train, mlp, time_from_mlp):
    @timer
    def train():
        return hybrid.fit(X_train, y_train)

    hybrid = HybridModel(
        mlp=mlp,
        tree_model=create_tree(cfg, for_hybrid=True),
    )

    model, time = train()

    return model, time + time_from_mlp


def _run_experiment(cfg):
    X_train, X_test, y_train, y_test = load_dataset(cfg)

    tree_results = _get_trained_tree(cfg, X_train, y_train)
    (mlp, loss_history), mlp_time = _get_trained_mlp(cfg, X_train, y_train)
    hybrid_results = _get_trained_hybrid(cfg, X_train, y_train, mlp, mlp_time)

    models = {
        "tree": tree_results,
        "mlp": (mlp, mlp_time),
        "hybrid": hybrid_results

    }

    results = {}
    trained_models = {}

    for name, (model, time) in models.items():
        res = _run_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results[name] = {
            "acc_train": res["acc_train"],
            "acc_test": res["acc_test"],
            "gap": res["gap"],
            "time": time
        }

    viz_data = {
        "hybrid_model": hybrid_results[0],
        "tree_model": tree_results[0],
        "mlp_model": mlp,
        "X": X_test,
        "y": y_test
    }

    return {
        "experiment": cfg.name,
        "dataset": cfg.dataset_name,
        "results": results,
        "viz_data": viz_data,
        "loss_history": loss_history
    }


def run_n_experiments(cfg):
    acc_train = {"tree": [], "mlp": [], "hybrid": []}
    acc_test = {"tree": [], "mlp": [], "hybrid": []}
    gap = {"tree": [], "mlp": [], "hybrid": []}
    time = {"tree": [], "mlp": [], "hybrid": []}

    loss_histories = []

    for i in range(cfg.n_runs):
        cfg_i = deepcopy(cfg)
        cfg_i.random_state = cfg.random_state + i

        res = _run_experiment(cfg_i)

        loss_histories.append(res["loss_history"])

        if i == 0: viz_data = res["viz_data"]

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
        "viz_data": viz_data,
        "loss_history": loss_histories,
    }

def run_tree_depth_sweep(
    mlp,
    tree_depths,
    base_cfg,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Runs a sweep over tree depth for a fixed MLP.
    Trains and evaluates only the hybrid model.
    """

    results = []

    # upewniamy się, że MLP jest zamrożone
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad = False

    for depth in tree_depths:
        cfg = deepcopy(base_cfg)
        cfg.hybrid.tree_max_depth = depth

        tree = create_tree(cfg, for_hybrid=True)
        hybrid = HybridModel(
            mlp=mlp,
            tree_model=tree,
        )

        hybrid.fit(X_train, y_train)

        acc_train = hybrid.score(X_train, y_train)
        acc_test = hybrid.score(X_test, y_test)
        y_pred = hybrid.predict(X_test)
        errors = (y_pred != y_test).sum()
        print(f"depth={depth} | errors={errors}/{len(y_train)} | acc={hybrid.score(X_test, y_test)}")

        results.append({
            "tree_depth": depth,
            "acc_train": acc_train,
            "acc_test": acc_test,
            "gap": acc_train - acc_test,
        })

    return results
