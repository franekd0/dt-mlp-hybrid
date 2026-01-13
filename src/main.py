from src.experiments.experiments_list import EXPERIMENTS
from src.experiments.run_experiment import run_n_experiments

from src.utils.data import get_columns
from src.utils.plots import (
    plot_train_test_comparison,
    plot_experiments_loss,
    plot_feature_importance,
    plot_mlp_embedding_importance,
    plot_embeddings_pca,
    visualize_mlp_structure,
    print_table, plot_embedding_importance_comparison,
)


def analyze_overfitting(tree, hybrid):
    tree_gap, tree_std = tree["gap_mean"], tree["gap_std"]
    hybrid_gap, hybrid_std = hybrid["gap_mean"], hybrid["gap_std"]

    delta = hybrid_gap - tree_gap
    threshold = tree_std + hybrid_std

    print("\n[Overfitting check]")
    print(f"Tree   gap: {tree_gap:.4f} ± {tree_std:.4f}")
    print(f"Hybrid gap: {hybrid_gap:.4f} ± {hybrid_std:.4f}")
    print(f"Δgap (hyb − tree): {delta:.4f}")

    if delta > threshold:
        print("→ Hybrid overfits MORE than Tree")
    elif delta < -threshold:
        print("→ Tree overfits MORE than Hybrid")
    else:
        print("→ Difference small / within variability")


def visualize_experiment(cfg, viz_data):
    """
    Visual diagnostics for a single experiment:
    - what MLP considers important
    - what Hybrid (tree) actually uses
    - how consistent these choices are
    """

    feature_names = get_columns(cfg.dataset_name)

    # --------------------------------------------------
    # 1) MLP: embedding importance (logits perspective)
    # --------------------------------------------------
    plot_mlp_embedding_importance(
        mlp_model=viz_data["mlp_model"],
        title=f"{cfg.name}"
    )

    # --------------------------------------------------
    # 2) Hybrid: embedding usage (tree perspective)
    # --------------------------------------------------
    plot_feature_importance(
        model=viz_data["hybrid_model"],
        model_name=f"{cfg.name} | Hybrid",
    )

    # --------------------------------------------------
    # 3) COMPARISON: MLP vs Hybrid (same embeddings)
    # --------------------------------------------------
    plot_embedding_importance_comparison(
        mlp_model=viz_data["mlp_model"],
        hybrid_model=viz_data["hybrid_model"],
        title=f"{cfg.name} | Embedding importance: MLP vs Hybrid"
    )


def collect_table_rows(results):
    labels, accuracies, times = [], [], []

    for r in results:
        for model_name in ["tree", "mlp", "hybrid"]:
            m = r["results"][model_name]
            labels.append(f"{r['experiment']} | {model_name}")
            accuracies.append(f"{m['acc_test_mean']:.4f} ± {m['acc_test_std']:.4f}")
            times.append(f"{m['time_mean']:.3f}")

    return labels, accuracies, times


def main():
    print("=" * 70)
    print("STARTING EXPERIMENTS")
    print("=" * 70)

    results = []
    all_loss_data = {}

    for cfg in EXPERIMENTS:
        print(f"\nRunning experiment: {cfg.name}")

        res = run_n_experiments(cfg)
        results.append(res)

        if "loss_data" in res:
            all_loss_data[cfg.name] = res["loss_data"]

        analyze_overfitting(
            tree=res["results"]["tree"],
            hybrid=res["results"]["hybrid"],
        )

        # plot_train_test_comparison(res)

        if res.get("viz_data"):
            visualize_experiment(cfg, res["viz_data"])

    print("\nPlotting Loss Comparison (MLP vs Hybrid)...")
    # plot_experiments_loss(all_loss_data)

    labels, accuracies, times = collect_table_rows(results)
    print_table(labels, accuracies, times)

if __name__ == "__main__":
    main()