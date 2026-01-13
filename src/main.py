from src.experiments.experiments_list import EXPERIMENTS, EXPERIMENTS_SWEEPS
from src.experiments.run_experiment import run_n_experiments
from src.utils.visualization_utils import (
    print_table,
    plot_accuracy_bar,
    plot_accuracy_vs_time_all,
    plot_accuracy_across_datasets,
    plot_experiments_loss,
    plot_train_test_comparison
)


def main():
    results = []
    all_loss_data = {}

    print("=" * 70)
    print("STARTING EXPERIMENTS")
    print("=" * 70)

    for cfg in EXPERIMENTS:
        print(f"\nRunning experiment: {cfg.name}")

        res = run_n_experiments(cfg)

        if "loss_data" in res:
            all_loss_data[cfg.name] = res["loss_data"]

        tree = res["results"]["tree"]
        hybrid = res["results"]["hybrid"]

        tree_gap = tree["gap_mean"]
        hybrid_gap = hybrid["gap_mean"]

        tree_std = tree["gap_std"]
        hybrid_std = hybrid["gap_std"]

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

        plot_accuracy_bar(res)
        plot_train_test_comparison(res)

        results.append(res)

    print("\nPlotting Loss Comparison (MLP vs Hybrid)...")
    plot_experiments_loss(all_loss_data)

    plot_accuracy_vs_time_all(results)
    plot_accuracy_across_datasets(results)

    labels = []
    accuracies = []
    times = []

    for r in results:
        for model_name in ["tree", "mlp", "hybrid"]:
            m = r["results"][model_name]
            labels.append(f"{r['experiment']} | {model_name}")
            accuracies.append(
                f"{m['acc_test_mean']:.4f} ± {m['acc_test_std']:.4f}"
            )
            times.append(
                f"{m['time_mean']:.3f}"
            )

    print_table(labels, accuracies, times)


def test():

    results = run_sweep(EXPERIMENTS_SWEEPS[0])
    import matplotlib.pyplot as plt

    xs = [results["value"] for r in results]
    ys = [results["acc_test"] for r in results]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Hybrid tree max depth")
    plt.ylabel("Test accuracy")
    plt.title("Sweep")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
    # test()