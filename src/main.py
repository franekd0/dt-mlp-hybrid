from src.experiments.experiments_list import EXPERIMENTS
from src.experiments.run_experiment import run_experiment_avg
from src.utils.visualization_utils import print_table


def main():
    results = []

    print("=" * 60)
    print("STARTING EXPERIMENTS")
    print("=" * 60)

    for cfg in EXPERIMENTS:
        print(f"\nRunning experiment: {cfg.name}")
        res = run_experiment_avg(cfg, n_runs=10, do_plots=False)
        results.append(res)

    labels = []
    accuracies = []
    times = []

    for r in results:
        for model_name in ["tree", "mlp", "hybrid"]:
            m = r["results"][model_name]
            labels.append(f"{r['experiment']} | {model_name}")
            accuracies.append(f"{m['acc_mean']:.4f} ± {m['acc_std']:.4f}")
            times.append(f"{m['time_mean']:.3f}")

    print_table(labels, accuracies, times)


if __name__ == "__main__":
    main()
