import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_bar(summary: dict):
    models = ["tree", "mlp", "hybrid"]
    means = [summary["results"][m]["acc_test_mean"] for m in models]
    stds = [summary["results"][m]["acc_test_std"] for m in models]

    plt.figure(figsize=(6, 4))
    plt.bar(models, means, yerr=stds, capsize=6)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(summary["experiment"])
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_train_test_comparison(summary: dict):
    dataset_name = summary["experiment"]
    models = ["tree", "mlp", "hybrid"]

    test_means = [summary["results"][m]["acc_test_mean"] for m in models]
    test_stds = [summary["results"][m]["acc_test_std"] for m in models]

    train_means = [summary["results"][m]["acc_train_mean"] for m in models]
    train_stds = [summary["results"][m]["acc_train_std"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    rects1 = ax.bar(x - width / 2, train_means, width, yerr=train_stds, label='Train',
                    capsize=5, color='skyblue', edgecolor='black', alpha=0.8)
    rects2 = ax.bar(x + width / 2, test_means, width, yerr=test_stds, label='Test',
                    capsize=5, color='salmon', edgecolor='black', alpha=0.8)

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Train vs Test Accuracy: {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_time_all(results):
    plt.figure(figsize=(6, 4))
    for model in ["tree", "mlp", "hybrid"]:
        times = []
        accs = []
        for r in results:
            times.append(r["results"][model]["time_mean"])
            accs.append(r["results"][model]["acc_test_mean"])
        plt.scatter(times, accs, label=model, s=70)

    plt.xlabel("Time [s]")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Time (all experiments)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_accuracy_across_datasets(summaries):
    datasets = [s["dataset"] for s in summaries]
    models = ["tree", "mlp", "hybrid"]
    x = np.arange(len(datasets))
    width = 0.25

    plt.figure(figsize=(8, 4))
    for i, model in enumerate(models):
        means = [s["results"][model]["acc_test_mean"] for s in summaries]
        stds = [s["results"][model]["acc_test_std"] for s in summaries]
        plt.bar(x + i * width, means, width, yerr=stds, capsize=5, label=model)

    plt.xticks(x + width, datasets)
    plt.ylabel("Accuracy")
    plt.title("Accuracy comparison across datasets")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
