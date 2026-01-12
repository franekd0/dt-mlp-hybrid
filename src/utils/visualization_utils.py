import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D


def print_results(time, accuracy):
    print(f"-> Accuracy: {accuracy:.4f}\n-> Time: {time:.4f}s")


def print_table(labels, accuracies, times):
    print("\n" + "=" * 110)
    print(f"{'Model':<70} | {'Accuracy (mean ± std)':<22} | {'Time [s]':<10}")
    print("-" * 110)
    for i in range(len(labels)):
        print(f"{labels[i]:<70} | {accuracies[i]:<22} | {times[i]:<10}")
    print("=" * 110)


def add_class_legend(fig):
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Class 0',
               markerfacecolor=plt.cm.coolwarm(0.1), markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Class 1',
               markerfacecolor=plt.cm.coolwarm(0.9), markersize=8),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2)


def plot_decision_boundary(model, X, y, ax, title="Decyzja", is_pytorch=False, is_hybrid=False):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    mesh_data = np.c_[xx.ravel(), yy.ravel()]

    if is_hybrid:
        Z = model.predict(mesh_data)
    elif is_pytorch:
        model.eval()
        with torch.no_grad():
            t_data = torch.FloatTensor(mesh_data)
            logits, _ = model(t_data)
            Z = torch.argmax(logits, dim=1).numpy()
    else:
        Z = model.predict(mesh_data)

    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=30)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def plot_embeddings(hybrid_model, X, y, ax, title="Embeddingi MLP"):
    embeddings = hybrid_model.transform(X)

    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=30)
    ax.set_title(title)
    ax.set_xlabel("Emb 1")
    ax.set_ylabel("Emb 2")

    x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
    y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = hybrid_model.tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.1, cmap='Greys')


def compare_models_viz(X, y, tree_model, mlp_model, hybrid_model):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    add_class_legend(fig)
    plot_decision_boundary(tree_model, X, y, axes[0, 0], title="1. Tree only")
    plot_decision_boundary(mlp_model, X, y, axes[0, 1], title="2. MLP only", is_pytorch=True)
    plot_decision_boundary(hybrid_model, X, y, axes[1, 0], title="3. Hybrid", is_hybrid=True)
    plot_embeddings(hybrid_model, X, y, axes[1, 1], title="4. Wnętrze Hybrydy (Embeddingi + Drzewo)")

    plt.tight_layout()
    plt.show()


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


def plot_experiments_loss(all_experiments_loss: dict):
    plt.figure(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_experiments_loss)))

    for idx, (exp_name, losses) in enumerate(all_experiments_loss.items()):
        color = colors[idx]

        if "mlp" in losses and losses["mlp"]:
            plt.plot(losses["mlp"], label=f"{exp_name} | MLP",
                     color=color, linestyle="-", linewidth=2, alpha=0.9)

        if "hybrid" in losses and losses["hybrid"]:
            plt.plot(losses["hybrid"], label=f"{exp_name} | Hybrid (Encoder)",
                     color=color, linestyle="--", linewidth=2, alpha=0.9)

    plt.title("Loss Comparison: Standalone MLP vs Hybrid Internal MLP", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (CrossEntropy)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
