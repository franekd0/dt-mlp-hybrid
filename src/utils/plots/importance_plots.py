import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.utils.analysis.embedding_analysis import mlp_embedding_importance


def plot_feature_importance(model, model_name, feature_names=None):
    tree_backend = None
    title_suffix = ""

    if hasattr(model, "tree"):
        if hasattr(model.tree, "tree_"):
            tree_backend = model.tree
        elif hasattr(model.tree, "model"):
            tree_backend = model.tree.model
        else:
            tree_backend = model.tree

        title_suffix = "(Tree view on embeddings)"
    elif hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        tree_backend = model.model
        title_suffix = "(Tree view on original features)"
    elif hasattr(model, "feature_importances_"):
        tree_backend = model
        title_suffix = "(Tree view)"

    if tree_backend is None or not hasattr(tree_backend, "feature_importances_"):
        print(f"Skipping feature importance for {model_name} (no tree found)")
        return

    importances = tree_backend.feature_importances_
    indices = np.argsort(importances)[::-1]

    if feature_names is not None and len(feature_names) == len(importances):
        labels = [feature_names[i] for i in indices]
    else:
        labels = [f"Emb {i + 1}" if "Hybrid" in model_name else f"Feat {i}" for i in indices]

    plt.figure(figsize=(10, 4))
    plt.bar(
        labels,
        importances[indices],
        color="#2aa198",   # TURKUS = DRZEWO
        alpha=0.8,
    )
    plt.ylabel("Gini importance")
    plt.xlabel("Feature")
    plt.title(f"{model_name} {title_suffix}")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_mlp_feature_importance(model, model_name, feature_names=None):
    """
    Feature importance for MLP input features
    based on mean absolute weights of the first Linear layer.
    Sorted from most to least important feature.
    """
    first_layer_weights = None

    for module in model.modules():
        if isinstance(module, nn.Linear):
            first_layer_weights = module.weight.detach().cpu().numpy()
            break

    if first_layer_weights is None:
        print(f"Could not find Linear layer in {model_name}.")
        return

    importances = np.mean(np.abs(first_layer_weights), axis=0)

    # sortowanie malejąco
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]

    if len(feature_names) > 0 and len(feature_names) == len(importances):
        labels = [feature_names[i] for i in indices]
    else:
        labels = [f"Feat {i + 1}" for i in indices]

    plt.figure(figsize=(10, 4))
    plt.bar(
        labels,
        sorted_importances,
        color="#8e44ad",   # spójny kolor MLP
        alpha=0.8
    )
    plt.ylabel("Relative importance")
    plt.xlabel("Input feature")
    plt.title(f"{model_name} | MLP input feature importance")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_mlp_embedding_importance(mlp_model, title):
    """
    Embedding importance for standalone MLP
    based on absolute output-layer weights.
    Sorted from most to least important embedding.
    """
    importance = mlp_embedding_importance(mlp_model)

    # sortowanie malejąco
    indices = np.argsort(importance)[::-1]
    sorted_importance = importance[indices]
    labels = [f"Emb {i + 1}" for i in indices]

    plt.figure(figsize=(10, 4))
    plt.bar(
        labels,
        sorted_importance,
        color="#8e44ad",
        alpha=0.8
    )
    plt.ylabel("Relative importance")
    plt.xlabel("Embedding dimension")
    plt.title(f"{title} | MLP embedding importance")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_embedding_importance_comparison(
        mlp_model,
        hybrid_model,
        title,
):
    """
    Compares embedding importance:
    - MLP: output-layer sensitivity
    - Hybrid: tree feature_importances_
    """
    mlp_imp = mlp_embedding_importance(mlp_model)
    tree_imp = hybrid_model.tree.model.feature_importances_

    if len(mlp_imp) != len(tree_imp):
        raise ValueError("Embedding dimension mismatch between MLP and Hybrid tree")

    labels = [f"Emb {i + 1}" for i in range(len(mlp_imp))]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, mlp_imp, width, label="MLP", color="purple", alpha=0.75)
    plt.bar(x + width / 2, tree_imp, width, label="Hybrid Tree", color="teal", alpha=0.75)

    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Relative importance")
    plt.xlabel("Embedding dimension")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
