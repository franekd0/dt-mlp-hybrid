import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D


def plot_embeddings_pca(hybrid_model, X, y, title="Hybrid Embeddings (PCA)"):
    embeddings = hybrid_model.transform(X)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=y,
        cmap='coolwarm',
        edgecolors='k',
        s=60,
        alpha=0.8
    )

    plt.title(f"{title}\n(PCA Reduction from {embeddings.shape[1]} dims)")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.colorbar(scatter, label="Class Label")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


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

        if "Hybrid" in model_name:
            title_suffix = "(Hybrid Embeddings)"
    elif hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        tree_backend = model.model
        title_suffix = "(Original Features)"
    elif hasattr(model, "feature_importances_"):
        tree_backend = model

    if tree_backend is None or not hasattr(tree_backend, "feature_importances_"):
        print(f"Skipping feature importance for {model_name} (no tree found)")
        return

    importances = tree_backend.feature_importances_
    indices = np.argsort(importances)[::-1]

    if feature_names is None:
        if "Hybrid" in model_name:
            names = [f"Emb {i}" for i in range(len(importances))]
        else:
            names = [f"Feat {i}" for i in range(len(importances))]
    else:
        names = feature_names

    sorted_names = [names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 5))
    plt.title(f"Feature Importance: {model_name} {title_suffix}")
    plt.bar(range(len(importances)), sorted_importances, align="center", color='teal', alpha=0.7)
    plt.xticks(range(len(importances)), sorted_names, rotation=45, ha='right')
    plt.ylabel("Gini Importance")
    plt.tight_layout()
    plt.show()


def plot_mlp_feature_importance(model, model_name, feature_names=None):
    """
    Plots feature importance for MLP based on the magnitude of weights in the first layer.
    """
    first_layer_weights = None

    if hasattr(model, "modules"):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_layer_weights = module.weight.detach().cpu().numpy()
                break

    if first_layer_weights is None:
        print(f"Could not find Linear layer in {model_name} for feature importance.")
        return

    importances = np.mean(np.abs(first_layer_weights), axis=0)
    indices = np.argsort(importances)[::-1]

    if feature_names is None:
        names = [f"Feat {i}" for i in range(len(importances))]
    else:
        names = feature_names
        if len(names) != len(importances):
            names = [f"Feat {i}" for i in range(len(importances))]

    sorted_names = [names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 5))
    plt.title(f"MLP Input Sensitivity (First Layer Weights): {model_name}")
    plt.bar(range(len(importances)), sorted_importances, align="center", color='purple', alpha=0.7)
    plt.xticks(range(len(importances)), sorted_names, rotation=45, ha='right')
    plt.ylabel("Mean Abs Weight Magnitude")
    plt.tight_layout()
    plt.show()


def visualize_mlp_structure(model, model_name, feature_names=None):
    layers_weights = []
    layer_sizes = []

    if hasattr(model, "modules"):
        first_layer_found = False
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if not first_layer_found:
                    layer_sizes.append(m.in_features)
                    first_layer_found = True
                layers_weights.append(m.weight.detach().cpu().numpy())
                layer_sizes.append(m.out_features)

    if not layers_weights:
        print(f"Nie znaleziono warstw liniowych w {model_name}")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    ax.axis('off')

    n_layers = len(layer_sizes)
    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(n_layers - 1)

    def get_neuron_pos(layer_idx, neuron_idx, num_neurons_in_layer):
        max_n = max(layer_sizes)
        layer_height = num_neurons_in_layer * v_spacing
        y_offset = (1.0 - layer_height) / 2.0

        x = layer_idx * h_spacing
        y = y_offset + neuron_idx * v_spacing
        return x, y

    for l_idx, weights in enumerate(layers_weights):
        n_next, n_curr = weights.shape
        w_max = np.abs(weights).max()
        if w_max == 0: w_max = 1

        for i in range(n_curr):
            for j in range(n_next):
                weight = weights[j, i]
                if abs(weight) < 0.05 * w_max: continue

                x1, y1 = get_neuron_pos(l_idx, i, n_curr)
                x2, y2 = get_neuron_pos(l_idx + 1, j, n_next)

                color = 'royalblue' if weight > 0 else 'crimson'
                alpha = min(1.0, abs(weight) / w_max)
                linewidth = (abs(weight) / w_max) * 2.0

                line = Line2D([x1, x2], [y1, y2], c=color, alpha=alpha, linewidth=linewidth)
                ax.add_line(line)

    for l_idx, n_neurons in enumerate(layer_sizes):
        for i in range(n_neurons):
            x, y = get_neuron_pos(l_idx, i, n_neurons)

            circle = plt.Circle((x, y), v_spacing / 4.0, color='white', ec='k', zorder=4)
            ax.add_patch(circle)

            if l_idx == 0:
                if feature_names and i < len(feature_names):
                    plt.text(x - 0.02, y, feature_names[i],
                             ha='right', va='center', fontsize=10, fontweight='bold', color='#333333')

            if i == 0:
                if l_idx == 0:
                    label = "Input Layer"
                elif l_idx == n_layers - 1:
                    label = "Output Layer"
                else:
                    label = f"Hidden {l_idx}"
                plt.text(x, -0.05, label, ha='center', fontsize=12, fontweight='bold')

    plt.title(f"Neural Network Architecture: {model_name}\n(Blue: +, Red: -, Thickness: Magnitude)", fontsize=14)
    plt.tight_layout()
    plt.show()