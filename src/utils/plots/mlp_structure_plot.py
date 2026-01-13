import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
                if len(feature_names) > 0 and i < len(feature_names):
                    plt.text(x - 0.02, y, feature_names[i],
                             ha='right', va='center', fontsize=10, fontweight='bold', color='#333333')

            if i == 0:
                if l_idx == 0:
                    label = "Input Layer"
                elif l_idx == n_layers - 1:
                    label = "Logits"
                elif l_idx == n_layers - 2:
                    label = "Embeddings"
                else:
                    label = f"Hidden {l_idx}"
                plt.text(x, -0.05, label, ha='center', fontsize=12, fontweight='bold')

    plt.title(f"Neural Network Architecture", fontsize=14)
    plt.tight_layout()
    plt.show()
