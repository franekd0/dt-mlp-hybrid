import numpy as np
import torch.nn as nn


def mlp_embedding_importance(mlp):
    """
    Importance of embedding dimensions based on output-layer weights.
    """
    last_linear = None
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    if last_linear is None:
        raise ValueError("MLP has no Linear layers")

    W = last_linear.weight.detach().cpu().numpy()  # (classes, emb_dim)
    importance = np.sum(np.abs(W), axis=0)
    importance /= importance.sum() + 1e-12
    return importance
