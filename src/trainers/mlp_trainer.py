from src.models import MLP
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def train_mlp(model: MLP, lr: float, epochs: int, X: np.ndarray, y: np.ndarray):
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    loss_history = []

    for _ in range(epochs):
        optimizer.zero_grad()

        logits, _ = model(X_t)
        loss = criterion(logits, y_t)

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return loss_history


