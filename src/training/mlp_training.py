import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.time_utils import timer


@timer
def train_mlp(
    model,
    X,
    y,
    epochs=200,
    lr=0.01,
    debug : tuple[bool, int] = False,
    ):
    """
    Generic training loop for MLP-like models.
    """
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits, _ = model(X_t)
        loss = criterion(logits, y_t)

        loss.backward()
        optimizer.step()

        if debug[0] and (epoch + 1) % debug[1] == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")
