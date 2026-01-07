import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_mlp(model, X_train, y_train, epochs=200, lr=0.01, print_every=50):
    """
    Training model MLP.
    """
    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits, _ = model(X_t)

        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % print_every == 0:
            print(f"[MLP] Epoka {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    return loss_history


def get_embeddings(model, X):
    """
    Data's flow through MLP and returns embeddings.
    """
    model.eval()
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        _, embeddings = model(X_t)
    return embeddings.numpy()


def evaluate_mlp(model, X, y):
    """
    MLP accuracy.
    """
    model.eval()
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    with torch.no_grad():
        logits, _ = model(X_t)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_t).float().mean()
    return accuracy.item()