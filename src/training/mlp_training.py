import torch
import torch.nn as nn
import torch.optim as optim



def train_mlp(
    model,
    X,
    y,
    epochs=200,
    lr=0.01,
    debug : bool = False,
    ):
    """
    Generic training loop for MLP-like models.
    """
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits, _ = model(X_t)
        loss = criterion(logits, y_t)

        loss.backward()
        optimizer.step()

        if debug and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")
