import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models import MLP
from src.utils.data_utils import load_wine_dataset


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_wine_dataset(
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=32,
        shuffle=False
    )

    input_dim = X_train.shape[1]   # 13 dla wine
    num_classes = len(set(y_train))  # 3 klasy

    model = MLP(
        input_dim=input_dim,
        hidden_dim=128,
        embedding_dim=64,
        num_layers=3,
        num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 30

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()

            logits, _ = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                logits, _ = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(
            f"Epoch {epoch+1:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
