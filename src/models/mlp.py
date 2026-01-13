import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.mlp_encoder import MLPEncoder


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            embedding_dim: int,
            num_layers: int,
            num_classes: int,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
    ):
        super(MLP, self).__init__()

        self.encoder = MLPEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.loss_history: list[float] = []

    def forward(self, X: torch.Tensor):
        embeddings = self.encoder(X)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int):
        self.train()

        X_t = torch.FloatTensor(X_train)
        y_t = torch.LongTensor(y_train)

        self.loss_history.clear()

        for _ in range(epochs):
            self.optimizer.zero_grad()

            logits, _ = self(X_t)
            loss = self.criterion(logits, y_t)

            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item())

        return self

    def predict(self, X: np.ndarray):
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            logits, _ = self(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        return float((y_pred == y).mean())
