import torch
import torch.nn as nn
import torch.optim as optim

from src.models import MLP


class MLPTrainer:

    def __init__(
            self,
            lr: float,
            epochs: int,
            input_dim: int | None = None,
            hidden_dim: int | None = None,
            embedding_dim: int | None = None,
            num_layers: int | None = None,
            num_classes: int | None = None,
    ):
        self.lr = lr
        self.epochs = int(epochs)
        self.model: MLP | None = None
        self.optimizer: optim.Adam | None = None
        self.criterion = nn.CrossEntropyLoss()

        if input_dim is not None:
            self.model = MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                num_classes=num_classes
            )
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=1e-4
            )

    def set_model(self, model):
        self.model = model
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )

    def fit(self, X, y):
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)

        self.model.train()

        for _ in range(self.epochs):
            self.optimizer.zero_grad()

            logits, _ = self.model(X_t)
            loss = self.criterion(logits, y_t)

            loss.backward()
            self.optimizer.step()

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
