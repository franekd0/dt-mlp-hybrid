import torch
import torch.nn as nn
import torch.optim as optim

from src.models import MLP


class MLPTrainer:

    def __init__(
            self,
            model : MLP,
            lr: float,
            epochs: int,
    ):
        self.lr = lr
        self.epochs = int(epochs)
        self.model: MLP = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        self.loss_history = []

    def fit(self, X, y):
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)

        self.model.train()
        self.loss_history = []

        for _ in range(self.epochs):
            self.optimizer.zero_grad()

            logits, _ = self.model(X_t)
            loss = self.criterion(logits, y_t)

            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
        return self.model
