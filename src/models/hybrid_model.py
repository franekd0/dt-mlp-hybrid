import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.mlp import MLP
from src.models.decision_tree_model import DecisionTreeModel


class HybridModel:
    """
    Model hybrydowy: MLP (Encoder) + Drzewo Decyzyjne.
    Cała logika treningu i ekstrakcji jest zamknięta wewnątrz tej klasy.
    """

    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            embedding_dim: int = 2,
            hidden_dim: int = 16,
            num_layers: int = 3,
            tree_max_depth: int = 5,
            epochs: int = 200,
            lr: float = 0.01,
            random_state: int = 42
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.epochs = epochs
        self.lr = lr

        self.tree_max_depth = tree_max_depth
        self.random_state = random_state

        self.mlp = None
        self.tree = None
        self.is_fitted = False

    def _train_mlp_internal(self, X_train, y_train):
        """
        Prywatna metoda do trenowania wewnętrznej sieci MLP.
        """
        X_t = torch.FloatTensor(X_train)
        y_t = torch.LongTensor(y_train)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.mlp.parameters(), lr=self.lr)

        self.mlp.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            logits, _ = self.mlp(X_t)

            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  [Hybrid Internal] MLP Epoch {epoch + 1}/{self.epochs} Loss: {loss.item():.4f}")

    def _get_embeddings(self, X):
        """
        Prywatna metoda wyciągająca embeddingi z MLP.
        """
        self.mlp.eval()
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            _, embeddings = self.mlp(X_t)
        return embeddings.numpy()

    def fit(self, X, y):
        """
        Główna metoda ucząca:
        1. Inicjalizuje MLP.
        2. Trenuje MLP na surowych danych.
        3. Transformuje dane na embeddingi.
        4. Trenuje Drzewo na embeddingach.
        """

        self.mlp = MLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )

        print("-> Rozpoczynam trening wewnętrznego MLP...")
        self._train_mlp_internal(X, y)

        X_emb = self._get_embeddings(X)

        print("-> Trenowanie Drzewa na embeddingach...")
        self.tree = DecisionTreeModel(
            max_depth=self.tree_max_depth,
            random_state=self.random_state
        )
        self.tree.fit(X_emb, y)

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Przewiduje klasy dla nowych danych.
        """
        if not self.is_fitted:
            raise Exception("Model nie jest wytrenowany!")

        X_emb = self._get_embeddings(X)
        return self.tree.predict(X_emb)

    def score(self, X, y):
        """
        Zwraca dokładność (accuracy).
        """
        if not self.is_fitted:
            raise Exception("Model nie jest wytrenowany!")

        X_emb = self._get_embeddings(X)
        return self.tree.score(X_emb, y)

    def transform(self, X):
        """
        Zwraca same embeddingi (przydatne do wizualizacji).
        """
        if not self.is_fitted:
            raise Exception("Model nie jest wytrenowany!")
        return self._get_embeddings(X)