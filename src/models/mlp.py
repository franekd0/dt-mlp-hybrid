import torch.nn as nn
import torch
from src.models.mlp_encoder import MLPEncoder


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, num_layers: int, num_classes: int):
        super(MLP, self).__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    # Methods eval(), train() and parameters() are inherited from nn.Module

    def forward(self, X):
        """
        Forward pass through the MLP model.
        """
        embeddings = self.encoder(X)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def predict(self, X):
        """
        Returns class predictions for input data X.
        """
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            logits, _ = self(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return float((y_pred == y_test).mean())


