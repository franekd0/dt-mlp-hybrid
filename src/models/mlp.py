import torch.nn as nn
from src.models import MLPEncoder


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, num_layers: int, num_classes: int):
        super(MLP, self).__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the MLP model.
        """
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return logits, embeddings
