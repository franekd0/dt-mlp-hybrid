import torch.nn as nn

class MLPEncoder(nn.Module):
    """
    Multi-Layer Perceptron Encoder.
    Return embeddings for input.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(MLPEncoder, self).__init__()
        layers = list()
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MLP.
        """
        return self.network(x)
