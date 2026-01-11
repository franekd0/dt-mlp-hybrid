import torch

from src.models.MLPTrainer import MLPTrainer
from src.models.mlp import MLP
from src.models.decision_tree_model import DecisionTreeModel


class HybridModel:
    """
    Hybrid model: MLP (Encoder) + Decision tree.
    Training and extraction logic is included in this class.
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
        #MLP
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Training
        self.epochs = epochs
        self.lr = lr

        self.tree_max_depth = tree_max_depth
        self.random_state = random_state

        self.mlp : MLP | None = None
        self.tree : DecisionTreeModel | None = None
        self.is_fitted = False

    def _set_mlp(self):
        self.mlp = MLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )

    def _get_embeddings(self, X):
        self.mlp.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            _, embeddings = self.mlp(X_t)
        return embeddings.cpu().numpy()

    def fit(self, X, y):
        """
        Main learning method:
        1. Initialize MLP.
        2. Train MLP with raw data.
        3. Extracts embeddings.
        4. Trains tree on extracted embeddings.
        """

        self._set_mlp()
        trainer = MLPTrainer(self.lr, self.epochs)
        trainer.set_model(self.mlp)
        trainer.fit(X, y)

        X_emb = self._get_embeddings(X)

        self.tree = DecisionTreeModel(
            max_depth=self.tree_max_depth,
            random_state=self.random_state
        )
        self.tree.fit(X_emb, y)

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predicts classes.
        """
        if not self.is_fitted:
            raise Exception("Train model first")

        X_emb = self._get_embeddings(X)
        return self.tree.predict(X_emb)

    def score(self, X, y):
        """
        Returns accuracy.
        """
        if not self.is_fitted:
            raise Exception("Train model first")

        X_emb = self._get_embeddings(X)
        return self.tree.score(X_emb, y)

    def transform(self, X):
        """
        Returns embeddings (for visualization).
        """
        if not self.is_fitted:
            raise Exception("Train model first")
        return self._get_embeddings(X)