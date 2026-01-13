import torch

from src.models.mlp import MLP
from src.models.decision_tree_model import DecisionTreeModel
from src.models.tree_model import TreeModel


class HybridModel:
    """
    Hybrid model: MLP (Encoder) + Decision tree.
    Training and extraction logic is included in this class.
    """

    def __init__(
            self,
            tree_model: TreeModel,
            mlp: MLP,
            tree_max_depth: int = 5,
            random_state: int = 42
    ):
        self.tree_max_depth = tree_max_depth
        self.random_state = random_state

        self.mlp : MLP = mlp
        self.tree : TreeModel = tree_model
        self.is_fitted = True


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


        self.tree.fit(self.transform(X), y)

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predicts classes.
        """
        if not self.is_fitted:
            raise Exception("Train model first")

        return self.tree.predict(self.transform(X))

    def score(self, X, y):
        """
        Returns accuracy.
        """
        if not self.is_fitted:
            raise Exception("Train model first")
        return self.tree.score(self.transform(X), y)

    def transform(self, X):
        """
        Returns embeddings (for visualization).
        """
        if not self.is_fitted:
            raise Exception("Train model first")
        return self._get_embeddings(X)