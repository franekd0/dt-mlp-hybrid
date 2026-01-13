from abc import ABC, abstractmethod
from typing import Any

class TreeModel(ABC):
    """
    Abstract base class for tree models.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any):
        """
        Fit the tree model.
        """
        pass

    @abstractmethod
    def predict(self, X: Any):
        """
        Predict using the tree model.
        """
        pass

    @abstractmethod
    def score(self, X: Any, y: Any):
        """
        Score the tree model.
        """
        pass