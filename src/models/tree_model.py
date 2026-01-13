from abc import ABC, abstractmethod
import numpy as np

class TreeModel(ABC):
    """
    Abstract base class for tree models.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the tree model.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        """
        Predict using the tree model.
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray):
        """
        Score the tree model.
        """
        pass

    def get_name(self):
        return self.name