from sklearn.ensemble import RandomForestClassifier
from numpy import ndarray
from .tree_model import TreeModel

class RandomForestModel(TreeModel):
    """
    A wrapper for the Random Forest Classifier from scikit-learn.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X: ndarray, y: ndarray):
        """
        Fit the Random Forest tree model.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: ndarray):
        """
        Predict using the Random Forest tree model.
        """
        return self.model.predict(X)

    def score(self, X: ndarray, y: ndarray):
        """
        Score the Random Forest tree model.
        """
        return self.model.score(X, y)