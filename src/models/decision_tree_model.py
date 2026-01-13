from sklearn.tree import DecisionTreeClassifier
from .tree_model import TreeModel

class DecisionTreeModel(TreeModel):
    """
    A wrapper for the Decision Tree Classifier from scikit-learn.
    """

    def __init__(self, max_depth: int = None, random_state = None, min_impurity_decrease: float = 0.0):
        self.model = DecisionTreeClassifier(
            criterion="entropy",
            min_impurity_decrease=min_impurity_decrease,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X, y):
        """
        Fit the Decision Tree model.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the Decision Tree model.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Score the Decision Tree model.
        """
        return self.model.score(X, y)
