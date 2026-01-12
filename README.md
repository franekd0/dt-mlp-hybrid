# MLP–Tree Hybrid Classifier

This project implements an innovative hybrid model for tabular data classification, combining nonlinear feature extraction performed by a neural network with the efficiency and interpretability of decision trees.

## 🚀 Project Overview
The core idea of the system is to use an **MLP (Multi-Layer Perceptron)** as an encoder that learns an optimal data representation (embeddings). A **decision tree** is then trained on these extracted features, resulting in a model that combines the strengths of deep learning with the interpretability of tree-based structures.

### Key Features
- **Hybrid architecture**: Integration of an `MLPEncoder` with a `DecisionTreeClassifier`.
- **Model comparison**: Tools for comparing the hybrid model against a standard MLP and a standalone decision tree.
- **Versatility**: Support for various tabular datasets (e.g., Wine, Cancer, Adult, Heart).
- **Statistical analysis**: Ability to run multiple experimental repetitions to compute mean accuracy and standard deviation.

## 🏗️ Detailed System Architecture

### 1. MLP Model (Classifier)
The `MLP` class represents a full neural network used for pre-training data representations. It consists of an encoder (`MLPEncoder`) and a classification head (`classifier`). The `forward` method returns both classification logits and learned embeddings.

```python
# src/models/mlp.py

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, num_classes):
        super(MLP, self).__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, X):
        embeddings = self.encoder(X)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            logits, _ = self(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()
```

### 2. Decision Tree Model (Wrapper)
The `DecisionTreeModel` class wraps scikit-learn’s `DecisionTreeClassifier` to unify the interface with the neural models.

```python
# src/models/decision_tree_model.py

class DecisionTreeModel:
    def __init__(self, max_depth=None, random_state=None):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
```

### 3. Hybrid Model
The `HybridModel` combines neural feature learning with tree-based classification.

```python
# src/models/hybrid_model.py

class HybridModel:
    def fit(self, X, y):
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
        if not self.is_fitted:
            raise Exception("Train model first")
        X_emb = self._get_embeddings(X)
        return self.tree.predict(X_emb)

    def _get_embeddings(self, X):
        self.mlp.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            _, embeddings = self.mlp(X_t)
        return embeddings.cpu().numpy()
```

## 📂 File Structure
| File | Description |
|------|-------------|
| `hybrid_model.py` | Hybrid model orchestration |
| `mlp_encoder.py` | Neural encoder definition |
| `mlp.py` | MLP with classifier head |
| `MLPTrainer.py` | Neural network training logic |
| `decision_tree_model.py` | Decision tree wrapper |
| `dataset_factory.py` | Dataset loading utilities |
| `run_experiment.py` | Experiment runner |
| `experiment_config.py` | Hyperparameter configuration |
| `experiments_list.py` | Predefined experiments |

## 📊 Research Methodology
Each experiment reports:
- Training and test accuracy
- Generalization gap
- Training time

## 🛠️ Usage
1. Define experiments in `experiments_list.py`.
2. Run `run_experiment_avg(cfg)` to obtain averaged metrics.
3. Optionally enable plots with `do_plots=True`.
