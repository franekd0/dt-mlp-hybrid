# MLP-Tree Hybrid Classifier

This repository experiments with a hybrid classifier for tabular data. It compares three approaches:

- a tree model trained directly on the original features,
- an MLP classifier trained on the original features,
- a hybrid model that trains a tree on embeddings produced by a trained MLP encoder.

The goal is to test whether a neural representation can improve tree-based classification while still keeping the final decision stage simple and inspectable.

## Project Structure

```text
src/
  main.py                         Experiment entry point
  experiments/
    dataset_factory.py            Dataset loading and preprocessing
    experiment_config.py          Dataclass-based experiment configuration
    experiments_list.py           Predefined experiment list
    run_experiment.py             Training, evaluation, and aggregation logic
  models/
    decision_tree_model.py        Decision tree wrapper
    random_forest_model.py        Random forest wrapper
    tree_factory.py               Tree model factory
    mlp_encoder.py                MLP encoder network
    mlp.py                        MLP classifier
    hybrid_model.py               MLP-embedding plus tree classifier
  trainers/
    mlp_trainer.py                MLP training helper
  utils/
    data/                         Dataset split and preprocessing utilities
    plots/                        Reporting and visualization helpers
    analysis/                     Embedding analysis helpers
    time/                         Timing utilities
```

## Model Design

The project uses a two-stage comparison setup.

### Tree baseline

The tree baseline trains either a decision tree or a random forest directly on preprocessed tabular features. The selected tree implementation is controlled by `tree_type` in the experiment configuration.

Supported values:

- `decision_tree`
- `random_forest`

### MLP baseline

The MLP baseline trains a neural classifier with an encoder and a linear classification head. The encoder produces a fixed-size embedding, and the classifier head maps that embedding to class logits.

Key MLP parameters:

- `hidden_dim`
- `embedding_dim`
- `num_layers`
- `epochs`
- `lr`

### Hybrid model

The hybrid model reuses the trained MLP as a feature extractor. It transforms each input row into an embedding, then trains a tree model on those embeddings.

In code, the pipeline is:

1. Train the MLP classifier.
2. Extract embeddings from the trained MLP encoder.
3. Train the tree model on the extracted embeddings.
4. Evaluate train accuracy, test accuracy, generalization gap, and training time.

## Supported Datasets

Datasets are selected with `dataset_name` in `ComparisonExperimentConfig`.

Currently supported:

- `wine`
- `cancer`
- `moons`

The `wine` and `cancer` datasets come from `sklearn.datasets`. The `moons` dataset is generated with `sklearn.datasets.make_moons`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Unix-like shells, activate the environment with:

```bash
source .venv/bin/activate
```

## Running Experiments

Run the configured experiments from the repository root:

```bash
python -m src.main
```

The entry point loads `EXPERIMENTS` from `src/experiments/experiments_list.py` and runs each configuration with `run_n_experiments`.

## Configuring Experiments

Edit `src/experiments/experiments_list.py` to change the experiment list.

Example:

```python
from src.experiments.experiment_config import (
    ComparisonExperimentConfig,
    TreeConfig,
    MLPConfig,
    HybridConfig,
)

EXPERIMENTS = [
    ComparisonExperimentConfig(
        name="Cancer | Baseline comparison",
        dataset_name="cancer",
        tree_type="decision_tree",
        n_runs=100,
        tree=TreeConfig(max_depth=3),
        mlp=MLPConfig(
            embedding_dim=4,
            hidden_dim=8,
            num_layers=2,
            epochs=60,
            lr=0.002,
        ),
        hybrid=HybridConfig(tree_max_depth=2),
    )
]
```

Important configuration fields:

| Field | Purpose |
| --- | --- |
| `name` | Human-readable experiment name |
| `dataset_name` | Dataset key: `wine`, `cancer`, or `moons` |
| `tree_type` | Tree backend: `decision_tree` or `random_forest` |
| `n_runs` | Number of repeated runs used for summary statistics |
| `test_size` | Test split fraction |
| `random_state` | Base random seed |
| `tree.max_depth` | Depth for the raw-feature tree baseline |
| `tree.n_estimators` | Number of estimators for random forest |
| `mlp.*` | MLP architecture and training parameters |
| `hybrid.tree_max_depth` | Depth for the tree trained on MLP embeddings |

## Reported Metrics

Each experiment reports summary statistics for:

- training accuracy,
- test accuracy,
- generalization gap,
- training time.

For repeated runs, the runner computes means and standard deviations across runs.

The main script also performs an overfitting comparison between the raw tree and the hybrid model by comparing their train-test gaps.

## Visualizations

The plotting utilities can generate diagnostics for:

- train/test accuracy comparison,
- feature importance for tree models,
- MLP structure and feature influence,
- embedding importance comparison between the MLP and hybrid model,
- MLP loss curves across runs.

These plots are produced from `src/main.py` when visualization data is available.

## Extending the Project

To add a new dataset:

1. Add a loader function in `src/utils/data/data_utils.py`.
2. Register the dataset key in `src/experiments/dataset_factory.py`.
3. Add feature names to `get_columns` if plots should show named features.

To add a new tree model:

1. Implement the `TreeModel` interface in `src/models/`.
2. Register the model in `src/models/tree_factory.py`.
3. Add a new `tree_type` value in experiment configurations.
