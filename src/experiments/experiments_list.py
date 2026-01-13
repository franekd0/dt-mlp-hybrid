from src.experiments.experiment_config import ComparisonExperimentConfig, TreeConfig, MLPConfig, HybridConfig, \
    SweepSpecConfig

EXPERIMENTS=[
    ComparisonExperimentConfig(
        name="Heart | Baseline (Tree vs MLP vs Hybrid)",
        dataset_name="heart",
        tree_type="random_forest",

        tree=TreeConfig(
            max_depth=6,
            n_estimators=10
        ),

        mlp=MLPConfig(
            embedding_dim=16,
            hidden_dim=32,
            num_layers=3,
            epochs=60,
            lr=0.002
        ),

        hybrid=HybridConfig(
            tree_max_depth=4,
            n_estimators=30
        )
    ),
    ComparisonExperimentConfig(
        name="Heart | Hybrid dominance (small emb)",
        dataset_name="heart",

        tree=TreeConfig(
            max_depth=6
        ),

        mlp=MLPConfig(
            embedding_dim=4,
            hidden_dim=8,
            num_layers=3,
            epochs=60,
            lr=0.003
        ),

        hybrid=HybridConfig(
            tree_max_depth=4,
        )
    ),
    ComparisonExperimentConfig(
        name="Heart | Hybrid dominance (regularized)",
        dataset_name="heart",

        tree=TreeConfig(
            max_depth=6
        ),

        mlp=MLPConfig(
            embedding_dim=4,
            hidden_dim=8,
            num_layers=4,
            epochs=80,
            lr=0.003
        ),

        hybrid=HybridConfig(
            tree_max_depth=3,
        )
    )
]

BASE_CFG=ComparisonExperimentConfig(
        name="Heart | Hybrid dominance (regularized)",
        dataset_name="heart",

        tree=TreeConfig(
            max_depth=6
        ),

        mlp=MLPConfig(
            embedding_dim=4,
            hidden_dim=8,
            num_layers=4,
            epochs=80,
            lr=0.003
        ),

        hybrid=HybridConfig(
            tree_max_depth=3,
        )
    )

EXPERIMENTS_SWEEPS = [
    SweepSpecConfig(
        name="Heart | Hybrid tree depth sweep",
        base_config=BASE_CFG,
        param_path="hybrid.tree_max_depth",
        values=[1, 2, 3, 4, 5, 6, 8, 10]
    ),
    SweepSpecConfig(
        name="Heart | Embedding dim sweep",
        base_config=BASE_CFG,
        param_path="mlp.embedding_dim",
        values=[2, 4, 8, 16, 32]
    ),
]
