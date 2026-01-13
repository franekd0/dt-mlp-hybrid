from src.experiments.experiment_config import ComparisonExperimentConfig, TreeConfig, MLPConfig, HybridConfig

EXPERIMENTS=[
    ComparisonExperimentConfig(
        name="Heart | Baseline comparison",
        dataset_name="cancer",

        tree=TreeConfig(
            max_depth=3
        ),

        mlp=MLPConfig(
            embedding_dim=4,
            hidden_dim=8,
            num_layers=2,
            epochs=60,
            lr=0.002
        ),

        hybrid=HybridConfig(
            tree_max_depth=2
        )
    )
]

BASE_CFG=ComparisonExperimentConfig(
        name="Heart",
        dataset_name="cancer",

        mlp=MLPConfig(
            embedding_dim=2,
            hidden_dim=4,
            num_layers=2,
            epochs=80,
            lr=0.003
        ),

        hybrid=HybridConfig(
            tree_max_depth=5,
        )
)