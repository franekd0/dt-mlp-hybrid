from src.experiments.experiment_config import ExperimentConfig, TreeConfig, MLPConfig, HybridConfig

EXPERIMENT_NAME = "wine"

# EXPERIMENTS = [
#
#     ExperimentConfig(
#         name="Wine | Baseline (Tree vs MLP vs Hybrid)",
#         dataset_name=EXPERIMENT_NAME,
#
#         tree=TreeConfig(
#             max_depth=8
#         ),
#
#         mlp=MLPConfig(
#             embedding_dim=16,
#             hidden_dim=32,
#             num_layers=4,
#             epochs=40,
#             lr=0.002
#         ),
#
#         hybrid=HybridConfig(
#             embedding_dim=16,
#             hidden_dim=32,
#             num_layers=4,
#             tree_max_depth=5,
#             epochs=40,
#             lr=0.002
#         )
#     ),
#
#     ExperimentConfig(
#         name="Wine | Stability (Hybrid vs MLP)",
#         dataset_name=EXPERIMENT_NAME,
#
#         tree=TreeConfig(
#             max_depth=8
#         ),
#
#         mlp=MLPConfig(
#             embedding_dim=16,
#             hidden_dim=8,
#             num_layers=4,
#             epochs=40,
#             lr=0.002
#         ),
#
#         hybrid=HybridConfig(
#             embedding_dim=16,
#             hidden_dim=8,
#             num_layers=4,
#             tree_max_depth=3,
#             epochs=40,
#             lr=0.002
#         )
#     ),
#
#     ExperimentConfig(
#         name="Wine | Hybrid dominance (small emb)",
#         dataset_name=EXPERIMENT_NAME,
#
#         tree=TreeConfig(
#             max_depth=8
#         ),
#
#         mlp=MLPConfig(
#             embedding_dim=4,
#             hidden_dim=8,
#             num_layers=3,
#             epochs=40,
#             lr=0.003
#         ),
#
#         hybrid=HybridConfig(
#             embedding_dim=4,
#             hidden_dim=8,
#             num_layers=3,
#             tree_max_depth=4,
#             epochs=40,
#             lr=0.003
#         )
#     ),
# ]

EXPERIMENTS=[
    ExperimentConfig(
        name="Heart | Baseline (Tree vs MLP vs Hybrid)",
        dataset_name="heart",

        tree=TreeConfig(
            max_depth=6
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
        )
    ),
    ExperimentConfig(
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
    ExperimentConfig(
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