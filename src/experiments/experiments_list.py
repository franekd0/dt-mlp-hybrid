from src.experiments.experiment_config import ExperimentConfig, TreeConfig, MLPConfig, HybridConfig

EXPERIMENTS = [

    # =========================
    # MOONS
    # =========================

    # ExperimentConfig(
    #     name="Circles | noise=0.1 | emb=2 | tree=3",
    #     dataset_name="circles",
    #     noise=0.1,
    #     factor=0.5,
    #     n_samples=2000,
    #     embedding_dim=2,
    #     tree_max_depth=2,
    #     epochs=150,
    #     lr=0.005
    #
    # ),
    #
    # ExperimentConfig(
    #     name="Moons | noise=0.2 | emb=2 | tree=4",
    #     dataset_name="moons",
    #     noise=0.25,
    #     n_samples=2000,
    #     embedding_dim=2,
    #     tree_max_depth=4,
    #     epochs=100,
    #     lr=0.003
    # ),

    # ExperimentConfig(
    #     name="Digits | emb=8 | tree=6",
    #     dataset_name="digits",
    #     embedding_dim=4,
    #     tree_max_depth=6,
    #     epochs=100,
    #     lr=0.001988
    # ),

    ExperimentConfig(
        name="Wine | emb=4 | tree=3",
        dataset_name="wine",

        tree=TreeConfig(
            max_depth=10
        ),

        mlp=MLPConfig(
            embedding_dim=16,
            hidden_dim=8,
            num_layers=4,
            epochs=40,
            lr=0.003
        ),

        hybrid=HybridConfig(
            embedding_dim=4,
            hidden_dim=8,
            num_layers=4,
            tree_max_depth=4,
            epochs=80,
            lr=0.003
        )
    ),

# ExperimentConfig(
#     name="Adult | emb=10 | tree=5",
#     dataset_name="adult",
#     embedding_dim=10,
#     tree_max_depth=3,
#     epochs=50,
#     lr=0.001
# ),

    # ExperimentConfig(
    #     name="Covertype | Hybrid Tuned",
    #     dataset_name="covertype",
    #     n_samples=100000,
    #
    #     tree=TreeConfig(
    #         max_depth=10,
    #         criterion="gini"
    #     ),
    #
    #     mlp=MLPConfig(
    #         embedding_dim=16,
    #         hidden_dim=64,
    #         num_layers=2,
    #         epochs=150,
    #         lr=0.001
    #     ),
    #
    #     hybrid=HybridConfig(
    #         embedding_dim=32,
    #         hidden_dim=16,
    #         num_layers=2,
    #         tree_max_depth=15,
    #         epochs=150,
    #         lr=0.001
    #     )
    # )
]

