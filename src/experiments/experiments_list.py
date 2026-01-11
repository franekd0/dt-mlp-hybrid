from src.experiments.experiment_config import ExperimentConfig

EXPERIMENTS = [

    ExperimentConfig(
        name="best wine",
        dataset_name="wine",
        embedding_dim=4,
        tree_max_depth=5,
    ),

    ExperimentConfig(
        name="cancer",
        dataset_name="cancer",
        embedding_dim=8,
        tree_max_depth=5,
    ),

    ExperimentConfig(
        name="Wine | emb=4 | tree=5",
        dataset_name="wine",
        embedding_dim=4,
        tree_max_depth=3,
        lr=0.01
    ),
    ExperimentConfig(
        name="Wine | emb=4 | tree=5",
        dataset_name="wine",
        embedding_dim=8,
        tree_max_depth=3,
        lr=0.001
    ),
    ExperimentConfig(
        name="Statlog Heart | hybrid-dominant",
        dataset_name="heart",

        # --- DATA ---
        test_size=0.2,
        val_size=0.2,
        n_runs=50,

        # --- TREE ---
        tree_max_depth=2,

        # --- MLP ---
        embedding_dim=3,
        hidden_dim=3,
        num_layers=2,
        epochs=80,
        lr=0.005
    ),

    ExperimentConfig(
        name="Heart | emb=4 | tree=5",
        dataset_name="heart",
        embedding_dim=3,
        tree_max_depth=3,
        # lr=0.008,
        n_runs=1
    ),
    ExperimentConfig(
        name="Overfit test | Tree should overfit",
        dataset_name="heart",
        n_runs=10,
        random_state=42,

        tree_max_depth=10,
        embedding_dim=2,
        hidden_dim=32,
        num_layers=2,
        epochs=50,
        lr=0.01
    ),
ExperimentConfig(
        name=f"Adult | emb={4} | tree={2} | lr={0.005}",
        dataset_name="adult",
        embedding_dim=4,
        tree_max_depth=4,
        hidden_dim=8,
        num_layers=3,
        epochs=40,
        lr=0.005,
        n_runs=5
    )
]