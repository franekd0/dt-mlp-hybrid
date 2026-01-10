from src.experiments.experiment_config import ExperimentConfig

EXPERIMENTS = [

    # =========================
    # MOONS
    # =========================

    ExperimentConfig(
        name="Moons | noise=0.1 | emb=1 | tree=2",
        dataset_name="circles",
        noise=0.4,
        n_samples=5000,
        embedding_dim=2,

    ),

    ExperimentConfig(
        name="Moons | noise=0.1 | emb=2 | tree=3",
        dataset_name="circles",
        noise=0.4,
        n_samples=1500,
        embedding_dim=2,
    ),

    ExperimentConfig(
        name="Moons | noise=0.3 | emb=2 | tree=4",
        dataset_name="circles",
        noise=0.4,
        n_samples=1500,
        embedding_dim=2,
    ),

    ExperimentConfig(
        name="Moons | noise=0.3 | emb=3 | tree=5",
        dataset_name="circles",
        noise=0.4,
        n_samples=1500,
        embedding_dim=2,
    ),
]

