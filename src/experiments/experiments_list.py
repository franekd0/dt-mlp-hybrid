from src.experiments.experiment_config import ExperimentConfig

EXPERIMENTS = [

    # =====================================================
    # A) MOONS noisy — CONST (ograniczona złożoność drzewa)
    # =====================================================

    ExperimentConfig(name="Moons n=0.3 | const d=1 e=1", dataset_name="moons", noise=0.3, embedding_dim=1, tree_max_depth=1),
    ExperimentConfig(name="Moons n=0.3 | const d=2 e=1", dataset_name="moons", noise=0.3, embedding_dim=1, tree_max_depth=2),
    ExperimentConfig(name="Moons n=0.3 | const d=3 e=1", dataset_name="moons", noise=0.3, embedding_dim=1, tree_max_depth=3),

    ExperimentConfig(name="Moons n=0.3 | const d=1 e=2", dataset_name="moons", noise=0.3, embedding_dim=2, tree_max_depth=1),
    ExperimentConfig(name="Moons n=0.3 | const d=2 e=2", dataset_name="moons", noise=0.3, embedding_dim=2, tree_max_depth=2),
    ExperimentConfig(name="Moons n=0.3 | const d=3 e=2", dataset_name="moons", noise=0.3, embedding_dim=2, tree_max_depth=3),

    # =====================================================
    # B) MOONS noisy — FREE (drzewo ma wystarczającą moc)
    # =====================================================

    ExperimentConfig(name="Moons n=0.3 | free d=5 e=2", dataset_name="moons", noise=0.3, embedding_dim=2, tree_max_depth=5),
    ExperimentConfig(name="Moons n=0.3 | free d=7 e=2", dataset_name="moons", noise=0.3, embedding_dim=2, tree_max_depth=7),
    ExperimentConfig(name="Moons n=0.3 | free d=10 e=2", dataset_name="moons", noise=0.3, embedding_dim=2, tree_max_depth=10),

    # =====================================================
    # C) MOONS — wpływ szumu (CONST)
    # =====================================================

    ExperimentConfig(name="Moons n=0.0 | const d=2 e=1", dataset_name="moons", noise=0.0, embedding_dim=1, tree_max_depth=2),
    ExperimentConfig(name="Moons n=0.1 | const d=2 e=1", dataset_name="moons", noise=0.1, embedding_dim=1, tree_max_depth=2),
    ExperimentConfig(name="Moons n=0.2 | const d=2 e=1", dataset_name="moons", noise=0.2, embedding_dim=1, tree_max_depth=2),
    ExperimentConfig(name="Moons n=0.3 | const d=2 e=1", dataset_name="moons", noise=0.3, embedding_dim=1, tree_max_depth=2),
    ExperimentConfig(name="Moons n=0.4 | const d=2 e=1", dataset_name="moons", noise=0.4, embedding_dim=1, tree_max_depth=2),

    # =====================================================
    # D) CIRCLES — CONST vs FREE
    # =====================================================

    ExperimentConfig(name="Circles f=0.4 n=0.0 | const d=2 e=1", dataset_name="circles", factor=0.4, noise=0.0,
                     embedding_dim=1, tree_max_depth=2),
    ExperimentConfig(name="Circles f=0.4 n=0.1 | const d=2 e=1", dataset_name="circles", factor=0.4, noise=0.1,
                     embedding_dim=1, tree_max_depth=2),
    ExperimentConfig(name="Circles f=0.4 n=0.2 | const d=2 e=1", dataset_name="circles", factor=0.4, noise=0.2,
                     embedding_dim=1, tree_max_depth=2),

    ExperimentConfig(name="Circles f=0.4 n=0.2 | free d=5 e=2", dataset_name="circles", factor=0.4, noise=0.2,
                     embedding_dim=2, tree_max_depth=5),
    ExperimentConfig(name="Circles f=0.4 n=0.2 | free d=7 e=2", dataset_name="circles", factor=0.4, noise=0.2,
                     embedding_dim=2, tree_max_depth=7),

    # =====================================================
    # F) CANCER — const vs free
    # =====================================================

    ExperimentConfig(name="Cancer | const d=2 e=2", dataset_name="cancer", embedding_dim=2, tree_max_depth=2),
    ExperimentConfig(name="Cancer | const d=3 e=2", dataset_name="cancer", embedding_dim=2, tree_max_depth=3),

    ExperimentConfig(name="Cancer | free d=5 e=3", dataset_name="cancer", embedding_dim=3, tree_max_depth=5),
    ExperimentConfig(name="Cancer | free d=7 e=3", dataset_name="cancer", embedding_dim=3, tree_max_depth=7),

    # =====================================================
    # G) DIGITS — const vs free
    # =====================================================

    ExperimentConfig(name="Digits | const d=3 e=8", dataset_name="digits", embedding_dim=8, tree_max_depth=3),
    ExperimentConfig(name="Digits | const d=4 e=8", dataset_name="digits", embedding_dim=8, tree_max_depth=4),

    ExperimentConfig(name="Digits | free d=7 e=16", dataset_name="digits", embedding_dim=16, tree_max_depth=7),
    ExperimentConfig(name="Digits | free d=10 e=16", dataset_name="digits", embedding_dim=16, tree_max_depth=10),
]

