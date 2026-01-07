import time
import numpy as np
import torch

from src.utils.data_utils import load_moons_dataset
from src.models.mlp import MLP
from src.models.decision_tree_model import DecisionTreeModel
from src.utils.trainer import train_mlp, get_embeddings, evaluate_mlp
from src.utils.visualization import plot_comparison


def main():
    print("--- 1. Ładowanie Danych (Make Moons) ---")
    X_train, X_val, X_test, y_train, y_val, y_test = load_moons_dataset(
        noise=0.3,
        n_samples=1000,
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )

    INPUT_DIM = X_train.shape[1]
    HIDDEN_DIM = 16
    OUTPUT_DIM = 2
    NUM_CLASSES = 2
    NUM_LAYERS = 3

    print("\n--- 2. Trenowanie Baseline: Czyste Drzewo (na surowych danych) ---")
    start_time = time.time()
    tree_baseline = DecisionTreeModel(max_depth=5, random_state=42)
    tree_baseline.fit(X_train, y_train)
    tree_time = time.time() - start_time

    acc_tree = tree_baseline.score(X_test, y_test)
    print(f"Drzewo (Baseline) Accuracy: {acc_tree:.4f} (Czas: {tree_time:.4f}s)")

    print("\n--- 3. Trenowanie MLP (Encoder) ---")
    mlp_model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, NUM_CLASSES)

    start_time = time.time()
    train_mlp(mlp_model, X_train, y_train, epochs=300, lr=0.01)
    mlp_time = time.time() - start_time

    acc_mlp = evaluate_mlp(mlp_model, X_test, y_test)
    print(f"Samo MLP Accuracy: {acc_mlp:.4f} (Czas: {mlp_time:.4f}s)")

    print("\n--- 4. Budowanie Hybrydy (MLP Embeddings -> Drzewo) ---")

    start_time = time.time()
    X_train_emb = get_embeddings(mlp_model, X_train)
    X_test_emb = get_embeddings(mlp_model, X_test)

    hybrid_tree = DecisionTreeModel(max_depth=5, random_state=42)
    hybrid_tree.fit(X_train_emb, y_train)
    hybrid_time = (time.time() - start_time) + mlp_time  # Czas MLP + Czas Drzewa

    acc_hybrid = hybrid_tree.score(X_test_emb, y_test)
    print(f"Hybryda Accuracy: {acc_hybrid:.4f} (Łączny czas: {hybrid_time:.4f}s)")

    print("\n" + "=" * 40)
    print(f"{'Model':<20} | {'Acc':<10} | {'Czas [s]':<10}")
    print("-" * 40)
    print(f"{'Drzewo (Raw)':<20} | {acc_tree:.4f}     | {tree_time:.4f}")
    print(f"{'MLP (Softmax)':<20} | {acc_mlp:.4f}     | {mlp_time:.4f}")
    print(f"{'Hybryda (MLP+DT)':<20} | {acc_hybrid:.4f}     | {hybrid_time:.4f}")
    print("=" * 40)

    print("\nGenerowanie wykresów...")
    plot_comparison(X_test, y_test, tree_baseline, X_test_emb, hybrid_tree)


if __name__ == "__main__":
    main()