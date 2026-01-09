import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils.data_utils import load_moons_dataset, load_circles_dataset
from src.models.decision_tree_model import DecisionTreeModel
from src.models.mlp import MLP
from src.models.hybrid_model import HybridModel
from src.utils.visualization import compare_models_viz


def train_standalone_mlp(X, y, input_dim, hidden_dim, classes, epochs=200):
    model = MLP(input_dim, hidden_dim, embedding_dim=2, num_layers=3, num_classes=classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
    return model


def evaluate_standalone_mlp(model, X, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(X))
        preds = torch.argmax(logits, dim=1).numpy()
    return (preds == y).mean()


def main():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("--- 1. Przygotowanie danych ---")
    X_train, X_val, X_test, y_train, y_val, y_test = load_circles_dataset(
        noise=0.2,
        factor=0.5,
        n_samples=1500,
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )

    input_dim = X_train.shape[1]

    print("\n--- 2. Trenowanie Drzewa Decyzyjnego (Baseline) ---")
    start = time.time()
    tree = DecisionTreeModel(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    time_tree = time.time() - start
    acc_tree = tree.score(X_test, y_test)
    print(f"Drzewo -> Acc: {acc_tree:.4f} | Czas: {time_tree:.4f}s")

    print("\n--- 3. Trenowanie samego MLP ---")
    start = time.time()
    mlp = train_standalone_mlp(X_train, y_train, input_dim, hidden_dim=16, classes=2, epochs=300)
    time_mlp = time.time() - start
    acc_mlp = evaluate_standalone_mlp(mlp, X_test, y_test)
    print(f"MLP    -> Acc: {acc_mlp:.4f} | Czas: {time_mlp:.4f}s")

    print("\n--- 4. Trenowanie Hybrydy ---")
    start = time.time()
    hybrid = HybridModel(
        input_dim=input_dim,
        num_classes=2,
        embedding_dim=2,
        hidden_dim=16,
        num_layers=3,
        tree_max_depth=3,
        epochs=300
    )
    hybrid.fit(X_train, y_train)
    time_hybrid = time.time() - start
    acc_hybrid = hybrid.score(X_test, y_test)
    print(f"Hybryda -> Acc: {acc_hybrid:.4f} | Czas: {time_hybrid:.4f}s")

    print("\n" + "=" * 50)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Czas [s]':<10}")
    print("-" * 50)
    print(f"{'Drzewo (DT)':<20} | {acc_tree:.4f}     | {time_tree:.4f}")
    print(f"{'Sieć (MLP)':<20} | {acc_mlp:.4f}     | {time_mlp:.4f}")
    print(f"{'Hybryda (MLP+DT)':<20} | {acc_hybrid:.4f}     | {time_hybrid:.4f}")
    print("=" * 50)

    # 6. Wizualizacja
    print("\nGenerowanie wykresów porównawczych...")
    compare_models_viz(X_test, y_test, tree, mlp, hybrid)


if __name__ == "__main__":
    main()