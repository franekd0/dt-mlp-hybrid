import time
from src.utils.data_utils import load_moons_dataset, load_circles_dataset
from src.models.decision_tree_model import DecisionTreeModel
from src.models.mlp import MLP
from src.models.hybrid_model import HybridModel
from src.utils.visualization import compare_models_viz, print_results, print_table
from src.training.mlp_training import train_mlp


def main():

    print("--- 1. Preparing date ---")
    dataset = load_circles_dataset(
        noise=0.2,
        factor=0.5,
        n_samples=1500,
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )
    X_train, X_val, X_test, y_train, y_val, y_test = dataset
    print("shapes: ", list(map(lambda x: x.shape, dataset)))

    input_dim = X_train.shape[1]

    print("\n--- 2. Decision Tree ---")
    tree = DecisionTreeModel(max_depth=5, random_state=42)
    _, time_tree = tree.fit(X_train, y_train)
    acc_tree = tree.score(X_test, y_test)
    print_results(time_tree, acc_tree)


    print("\n--- 3. MLP ---")
    model = MLP(input_dim, hidden_dim=16, embedding_dim=2, num_layers=8, num_classes=2)
    _, time_mlp = train_mlp(model, X_train, y_train, epochs=200, lr=0.01, debug=(False, 0))
    acc_mlp = model.score(X_test, y_test)
    print_results(time_mlp, acc_mlp)

    print("\n--- 4. MLP + TREE hybrid ---")
    hybrid = HybridModel(
        input_dim=input_dim,
        num_classes=2,
        embedding_dim=2,
        hidden_dim=16,
        num_layers=3,
        tree_max_depth=3,
        epochs=300
    )
    _, time_hybrid = hybrid.fit(X_train, y_train)
    acc_hybrid = hybrid.score(X_test, y_test)
    print_results(time_hybrid, acc_hybrid)

    print_table(["Decisional Tree", "MLP", "Hybrid"], [acc_tree, acc_mlp, acc_hybrid], [time_tree, time_mlp, time_hybrid])

    # 6. Wizualizacja
    print("\nGenerating plots...")
    compare_models_viz(X_test, y_test, tree, model, hybrid)


if __name__ == "__main__":
    main()