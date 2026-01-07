import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(X, y, model_tree, X_emb, model_hybrid, title="Porównanie"):
    """
    Rysuje wykresy:
    1. Dane oryginalne
    2. Przestrzeń embeddingów (jak MLP widzi dane)
    3. Granice decyzyjne Hybrydy na embeddingach
    """
    plt.figure(figsize=(18, 5))

    # 1. Oryginalne dane
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=40)
    plt.title("1. Oryginalne Dane (Wejście)")
    plt.xlabel("Cecha 1")
    plt.ylabel("Cecha 2")

    # 2. Embeddingi (Wizualizacja przestrzeni latentnej)
    # Uwaga: To zadziała najlepiej, jeśli embedding_dim=2. Jeśli więcej, używamy PCA/TSNE (tu uproszczone dla 2D)
    plt.subplot(1, 3, 2)
    if X_emb.shape[1] == 2:
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=40)
        plt.title("2. Embeddingi z MLP (Przestrzeń ukryta)")
        plt.xlabel("Emb 1")
        plt.ylabel("Emb 2")
    else:
        plt.text(0.5, 0.5, "Embeddingi > 2D\n(użyj PCA do wizualizacji)",
                 ha='center', va='center')
        plt.title("2. Embeddingi z MLP")

    # 3. Granice decyzji Hybrydy (na embeddingach)
    plt.subplot(1, 3, 3)
    if X_emb.shape[1] == 2:
        x_min, x_max = X_emb[:, 0].min() - 1, X_emb[:, 0].max() + 1
        y_min, y_max = X_emb[:, 1].min() - 1, X_emb[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = model_hybrid.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=40)
        plt.title("3. Decyzja Drzewa na Embeddingach")
    else:
        plt.text(0.5, 0.5, "Wizualizacja granic\ndostępna dla emb_dim=2",
                 ha='center', va='center')

    plt.tight_layout()
    plt.show()