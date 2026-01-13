import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embeddings_pca(embeddings, y, title):
    pca = PCA(n_components=2, svd_solver="full")
    emb_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=y, cmap="coolwarm",
        edgecolors="k", s=60, alpha=0.8
    )

    plt.title(
        f"{title}\n"
        f"PC1: {pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2: {pca.explained_variance_ratio_[1]:.1%}"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
