import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
def find_optimal_k(X: pd.DataFrame, k_range: range = range(2, 10)) -> int:
    """
    Elbow Method + Silhouette Score para encontrar o K ideal.
    Salva gráfico em reports/figures/.
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    best_k = k_range[np.argmax(silhouettes)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_range, inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inércia")

    axes[1].plot(k_range, silhouettes, marker="s", color="green")
    axes[1].axvline(best_k, linestyle="--", color="red", label=f"Melhor k={best_k}")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clustering_elbow_silhouette.png", dpi=120)
    plt.close()
    print(f" Gráfico salvo. Melhor k = {best_k} (Silhouette = {max(silhouettes):.3f})")
    return best_k


def run_kmeans(X: pd.DataFrame, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    print(f" K-Means (k={k}) - Distribuição: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return labels


def run_hierarchical(X: pd.DataFrame, k: int) -> np.ndarray:
    hc = AgglomerativeClustering(n_clusters=k)
    labels = hc.fit_predict(X)
    print(f" Hierárquico (k={k}) - Distribuição: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return labels

def plot_clusters_pca(X: pd.DataFrame, labels: np.ndarray, title: str = "Clusters"):
    pca   = PCA(n_components=2)
    X_2d  = pca.fit_transform(X)
    var   = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=30)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"{title}\n(PCA - {var[0]:.1%} + {var[1]:.1%} variância explicada)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + "_pca.png"
    plt.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close()
    print(f" Gráfico PCA salvo: {filename}")

def cluster_profile(df_original: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df_original.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster").mean(numeric_only=True).round(2)
    print("\n Perfil dos Clusters:")
    print(profile.T.to_string())
    return profile