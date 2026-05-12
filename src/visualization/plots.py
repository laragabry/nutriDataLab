"""
plots.py
Funções de visualização para EDA e resultados de modelos.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figura salva: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# TARGETS
# ══════════════════════════════════════════════════════════════════════════════

def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts = df["diet_success"].value_counts()
    axes[0].bar(["Insucesso", "Sucesso"], counts.values, color=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Distribuição: Sucesso da Dieta")
    axes[0].set_ylabel("Quantidade")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    axes[1].hist(df["weight_loss_kg"], bins=30, color="#3498db", edgecolor="white")
    axes[1].set_title("Distribuição: Perda de Peso (kg)")
    axes[1].set_xlabel("kg perdidos")
    axes[1].set_ylabel("Frequência")

    _save(fig, "target_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# CORRELAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Mapa de Correlação — Todas as Variáveis Numéricas", fontsize=14)
    _save(fig, "correlation_heatmap.png")


def plot_correlation_with_target(df: pd.DataFrame):
    """Correlação de cada variável numérica com diet_success e weight_loss_kg."""
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()[["diet_success", "weight_loss_kg"]].drop(
        ["diet_success", "weight_loss_kg"], errors="ignore"
    ).sort_values("diet_success", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, col, color, title in zip(
        axes,
        ["diet_success", "weight_loss_kg"],
        ["#2ecc71", "#3498db"],
        ["Correlação com Sucesso da Dieta", "Correlação com Perda de Peso (kg)"],
    ):
        vals = corr[col].sort_values(ascending=True)
        colors = ["#e74c3c" if v < 0 else color for v in vals]
        ax.barh(vals.index, vals.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Correlação de Pearson")
    plt.tight_layout()
    _save(fig, "correlation_with_targets.png")


# ══════════════════════════════════════════════════════════════════════════════
# SEXO / GÉNERO
# ══════════════════════════════════════════════════════════════════════════════

def plot_gender_vs_success(df: pd.DataFrame):
    """Taxa de sucesso e perda média de peso por sexo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mapeia valores numéricos de volta se já foi encoded
    gender_col = df["gender"].copy()
    if gender_col.dtype != object:
        gender_col = gender_col.map({0: "Feminino", 1: "Masculino"}).fillna(gender_col.astype(str))

    tmp = df.copy()
    tmp["gender_label"] = gender_col

    success_rate = tmp.groupby("gender_label")["diet_success"].mean() * 100
    axes[0].bar(success_rate.index, success_rate.values, color=["#e91e8c", "#1e90ff"])
    axes[0].set_title("Taxa de Sucesso por Sexo (%)")
    axes[0].set_ylabel("% Sucesso")
    axes[0].set_ylim(0, 110)
    for i, (label, val) in enumerate(success_rate.items()):
        axes[0].text(i, val + 1, f"{val:.1f}%", ha="center", fontweight="bold")

    weight_mean = tmp.groupby("gender_label")["weight_loss_kg"].mean()
    axes[1].bar(weight_mean.index, weight_mean.values, color=["#e91e8c", "#1e90ff"])
    axes[1].set_title("Perda Média de Peso por Sexo (kg)")
    axes[1].set_ylabel("kg perdidos")
    for i, (label, val) in enumerate(weight_mean.items()):
        axes[1].text(i, val + 0.1, f"{val:.2f} kg", ha="center", fontweight="bold")

    plt.tight_layout()
    _save(fig, "gender_vs_success.png")


# ══════════════════════════════════════════════════════════════════════════════
# TIPO DE DIETA
# ══════════════════════════════════════════════════════════════════════════════

def plot_diet_vs_success(df: pd.DataFrame):
    """Taxa de sucesso e perda média por tipo/nome de dieta."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    col = "diet_name" if "diet_name" in df.columns else "difficulty_level"

    success_rate = df.groupby(col)["diet_success"].mean().sort_values(ascending=False) * 100
    colors = sns.color_palette("Blues_r", len(success_rate))
    bars = axes[0].barh(success_rate.index, success_rate.values, color=colors)
    axes[0].set_title("Taxa de Sucesso por Tipo de Dieta (%)")
    axes[0].set_xlabel("% Sucesso")
    for bar, val in zip(bars, success_rate.values):
        axes[0].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center")

    weight_mean = df.groupby(col)["weight_loss_kg"].mean().sort_values(ascending=False)
    colors2 = sns.color_palette("Greens_r", len(weight_mean))
    bars2 = axes[1].barh(weight_mean.index, weight_mean.values, color=colors2)
    axes[1].set_title("Perda Média de Peso por Tipo de Dieta (kg)")
    axes[1].set_xlabel("kg perdidos")
    for bar, val in zip(bars2, weight_mean.values):
        axes[1].text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center")

    plt.tight_layout()
    _save(fig, "diet_vs_success.png")


# ══════════════════════════════════════════════════════════════════════════════
# PERFIL DO NUTRICIONISTA
# ══════════════════════════════════════════════════════════════════════════════

def plot_nutritionist_vs_outcome(df: pd.DataFrame):
    """Impacto do perfil do nutricionista no sucesso e perda de peso."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Abordagem (approach → health_condition após rename)
    col = "health_condition" if "health_condition" in df.columns else None

    if col:
        success_rate = df.groupby(col)["diet_success"].mean().sort_values(ascending=False) * 100
        axes[0].bar(success_rate.index.astype(str), success_rate.values,
                    color=sns.color_palette("Oranges_r", len(success_rate)))
        axes[0].set_title("Taxa de Sucesso por Abordagem do Nutricionista (%)")
        axes[0].set_ylabel("% Sucesso")
        axes[0].tick_params(axis="x", rotation=30)
        for i, val in enumerate(success_rate.values):
            axes[0].text(i, val + 0.5, f"{val:.1f}%", ha="center", fontsize=9)

    # Experiência do nutricionista vs perda de peso
    if "experience_years" in df.columns:
        axes[1].scatter(df["experience_years"], df["weight_loss_kg"],
                        alpha=0.3, color="#e67e22", s=15)
        # linha de tendência
        tmp = df[["experience_years", "weight_loss_kg"]].dropna()
        if len(tmp) > 1:
            m, b = np.polyfit(tmp["experience_years"], tmp["weight_loss_kg"], 1)
            xs = np.linspace(tmp["experience_years"].min(), tmp["experience_years"].max(), 100)
            axes[1].plot(xs, m * xs + b, color="red", linewidth=1.5,
                         label=f"Tendência: {m:.2f}x+{b:.2f}")
            axes[1].legend()
        axes[1].set_title("Experiência do Nutricionista vs Perda de Peso")
        axes[1].set_xlabel("Anos de Experiência")
        axes[1].set_ylabel("Perda de Peso (kg)")

    plt.tight_layout()
    _save(fig, "nutritionist_vs_outcome.png")


def plot_specialty_vs_success(df: pd.DataFrame):
    """Taxa de sucesso por especialidade do nutricionista."""
    if "specialty" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    rate = df.groupby("specialty")["diet_success"].mean().sort_values(ascending=True) * 100
    colors = ["#e74c3c" if v < 50 else "#2ecc71" for v in rate.values]
    ax.barh(rate.index.astype(str), rate.values, color=colors)
    ax.axvline(50, linestyle="--", color="gray", label="50%")
    ax.set_title("Taxa de Sucesso por Especialidade do Nutricionista (%)")
    ax.set_xlabel("% Sucesso")
    ax.legend()
    for i, val in enumerate(rate.values):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, "specialty_vs_success.png")


# ══════════════════════════════════════════════════════════════════════════════
# IDADE / ADERÊNCIA / PESO
# ══════════════════════════════════════════════════════════════════════════════

def plot_age_vs_weight_loss(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    tmp = df[["age", "weight_loss_kg"]].dropna()
    ax.scatter(tmp["age"], tmp["weight_loss_kg"], alpha=0.4, color="#8e44ad", s=20)
    m, b = np.polyfit(tmp["age"], tmp["weight_loss_kg"], 1)
    ax.plot(tmp["age"].sort_values(), m * tmp["age"].sort_values() + b,
            color="red", linewidth=1.5, label=f"Tendência: y={m:.2f}x+{b:.2f}")
    ax.set_xlabel("Idade")
    ax.set_ylabel("Perda de Peso (kg)")
    ax.set_title("Idade vs Perda de Peso")
    ax.legend()
    _save(fig, "age_vs_weight_loss.png")


def plot_age_group_vs_success(df: pd.DataFrame):
    """Taxa de sucesso e perda média por faixa etária."""
    if "age_group" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ["18-25", "26-35", "36-45", "46-55", "55+"]
    present = [o for o in order if o in df["age_group"].values]

    rate = df.groupby("age_group")["diet_success"].mean().reindex(present) * 100
    axes[0].bar(rate.index, rate.values, color=sns.color_palette("coolwarm", len(rate)))
    axes[0].set_title("Taxa de Sucesso por Faixa Etária (%)")
    axes[0].set_ylabel("% Sucesso")
    for i, val in enumerate(rate.values):
        axes[0].text(i, val + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")

    weight = df.groupby("age_group")["weight_loss_kg"].mean().reindex(present)
    axes[1].bar(weight.index, weight.values, color=sns.color_palette("coolwarm", len(weight)))
    axes[1].set_title("Perda Média de Peso por Faixa Etária (kg)")
    axes[1].set_ylabel("kg perdidos")
    for i, val in enumerate(weight.values):
        axes[1].text(i, val + 0.1, f"{val:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    _save(fig, "age_group_vs_success.png")


def plot_adherence_vs_success(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, name in [(0, "#e74c3c", "Insucesso"), (1, "#2ecc71", "Sucesso")]:
        subset = df[df["diet_success"] == label]["adherence_pct"]
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=name)
    ax.set_xlabel("Aderência (%)")
    ax.set_ylabel("Frequência")
    ax.set_title("Aderência à Dieta vs Sucesso")
    ax.legend()
    _save(fig, "adherence_vs_success.png")


def plot_motivation_vs_success(df: pd.DataFrame):
    """Boxplot de motivação inicial por resultado da dieta."""
    if "initial_motivation" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        df[df["diet_success"] == 0]["initial_motivation"].dropna(),
        df[df["diet_success"] == 1]["initial_motivation"].dropna(),
    ]
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="#f0f0f0"),
                    medianprops=dict(color="red", linewidth=2))
    bp["boxes"][0].set_facecolor("#e74c3c")
    bp["boxes"][1].set_facecolor("#2ecc71")
    ax.set_xticklabels(["Insucesso", "Sucesso"])
    ax.set_title("Motivação Inicial vs Sucesso da Dieta")
    ax.set_ylabel("Score de Motivação")
    _save(fig, "motivation_vs_success.png")


def plot_bmi_vs_weight_loss(df: pd.DataFrame):
    """IMC inicial vs perda de peso, colorido por sucesso."""
    if "bmi" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color, name in [(0, "#e74c3c", "Insucesso"), (1, "#2ecc71", "Sucesso")]:
        sub = df[df["diet_success"] == label]
        ax.scatter(sub["bmi"], sub["weight_loss_kg"], alpha=0.4, color=color, s=20, label=name)
    ax.set_xlabel("IMC Inicial")
    ax.set_ylabel("Perda de Peso (kg)")
    ax.set_title("IMC Inicial vs Perda de Peso")
    ax.legend()
    _save(fig, "bmi_vs_weight_loss.png")


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def plot_clustering_comparison(X: pd.DataFrame, labels_kmeans: np.ndarray, labels_hier: np.ndarray):
    """
    Compara K-Means vs Clustering Hierárquico lado a lado em PCA 2D.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_

    sil_km   = silhouette_score(X, labels_kmeans)
    sil_hier = silhouette_score(X, labels_hier)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, labels, title, sil in zip(
        axes,
        [labels_kmeans, labels_hier],
        ["K-Means", "Clustering Hierárquico"],
        [sil_km, sil_hier],
    ):
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=25)
        plt.colorbar(sc, ax=ax, label="Cluster")
        ax.set_title(f"{title}\nSilhouette = {sil:.3f}")
        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")

    plt.suptitle("Comparação de Métodos de Clustering (PCA 2D)", fontsize=13)
    plt.tight_layout()
    _save(fig, "clustering_comparison.png")
    print(f"   K-Means Silhouette: {sil_km:.3f} | Hierárquico Silhouette: {sil_hier:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_outliers_boxplot(df: pd.DataFrame):
    """
    Boxplots das principais variáveis numéricas para identificar outliers
    visualmente ANTES da remoção pelo IQR.
    """
    cols = [c for c in [
        "age", "weight_kg", "bmi", "height_cm",
        "adherence_pct", "initial_motivation", "sleep_hours",
        "experience_years", "weight_loss_kg",
        "carb_pct", "protein_pct", "fat_pct",
    ] if c in df.columns]

    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr    = q3 - q1
        n_out  = ((data < q1 - 3 * iqr) | (data > q3 + 3 * iqr)).sum()

        axes[i].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#aed6f1"),
                        medianprops=dict(color="red", linewidth=2),
                        flierprops=dict(marker="o", color="#e74c3c", alpha=0.5, markersize=4))
        axes[i].set_title(f"{col}\n({n_out} outliers extremos)", fontsize=9)
        axes[i].set_ylabel(col, fontsize=8)

    # Esconde eixos vazios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots — Identificação de Outliers (antes da remoção)", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "outliers_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature",
                ax=ax, palette="Blues_r", legend=False)
    ax.set_title(title)
    _save(fig, f"feature_importance_{title.lower().replace(' ', '_')}.png")


def plot_model_comparison(results_df: pd.DataFrame, metric: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("muted", len(results_df))
    bars = ax.bar(results_df["Model"], results_df[metric], color=colors)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_ylim(0, results_df[metric].max() * 1.15)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=15)
    _save(fig, f"model_comparison_{metric.lower()}.png")