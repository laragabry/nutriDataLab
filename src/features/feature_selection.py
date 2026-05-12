"""
feature_selection.py
Seleciona as features mais relevantes com base na importância do Random Forest.
Compara desempenho do modelo com todas as features vs. features selecionadas.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
THRESHOLD    = 0.01   # features com importância abaixo deste valor são removidas


def get_feature_importance(X: pd.DataFrame, y: pd.Series, task: str = "classification") -> pd.DataFrame:
    """
    Treina um Random Forest simples e extrai a importância de cada feature.
    task: 'classification' ou 'regression'
    """
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    imp = pd.DataFrame({
        "Feature":    X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return imp


def select_features(X: pd.DataFrame, importance_df: pd.DataFrame, threshold: float = THRESHOLD) -> list[str]:
    """Retorna lista de features com importância acima do threshold."""
    selected = importance_df[importance_df["Importance"] >= threshold]["Feature"].tolist()
    removed  = importance_df[importance_df["Importance"] <  threshold]["Feature"].tolist()
    print(f"\n Features selecionadas ({len(selected)}): {selected}")
    print(f"  Features removidas   ({len(removed)}): {removed}")
    return selected


def compare_with_without_features(
    X: pd.DataFrame,
    y_class: pd.Series,
    y_reg: pd.Series,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Treina Random Forest com todas as features e só com as selecionadas.
    Compara F1 (classificação) e R² (regressão).
    """
    results = []

    for label, cols in [("Todas as features", X.columns.tolist()), ("Features selecionadas", selected_features)]:
        X_sub = X[cols]

        # Classificação
        X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_class, test_size=0.3,
                                                    random_state=RANDOM_STATE, stratify=y_class)
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        f1 = f1_score(y_te, clf.predict(X_te), zero_division=0)

        # Regressão
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_sub, y_reg, test_size=0.3,
                                                        random_state=RANDOM_STATE)
        reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        reg.fit(X_tr2, y_tr2)
        r2 = r2_score(y_te2, reg.predict(X_te2))

        results.append({"Configuração": label, "Nº Features": len(cols), "F1": round(f1, 4), "R²": round(r2, 4)})
        print(f"  {label:<25} | Features: {len(cols):>3} | F1: {f1:.4f} | R²: {r2:.4f}")

    return pd.DataFrame(results)


def plot_feature_importance_full(importance_df: pd.DataFrame, threshold: float = THRESHOLD, title: str = ""):
    """
    Gráfico de barras horizontais com todas as features,
    destacando as que ficam abaixo do threshold em vermelho.
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.35)))
    colors = ["#e74c3c" if v < threshold else "#2980b9" for v in importance_df["Importance"]]
    ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color=colors[::-1])
    ax.axvline(threshold, linestyle="--", color="gray", label=f"Threshold = {threshold}")
    ax.set_title(f"Importancia das Features - {title}\nVermelho: abaixo do threshold | Azul: selecionadas")
    ax.set_xlabel("Importância")
    ax.legend()
    plt.tight_layout()
    fname = f"feature_importance_full_{title.lower().replace(' ', '_')}.png"
    fig.savefig(FIGURES_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" Figura salva: {fname}")


def plot_comparison(results_df: pd.DataFrame):
    """Gráfico comparativo com/sem seleção de features."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#3498db", "#2ecc71"]

    for ax, metric, title in zip(axes, ["F1", "R²"], ["F1 - Classificação", "R² - Regressão"]):
        bars = ax.bar(results_df["Configuração"], results_df[metric], color=colors)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_ylim(0, results_df[metric].max() * 1.15)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontweight="bold")
        ax.tick_params(axis="x", rotation=10)

    plt.suptitle("Comparação: Todas as Features vs. Features Selecionadas", fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_selection_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(" Figura salva: feature_selection_comparison.png")


def run_feature_selection(X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series) -> list[str]:
    """
    Função principal - chama tudo em sequência.
    Retorna a lista de features selecionadas.
    """
    print("\n" + "=" * 60)
    print(" SELEÇÃO DE FEATURES")
    print("=" * 60)

    # Importância para classificação
    print("\n[Classificação]")
    imp_class = get_feature_importance(X, y_class, task="classification")
    plot_feature_importance_full(imp_class, title="Classificação")

    # Importância para regressão
    print("\n[Regressão]")
    imp_reg = get_feature_importance(X, y_reg, task="regression")
    plot_feature_importance_full(imp_reg, title="Regressão")

    # Seleciona com base na classificação (target principal do enunciado)
    selected = select_features(X, imp_class)

    # Compara desempenho
    print("\n Comparação de desempenho:")
    comparison = compare_with_without_features(X, y_class, y_reg, selected)
    print("\n", comparison.to_string(index=False))

    plot_comparison(comparison)

    return selected