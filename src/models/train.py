"""
train.py
Treina modelos de classificação (diet_success) e regressão (weight_loss_kg).
Divisão: 70% treino | 15% validação | 15% teste
Otimização de hiperparâmetros com RandomizedSearchCV no conjunto de treino+validação.
"""

import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.metrics import f1_score, r2_score

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── Grids de hiperparâmetros ───────────────────────────────────────────────────
PARAM_GRIDS_CLASS = {
    "LogisticRegression": {
        "C":           [0.01, 0.1, 1, 10, 100],
        "l1_ratio":    [0.0, 0.5, 1.0],
        "solver":      ["saga"],
        "max_iter":    [2000],
    },
    "DecisionTree": {
        "max_depth":        [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "criterion":        ["gini", "entropy"],
    },
    "RandomForest": {
        "n_estimators":     [50, 100, 200],
        "max_depth":        [5, 10, 20, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features":     ["sqrt", "log2"],
    },
    "GradientBoosting": {
        "n_estimators":  [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth":     [3, 5, 7],
        "subsample":     [0.7, 0.8, 1.0],
    },
}

PARAM_GRIDS_REG = {
    "Ridge": {
        "alpha": [0.01, 0.1, 1, 10, 100, 500],
    },
    "DecisionTree": {
        "max_depth":        [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 5, 10, 20],
    },
    "RandomForest": {
        "n_estimators":     [50, 100, 200],
        "max_depth":        [5, 10, 20, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features":     ["sqrt", "log2"],
    },
    "GradientBoosting": {
        "n_estimators":  [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth":     [3, 5, 7],
        "subsample":     [0.7, 0.8, 1.0],
    },
}

# ── Modelos base (sem hiperparâmetros — ponto de partida para o search) ────────
BASE_CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "DecisionTree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest":       RandomForestClassifier(random_state=RANDOM_STATE),
    "GradientBoosting":   GradientBoostingClassifier(random_state=RANDOM_STATE),
}

BASE_REGRESSORS = {
    "Ridge":            Ridge(),
    "DecisionTree":     DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest":     RandomForestRegressor(random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}


def _split_three(X, y, stratify=None):
    """
    Divide em treino (70%) | validação (15%) | teste (15%).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=stratify
    )
    strat_temp = y_temp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=strat_temp
    )
    print(f"   Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _tune(base_model, param_grid, X_train, y_train, scoring, n_iter=20):
    """
    RandomizedSearchCV no conjunto de treino.
    Retorna o melhor estimador já treinado.
    """
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=n_iter,
        cv=5,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


# ── Classificação ──────────────────────────────────────────────────────────────
def train_classifiers(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = _split_three(X, y, stratify=y)

    # Treino+validação juntos para avaliação final após tuning
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    trained = {}
    print("\n🔵 Treinando e otimizando Classificadores...")
    for name, base_model in BASE_CLASSIFIERS.items():
        print(f"\n   🔍 {name} — RandomizedSearchCV...")
        model, best_params, best_cv = _tune(
            base_model, PARAM_GRIDS_CLASS[name], X_train, y_train, scoring="f1"
        )
        # Retreina com treino+validação usando os melhores parâmetros
        model.fit(X_trainval, y_trainval)

        val_f1  = f1_score(y_val,  model.predict(X_val),  zero_division=0)
        test_f1 = f1_score(y_test, model.predict(X_test), zero_division=0)

        print(f"      Melhores params: {best_params}")
        print(f"      CV F1 (treino): {best_cv:.4f} | Val F1: {val_f1:.4f} | Teste F1: {test_f1:.4f}")

        trained[name] = {
            "model":       model,
            "best_params": best_params,
            "cv_f1":       round(best_cv, 4),
            "val_f1":      round(val_f1, 4),
            "X_test":      X_test,
            "y_test":      y_test,
        }
        _save_model(model, f"classifier_{name}.pkl")

    return trained


# ── Regressão ──────────────────────────────────────────────────────────────────
def train_regressors(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = _split_three(X, y)

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    trained = {}
    print("\n🟢 Treinando e otimizando Regressores...")
    for name, base_model in BASE_REGRESSORS.items():
        print(f"\n   🔍 {name} — RandomizedSearchCV...")
        model, best_params, best_cv = _tune(
            base_model, PARAM_GRIDS_REG[name], X_train, y_train,
            scoring="neg_root_mean_squared_error"
        )
        model.fit(X_trainval, y_trainval)

        val_r2  = r2_score(y_val,  model.predict(X_val))
        test_r2 = r2_score(y_test, model.predict(X_test))

        print(f"      Melhores params: {best_params}")
        print(f"      CV RMSE (treino): {abs(best_cv):.4f} | Val R²: {val_r2:.4f} | Teste R²: {test_r2:.4f}")

        trained[name] = {
            "model":       model,
            "best_params": best_params,
            "cv_rmse":     round(abs(best_cv), 4),
            "val_r2":      round(val_r2, 4),
            "X_test":      X_test,
            "y_test":      y_test,
        }
        _save_model(model, f"regressor_{name}.pkl")

    return trained


def _save_model(model, filename: str):
    with open(MODELS_DIR / filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename: str):
    with open(MODELS_DIR / filename, "rb") as f:
        return pickle.load(f)