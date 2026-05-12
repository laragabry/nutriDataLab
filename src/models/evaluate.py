"""
evaluate.py
Avalia e compara modelos de classificação e regressão.
Mostra métricas de validação (durante tuning) e teste (avaliação final).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, mean_squared_error, mean_absolute_error, r2_score,
)


# ?? Classificação ??????????????????????????????????????????????????????????????
def evaluate_classifiers(trained_models: dict) -> pd.DataFrame:
    rows = []
    print("\nAvaliacao Final -- Classificacao (diet_success)\n" + "-" * 60)
    for name, info in trained_models.items():
        model, X_test, y_test = info["model"], info["X_test"], info["y_test"]
        y_pred = model.predict(X_test)
        row = {
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1_teste":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "F1_val":    info.get("val_f1", "-"),
            "CV_F1":     info.get("cv_f1", "-"),
        }
        rows.append(row)
        print(f"\n  {name}")
        print(f"  CV F1: {row['CV_F1']} | Val F1: {row['F1_val']} | Teste F1: {row['F1_teste']}")
        print(classification_report(y_test, y_pred, target_names=["Insucesso", "Sucesso"]))

    results = pd.DataFrame(rows).sort_values("F1_teste", ascending=False).reset_index(drop=True)
    print("\nRanking Classificação (por F1 no teste):")
    print(results.to_string(index=False))
    return results


# ?? Regressão ??????????????????????????????????????????????????????????????????
def evaluate_regressors(trained_models: dict) -> pd.DataFrame:
    rows = []
    print("\nAvaliacao Final -- Regressao (weight_loss_kg)\n" + "-" * 60)
    for name, info in trained_models.items():
        model, X_test, y_test = info["model"], info["X_test"], info["y_test"]
        y_pred = model.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        row = {
            "Model":    name,
            "RMSE":     round(rmse, 4),
            "MAE":      round(mean_absolute_error(y_test, y_pred), 4),
            "R2_teste": round(r2_score(y_test, y_pred), 4),
            "R2_val":   info.get("val_r2", "-"),
            "CV_RMSE":  info.get("cv_rmse", "-"),
        }
        rows.append(row)
        print(f"  {name:<22} RMSE={rmse:.3f} | MAE={row['MAE']:.3f} | "
              f"Val R²={row['R2_val']} | Teste R²={row['R2_teste']}")

    results = pd.DataFrame(rows).sort_values("R2_teste", ascending=False).reset_index(drop=True)
    print("\nRanking Regressão (por R² no teste):")
    print(results.to_string(index=False))
    return results


def feature_importance(model, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        print("  Modelo não suporta feature_importances_")
        return pd.DataFrame()
    imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    return imp