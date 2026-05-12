"""
predict.py
Inferência com modelos treinados. Carrega o melhor modelo e faz previsões.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_diet_success(patient_features: pd.DataFrame, model_name: str = "RandomForest") -> np.ndarray:
    """
    Prevê sucesso da dieta (0/1) para um ou mais pacientes.

    Parâmetros
    ----------
    patient_features : DataFrame com as mesmas colunas usadas no treino
    model_name       : nome do modelo (sem prefixo/sufixo)

    Retorna
    -------
    Array de previsões (0 = insucesso, 1 = sucesso)
    """
    model  = load_model(f"classifier_{model_name}.pkl")
    preds  = model.predict(patient_features)
    probas = model.predict_proba(patient_features)[:, 1] if hasattr(model, "predict_proba") else None
    return preds, probas


def predict_weight_loss(patient_features: pd.DataFrame, model_name: str = "RandomForest") -> np.ndarray:
    """
    Prevê perda de peso (kg) para um ou mais pacientes.
    """
    model = load_model(f"regressor_{model_name}.pkl")
    return model.predict(patient_features)


# ?? Exemplo de uso ?????????????????????????????????????????????????????????????
if __name__ == "__main__":
    print("Carregando dados processados para demonstração...")
    processed_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "dataset_processed.csv"
    df = pd.read_csv(processed_path)

    target_cols = ["diet_success", "weight_loss_kg"]
    X = df.drop(columns=target_cols)

    sample = X.sample(5, random_state=0)

    preds_class, probas = predict_diet_success(sample)
    preds_reg = predict_weight_loss(sample)

    result = sample.copy()
    result["pred_success"]     = preds_class
    result["proba_success"]    = probas.round(3) if probas is not None else None
    result["pred_weight_loss"] = preds_reg.round(2)

    print("\nPrevisões para 5 pacientes amostrados:")
    print(result[["pred_success", "proba_success", "pred_weight_loss"]].to_string())