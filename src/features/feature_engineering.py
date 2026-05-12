"""
feature_engineering.py
Criação de features derivadas para enriquecer o dataset.
Adaptado para os novos CSVs do professor.
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ?? IMC categórico (baseline_bmi -> renomeado para bmi no merge) ????????????
    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 24.9, 29.9, 40, 999],
            labels=["Abaixo do Peso", "Normal", "Sobrepeso", "Obeso I", "Obeso II+"],
        ).astype(str)

    # ?? Score de estilo de vida (sem activity_level real -> usa adherence_pct) ??
    # activity_level foi preenchido como "Moderado" no merge; contribui com 50 fixo
    activity_map = {"Sedentário": 0, "Leve": 25, "Moderado": 50, "Ativo": 75, "Muito Ativo": 100}
    activity_score = df["activity_level"].map(activity_map).fillna(50) if "activity_level" in df.columns else 50

    sleep_score = 0
    if "sleep_hours" in df.columns:
        sleep_score = (df["sleep_hours"].clip(6, 9) - 6) / 3 * 100

    water_score = 0
    if "water_intake_liters" in df.columns:
        water_score = (df["water_intake_liters"].clip(1.5, 3) - 1.5) / 1.5 * 100

    adherence_val = df["adherence_pct"] if "adherence_pct" in df.columns else 50

    df["lifestyle_score"] = (
        activity_score * 0.4
        + adherence_val * 0.4
        + sleep_score   * 0.1
        + water_score   * 0.1
    ).round(2)

    # ?? Perda esperada por semana (proxy: 0.5% do peso atual) ?????????????????
    if "weight_kg" in df.columns:
        df["kg_per_week_expected"] = (df["weight_kg"] * 0.005).round(3)

    # ?? Faixa etária ??????????????????????????????????????????????????????????
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "55+"],
        ).astype(str)

    # ?? Interação: aderência x motivação ?????????????????????????????????????
    if "adherence_pct" in df.columns and "initial_motivation" in df.columns:
        df["adherence_x_motivation"] = (
            df["adherence_pct"] * df["initial_motivation"] / 100
        ).round(2)

    # ?? Balanço de macros (desequilíbrio em relação a 33/33/34) ??????????????
    if all(c in df.columns for c in ["carb_pct", "protein_pct", "fat_pct"]):
        df["macro_balance_score"] = (
            abs(df["carb_pct"] - 33)
            + abs(df["protein_pct"] - 33)
            + abs(df["fat_pct"] - 34)
        ).round(2)

    # ?? Fumante como risco (binário já está em smoker) ????????????????????????
    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].astype(int) if df["smoker"].dtype == bool else df["smoker"]

    print(f"Features engenheiradas adicionadas. Shape: {df.shape}")
    return df