"""
preprocess.py
Limpeza, encoding, normalização e tratamento de outliers.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Colunas que não entram no modelo
DROP_COLS = ["patient_id", "diet_id", "nutritionist_id"]

# Colunas numéricas para normalização (apenas as que existem nos novos dados)
SCALE_COLS = [
    "age", "height_cm", "weight_kg", "bmi",
    "adherence_pct", "initial_motivation", "sleep_hours",
    "water_intake_liters", "experience_years", "avg_patient_rating",
    "avg_daily_calories", "carb_pct", "protein_pct", "fat_pct",
    "sodium_limit_mg", "fiber_target_g", "adherence_ratio",
    "baseline_motivation",
]


def remove_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= df[col].between(q1 - 3 * iqr, q3 + 3 * iqr)
    removed = (~mask).sum()
    if removed:
        print(f"Outliers removidos: {removed} linhas")
    return df[mask].reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    cols_present = [c for c in SCALE_COLS if c in df.columns]
    scaler = StandardScaler()
    df[cols_present] = scaler.fit_transform(df[cols_present])
    return df, scaler


def preprocess(df_raw: pd.DataFrame, scale: bool = True) -> dict:
    df = df_raw.copy()

    # 1. Remove IDs
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # 2. Trata nulos
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Remove outliers
    df = remove_outliers_iqr(df, df.select_dtypes(include="number").columns.tolist())

    # 4. Separa targets
    y_class = df.pop("diet_success")
    y_reg   = df.pop("weight_loss_kg")

    # 5. Remove leakage (colunas que não existem mais nos novos dados, mas por segurança)
    leakage_cols = ["final_bmi", "satisfaction_score"]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

    # 6. Encoding
    df, encoders = encode_categoricals(df)

    # 7. Normalização
    scaler = None
    if scale:
        df, scaler = scale_features(df)

    df_out = df.copy()
    df_out["diet_success"]   = y_class.values
    df_out["weight_loss_kg"] = y_reg.values
    df_out.to_csv(PROCESSED_DIR / "dataset_processed.csv", index=False)
    print(f"Dados processados salvos em: {PROCESSED_DIR / 'dataset_processed.csv'}")

    return {
        "df":       df,
        "encoders": encoders,
        "scaler":   scaler,
        "X":        df,
        "y_class":  y_class,
        "y_reg":    y_reg,
    }