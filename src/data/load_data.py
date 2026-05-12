"""
load_data.py
Carrega os 4 CSVs brutos e retorna DataFrames com validação básica.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Nome real dos arquivos no disco -> chave interna usada no pipeline
EXPECTED_FILES = {
    "patients":      "patients.csv",
    "diets":         "diets.csv",
    "nutritionists": "nutritionists.csv",
    "results":       "outcomes.csv",
}


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    data = {}
    for key, filename in EXPECTED_FILES.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        df = pd.read_csv(filepath)
        print(f"[OK] {filename}: {df.shape[0]} linhas x {df.shape[1]} colunas")
        data[key] = df
    return data


def quick_info(data: dict[str, pd.DataFrame]) -> None:
    for name, df in data.items():
        print(f"\n{'?'*40}\n {name.upper()}")
        print(df.dtypes.to_string())
        print(f"Nulos: {df.isnull().sum().sum()}")