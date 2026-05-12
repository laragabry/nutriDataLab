"""
helpers.py
Funções utilitárias gerais do projeto.
"""

import time
import json
import pandas as pd
from pathlib import Path
from functools import wraps

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def timer(func):
    """Decorator que mede o tempo de execução de uma função."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} concluído em {elapsed:.2f}s")
        return result
    return wrapper


def describe_dataset(df: pd.DataFrame, name: str = "dataset") -> dict:
    """Retorna e imprime estatísticas descritivas completas."""
    stats = {
        "shape":          df.shape,
        "missing_total":  int(df.isnull().sum().sum()),
        "missing_pct":    round(df.isnull().mean().mean() * 100, 2),
        "duplicates":     int(df.duplicated().sum()),
        "numeric_cols":   df.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df.select_dtypes(include="object").columns.tolist(),
    }
    print(f"\n{name.upper()}")
    print(f"   Shape: {stats['shape']}")
    print(f"   Nulos: {stats['missing_total']} ({stats['missing_pct']}%)")
    print(f"   Duplicatas: {stats['duplicates']}")
    print(f"   Colunas numéricas: {len(stats['numeric_cols'])}")
    print(f"   Colunas categóricas: {len(stats['categorical_cols'])}")
    return stats


def save_report(content: dict, filename: str = "model_results.json"):
    """Salva dicionário de resultados como JSON."""
    path = REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, default=str, ensure_ascii=False)
    print(f"Relatório salvo: {path}")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converte DataFrame para tabela Markdown."""
    return df.to_markdown(index=False)