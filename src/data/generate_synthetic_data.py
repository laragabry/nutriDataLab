import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PATIENTS     = 500
N_DIETS        = 6
N_NUTRITIONISTS = 20


# ── 1. Nutricionistas ──────────────────────────────────────────────────────────
def generate_nutritionists() -> pd.DataFrame:
    specialties = ["Esportiva", "Clínica", "Comportamental", "Pediatria", "Geral"]
    df = pd.DataFrame({
        "nutritionist_id":   range(1, N_NUTRITIONISTS + 1),
        "experience_years":  np.random.randint(1, 25, N_NUTRITIONISTS),
        "specialty":         np.random.choice(specialties, N_NUTRITIONISTS),
        "avg_patient_rating": np.round(np.random.uniform(3.0, 5.0, N_NUTRITIONISTS), 1),
    })
    return df


# ── 2. Dietas ──────────────────────────────────────────────────────────────────
def generate_diets() -> pd.DataFrame:
    diet_names = ["Low-Carb", "Mediterrânea", "Vegana", "Cetogênica", "Paleo", "Flexitariana"]
    df = pd.DataFrame({
        "diet_id":              range(1, N_DIETS + 1),
        "diet_name":            diet_names,
        "avg_daily_calories":   [1600, 1800, 1700, 1500, 1750, 1850],
        "carb_pct":             [10, 45, 55, 5, 25, 40],
        "protein_pct":          [35, 20, 15, 30, 35, 25],
        "fat_pct":              [55, 35, 30, 65, 40, 35],
        "difficulty_level":     ["Alta", "Média", "Média", "Alta", "Alta", "Baixa"],
    })
    return df


# ── 3. Pacientes ───────────────────────────────────────────────────────────────
def generate_patients() -> pd.DataFrame:
    genders           = ["M", "F"]
    activity_levels   = ["Sedentário", "Leve", "Moderado", "Ativo", "Muito Ativo"]
    health_conditions = ["Nenhuma", "Diabetes", "Hipertensão", "Obesidade", "Colesterol Alto"]

    ages    = np.random.randint(18, 70, N_PATIENTS)
    heights = np.random.normal(170, 10, N_PATIENTS).round(1)
    weights = np.random.normal(80, 15, N_PATIENTS).round(1)
    bmi     = (weights / (heights / 100) ** 2).round(2)

    df = pd.DataFrame({
        "patient_id":          range(1, N_PATIENTS + 1),
        "age":                 ages,
        "gender":              np.random.choice(genders, N_PATIENTS),
        "height_cm":           heights,
        "weight_kg":           weights,
        "bmi":                 bmi,
        "activity_level":      np.random.choice(activity_levels, N_PATIENTS),
        "health_condition":    np.random.choice(health_conditions, N_PATIENTS),
        "nutritionist_id":     np.random.randint(1, N_NUTRITIONISTS + 1, N_PATIENTS),
        "diet_id":             np.random.randint(1, N_DIETS + 1, N_PATIENTS),
        "diet_duration_weeks": np.random.randint(4, 24, N_PATIENTS),
        "adherence_pct":       np.random.randint(40, 100, N_PATIENTS),
        "initial_motivation":  np.random.randint(1, 11, N_PATIENTS),   # 1-10
        "sleep_hours":         np.round(np.random.uniform(4, 9, N_PATIENTS), 1),
        "water_intake_liters": np.round(np.random.uniform(1.0, 3.5, N_PATIENTS), 1),
    })
    return df


# ── 4. Resultados ──────────────────────────────────────────────────────────────
def generate_results(patients: pd.DataFrame) -> pd.DataFrame:
    weight_loss = (
        patients["adherence_pct"] * 0.08
        + patients["diet_duration_weeks"] * 0.15
        + patients["initial_motivation"] * 0.3
        + patients["activity_level"].map({
            "Sedentário": -1, "Leve": 0, "Moderado": 1, "Ativo": 2, "Muito Ativo": 3
        })
        - patients["age"] * 0.05
        + np.random.normal(0, 1.5, N_PATIENTS)
    ).round(2)
    weight_loss = weight_loss.clip(lower=-2, upper=25)

    # Sucesso: perda >= 5% do peso inicial e aderência >= 70%
    target_loss  = patients["weight_kg"] * 0.05
    diet_success = ((weight_loss >= target_loss) & (patients["adherence_pct"] >= 70)).astype(int)

    df = pd.DataFrame({
        "patient_id":        patients["patient_id"],
        "weight_loss_kg":    weight_loss,
        "final_bmi":         (patients["bmi"] - weight_loss / (patients["height_cm"] / 100) ** 2).round(2),
        "diet_success":      diet_success,          # 0 = insucesso, 1 = sucesso
        "cholesterol_change": np.random.normal(-5, 10, N_PATIENTS).round(1),
        "blood_pressure_change": np.random.normal(-3, 8, N_PATIENTS).round(1),
        "satisfaction_score": np.random.randint(1, 11, N_PATIENTS),
    })
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Gerando dados sintéticos...")

    nutritionists = generate_nutritionists()
    diets         = generate_diets()
    patients      = generate_patients()
    results       = generate_results(patients)

    nutritionists.to_csv(OUTPUT_DIR / "nutritionists.csv", index=False)
    diets.to_csv(OUTPUT_DIR / "diets.csv", index=False)
    patients.to_csv(OUTPUT_DIR / "patients.csv", index=False)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)

    print(f"✅ CSVs salvos em: {OUTPUT_DIR}")
    for name, df in [("nutritionists", nutritionists), ("diets", diets),
                     ("patients", patients), ("results", results)]:
        print(f"   {name}.csv → {df.shape[0]} linhas × {df.shape[1]} colunas")


if __name__ == "__main__":
    main()