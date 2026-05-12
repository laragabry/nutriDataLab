import pandas as pd


def _rename_patients(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "sex":                "gender",
        "baseline_weight_kg": "weight_kg",
        "baseline_bmi":       "bmi",
        "motivation_score":   "baseline_motivation", 
    }).drop(columns=["bmi_redundant", "record_created_at"], errors="ignore")


def _rename_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "mean_adherence_pct":        "adherence_pct",
        "weight_change_kg_6m":       "weight_loss_kg",
        "motivation_score_program":  "initial_motivation",
    }).drop(columns=["program_index", "program_id"], errors="ignore")


def _rename_nutritionists(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "years_experience": "experience_years",
        "approach":         "health_condition",  
    }).drop(columns=["experience_years"], errors="ignore")


def _rename_diets(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "diet_type": "difficulty_level",   
    }).drop(columns=["total_macros"], errors="ignore")


def merge_all(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    patients      = _rename_patients(data["patients"].copy())
    outcomes      = _rename_outcomes(data["results"].copy())
    diets         = _rename_diets(data["diets"].copy())
    nutritionists = _rename_nutritionists(data["nutritionists"].copy())

    df = patients.merge(outcomes, on="patient_id", how="inner")

    df = df.merge(diets, on="diet_id", how="left", suffixes=("", "_diet"))

    df = df.merge(nutritionists, on="nutritionist_id", how="left", suffixes=("", "_nutr"))

    if "activity_level" not in df.columns:
        df["activity_level"] = "Moderado"
    if "water_intake_liters" not in df.columns:
        df["water_intake_liters"] = 2.0
    if "avg_patient_rating" not in df.columns:
        df["avg_patient_rating"] = df.get("adherence_ratio", pd.Series(dtype=float))
    if "avg_daily_calories" not in df.columns:
        df["avg_daily_calories"] = (
            df["carb_pct"] * 4 + df["protein_pct"] * 4 + df["fat_pct"] * 9
            if "carb_pct" in df.columns else 2000.0
        )

    if "weight_loss_kg" in df.columns:
        median_loss = df["weight_loss_kg"].median()
        df["diet_success"] = (df["weight_loss_kg"] >= median_loss).astype(int)
        print(f"   Mediana de perda: {median_loss:.2f} kg | "
              f"Sucesso: {df['diet_success'].sum()} | "
              f"Insucesso: {(df['diet_success']==0).sum()}")

    print(f" Dataset consolidado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df