from __future__ import annotations

import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

DATA_PATH = "data/processed/hazard_dataset.parquet"
REPORT_DIR = Path("reports/hazard_logreg_pricing")

# Importiere deine Funktionen:
# from src.pricing.ltv_pricing_simulator import predict_hazards
# (oder wenn du sie noch lokal hast: einfach hier reinkopieren)

def build_rows_over_horizon(base_profile: dict, price: float, horizon_months: int) -> pd.DataFrame:
    rows = []
    for k in range(horizon_months):
        row = dict(base_profile)
        if "tenure_months" in row:
            row["tenure_months"] = float(row["tenure_months"]) + k
        row["price"] = float(price)
        row["log_price"] = float(np.log(max(price, 1e-12)))
        rows.append(row)
    return pd.DataFrame(rows)

def predict_hazards(model, feature_columns, base_profile: dict, price: float, horizon_months: int) -> np.ndarray:
    X = build_rows_over_horizon(base_profile, price, horizon_months)
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X = X.loc[:, feature_columns]
    hazards = model.predict_proba(X)[:, 1].astype(float)
    return np.clip(hazards, 1e-6, 1 - 1e-6)

def main():
    model = joblib.load(REPORT_DIR / "model.joblib")
    feature_cols = json.loads((REPORT_DIR / "feature_columns.json").read_text(encoding="utf-8"))

    df = pd.read_parquet(DATA_PATH)

    # Zieh eine Zeile als "Profil"
    row = df.sample(1, random_state=42).iloc[0].to_dict()

    # Wichtig: die Simulator-Logik injiziert price/log_price selbst
    row.pop("price", None)
    row.pop("log_price", None)

    hazards = predict_hazards(
        model=model,
        feature_columns=feature_cols,
        base_profile=row,
        price=50.0,
        horizon_months=48,
    )

    print("hazards[:5] =", hazards[:5])
    print("mean/min/max =", float(hazards.mean()), float(hazards.min()), float(hazards.max()))

if __name__ == "__main__":
    main()
