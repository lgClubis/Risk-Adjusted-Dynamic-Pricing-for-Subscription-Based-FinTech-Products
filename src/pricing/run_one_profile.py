from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.pricing.ltv_pricing_simulator import (
    predict_hazards,
    survival_from_hazards,
    expected_remaining_lifetime_months,
    expected_ltv,
    simulate_price_grid,
    argmax_price
)

DATA_PATH = "data/processed/hazard_dataset.parquet"
REPORT_DIR = Path("reports/hazard_logreg_pricing")

def main():
    model = joblib.load(REPORT_DIR / "model.joblib")
    feature_cols = json.loads((REPORT_DIR / "feature_columns.json").read_text(encoding="utf-8"))

    df = pd.read_parquet(DATA_PATH)
    row = df.sample(1, random_state=42).iloc[0].to_dict()
    row.pop("price", None)
    row.pop("log_price", None)

    price = 50.0
    H = 48

    hazards = predict_hazards(model, feature_cols, row, price=price, horizon_months=H)
    S = survival_from_hazards(hazards)
    elt = expected_remaining_lifetime_months(hazards)
    ltv = expected_ltv(hazards, price=price, fixed_cost=5.0, variable_cost_rate=0.15, monthly_discount_rate=0.01)

    print("hazards[:5] =", hazards[:5])
    print("mean/min/max =", float(hazards.mean()), float(hazards.min()), float(hazards.max()))
    print("S[1] =", float(S[1]))
    print("S[6] =", float(S[6]) if len(S) > 6 else None)
    print("S_end =", float(S[-1]))
    print("E[remaining months] =", elt)
    print("LTV(price=50) =", ltv)

    price_grid = np.arange(20, 121, 2)   #reasonable grid
    grid_df = simulate_price_grid(
        model=model,
        feature_columns=feature_cols,
        base_profile=row,
        price_grid=price_grid,
        horizon_months=48,
        fixed_cost=5.0,
        variable_cost_rate=0.15,
        monthly_discount_rate=0.01,
    )

    best = argmax_price(grid_df, col="LTV")
    print("\nBest price by mean LTV:", best)

    #Save artifacts
    out_dir = Path("reports/pricing_simulator_one_profile")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_df.to_csv(out_dir / "ltv_vs_price.csv", index=False)

    #Plot
    plt.figure()
    plt.plot(grid_df["price"], grid_df["LTV"])
    plt.xlabel("Price")
    plt.ylabel("Expected LTV")
    plt.title("LTV vs Price (one profile)")
    plt.tight_layout()
    plt.savefig(out_dir / "ltv_vs_price.png", dpi=160)
    plt.close()

    print(f"\nSaved: {out_dir / 'ltv_vs_price.csv'}")
    print(f"Saved: {out_dir / 'ltv_vs_price.png'}")



if __name__ == "__main__":
    main()
