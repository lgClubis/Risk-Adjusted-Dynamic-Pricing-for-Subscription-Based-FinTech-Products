from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.pricing.ltv_pricing_simulator import ( #was used prior
    predict_hazards,
    survival_from_hazards,
    expected_remaining_lifetime_months,
    expected_ltv,
    simulate_price_grid,
    argmax_price
)
from src.pricing.risk_adjusted_ltv_optimizer import (
    churn_pmf_from_hazards,
    discounted_cashflow_given_T,
    ltv_mean_std_from_hazards,
    simulate_price_grid_risk,
    argmax_row

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

    out_dir = Path("reports/adjusted_p_simulator_one_profile")
    price_grid = np.arange(20, 121, 2)   #reasonable grid
    lambdas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    summary = []

    for lam in lambdas:
        grid_df = simulate_price_grid_risk(
            model=model,
            feature_columns=feature_cols,
            base_profile=row,
            price_grid=price_grid,
            horizon_months=48,
            fixed_cost=5.0,
            variable_cost_rate=0.15,
            monthly_discount_rate=0.01,
            risk_aversion=lam,
        )
        best_mean = argmax_row(grid_df, col="mean_ltv")
        best_ra = argmax_row(grid_df, col="risk_adj_ltv")
        summary.append({
            "lambda": lam,
            "best_mean_price": best_mean["price"],
            "best_ra_price": best_ra["price"],
            "best_ra_value": best_ra["risk_adj_ltv"],
            "best_ra_std": best_ra["std_ltv"],
            "best_ra_mean": best_ra["mean_ltv"],
        })

    summary_df = pd.DataFrame(summary)
    print("\nLambda sweep:\n", summary_df)
    summary_df.to_csv(out_dir / "lambda_sweep.csv", index=False)

    print("\nBest price by mean LTV:", best_mean)

    lam_for_plot = 0.25
    grid_df = simulate_price_grid_risk(
        model=model,
        feature_columns=feature_cols,
        base_profile=row,
        price_grid=price_grid,
        horizon_months=48,
        fixed_cost=5.0,
        variable_cost_rate=0.15,
        monthly_discount_rate=0.01,
        risk_aversion=lam_for_plot,
    )

    """grid_df = simulate_price_grid(       #old not risk adjusted
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
    """

    #Save artifacts
    out_dir = Path("reports/adjusted_p_simulator_one_profile")

    grid_df.to_csv(out_dir / "ltv_vs_price.csv", index=False)

    plt.figure()
    plt.plot(grid_df["price"], grid_df["mean_ltv"], label="mean_ltv")
    plt.plot(grid_df["price"], grid_df["risk_adj_ltv"], label=f"risk_adj_ltv (λ={lam_for_plot})")
    plt.xlabel("Price")
    plt.ylabel("LTV")
    plt.title("Mean vs Risk-Adjusted LTV (one profile)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ltv_vs_price.png", dpi=160)
    plt.close()

    """
    #Old plot
    plt.figure()
    plt.plot(grid_df["price"], grid_df["LTV"])
    plt.xlabel("Price")
    plt.ylabel("Expected LTV")
    plt.title("LTV vs Price (one profile)")
    plt.tight_layout()
    plt.savefig(out_dir / "ltv_vs_price.png", dpi=160)
    plt.close()"""

    print(f"\nSaved: {out_dir / 'ltv_vs_price.csv'}")
    print(f"Saved: {out_dir / 'ltv_vs_price.png'}")



if __name__ == "__main__":
    main()
