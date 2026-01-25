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
)
from src.pricing.risk_adjusted_ltv_optimizer import (
    simulate_price_grid_risk,
    argmax_row

)
from src.pricing.calibration import estimate_logit_shift
from src.pricing.adjusted_ltv_pricing_simulator import (
    predict_hazards_log,
    simulate_price_grid
)

DATA_PATH = "data/processed/hazard_dataset.parquet"
REPORT_DIR = Path("reports/hazard_logreg_pricing")


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("month")
    months = df["month"].drop_duplicates().sort_values()
    cut_idx = int((1 - test_frac) * len(months))
    cut_month = months.iloc[cut_idx]
    train = df[df["month"] < cut_month].copy()
    test = df[df["month"] >= cut_month].copy()
    return train, test, cut_month



def main():
    model = joblib.load(REPORT_DIR / "model.joblib")
    feature_cols = json.loads((REPORT_DIR / "feature_columns.json").read_text(encoding="utf-8"))

        #CALIBRATION: estimate logit shift on the same time-split test set as training
    df_all = pd.read_parquet(DATA_PATH)
    df_all["month"] = pd.to_datetime(df_all["month"])

    train_df, test_df, cut_month = time_split(df_all, test_frac=0.2)

    target = "y_churn"
    X_test = test_df[feature_cols]
    y_test = test_df[target].astype(int)

    p_test = model.predict_proba(X_test)[:, 1]
    target_mean = float(y_test.mean())

    shift = estimate_logit_shift(p_test, target_mean=target_mean)

    #Debug prints
    p_clip = np.clip(p_test, 1e-12, 1 - 1e-12)
    mean_shifted = float(np.mean(1/(1+np.exp(-(np.log(p_clip/(1-p_clip)) + shift)))))

    print("Cut month:", str(pd.Timestamp(cut_month).date()))
    print("Target mean hazard (event rate test):", target_mean)
    print("Estimated logit shift:", shift)
    print("Mean raw p_test:", float(np.mean(p_test)))
    print("Mean shifted p_test:", mean_shifted)



    df = pd.read_parquet(DATA_PATH)
    row = df.sample(1, random_state=42).iloc[0].to_dict()
    row.pop("price", None)
    row.pop("log_price", None)

    price = 50.0
    H = 48

    hazards = predict_hazards_log(model, feature_cols, row, price=price, horizon_months=H, logit_shift=shift)
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
    haz_uncal = predict_hazards_log(model, feature_cols, row, price=price, horizon_months=H, logit_shift=0.0)
    haz_cal   = predict_hazards_log(model, feature_cols, row, price=price, horizon_months=H, logit_shift=shift)
    print("mean hazard uncal:", float(haz_uncal.mean()))
    print("mean hazard cal  :", float(haz_cal.mean()))
    hazards = haz_cal


    # -----------------------------
    # A/B: uncalibrated vs calibrated
    # -----------------------------
    base_out = Path("reports/adjusted_p_simulator_one_profile_calibration")
    base_out.mkdir(parents=True, exist_ok=True)

    price_grid = np.arange(20, 121, 2)
    lambdas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    lam_for_plot = 0.25

    variants = {
        "uncalibrated": 0.0,
        "calibrated": shift,
    }

    for name, logit_shift in variants.items():
        out_dir = base_out / name
        out_dir.mkdir(parents=True, exist_ok=True)

        #Lambda sweep (per variant)
        summary = []
        for lam in lambdas:
            df_lam = simulate_price_grid(
                model=model,
                feature_columns=feature_cols,
                base_profile=row,
                price_grid=price_grid,
                horizon_months=48,
                fixed_cost=5.0,
                variable_cost_rate=0.15,
                monthly_discount_rate=0.01,
                risk_aversion=lam,
                logit_shift=logit_shift,   
            )
            best_mean = argmax_row(df_lam, col="mean_ltv")
            best_ra = argmax_row(df_lam, col="risk_adj_ltv")
            summary.append({
                "lambda": lam,
                "best_mean_price": best_mean["price"],
                "best_ra_price": best_ra["price"],
                "best_ra_value": best_ra["risk_adj_ltv"],
                "best_ra_std": best_ra["std_ltv"],
                "best_ra_mean": best_ra["mean_ltv"],
            })

        summary_df = pd.DataFrame(summary)
        print(f"\nLambda sweep ({name}):\n", summary_df)
        summary_df.to_csv(out_dir / "lambda_sweep.csv", index=False)

        #Save a reference grid + plot for lam_for_plot
        grid_df = simulate_price_grid(
            model=model,
            feature_columns=feature_cols,
            base_profile=row,
            price_grid=price_grid,
            horizon_months=48,
            fixed_cost=5.0,
            variable_cost_rate=0.15,
            monthly_discount_rate=0.01,
            risk_aversion=lam_for_plot,
            logit_shift=logit_shift,   
        )
        print("Calibrated: LTV at min price:", float(grid_df["mean_ltv"].iloc[0]), "price", float(grid_df["price"].iloc[0]))
        print("Calibrated: LTV at max price:", float(grid_df["mean_ltv"].iloc[-1]), "price", float(grid_df["price"].iloc[-1]))

        #check monotonicity
        mono_mean = bool((grid_df["mean_ltv"].diff().dropna() >= -1e-9).all())
        mono_ra = bool((grid_df["risk_adj_ltv"].diff().dropna() >= -1e-9).all())
        print("mean_ltv monotone non-decreasing?", mono_mean)
        print("risk_adj_ltv monotone non-decreasing?", mono_ra)


        grid_df.to_csv(out_dir / "ltv_vs_price.csv", index=False)

        plt.figure()
        plt.plot(grid_df["price"], grid_df["mean_ltv"], label="mean_ltv")
        plt.plot(grid_df["price"], grid_df["risk_adj_ltv"], label=f"risk_adj_ltv (λ={lam_for_plot})")
        plt.xlabel("Price")
        plt.ylabel("LTV")
        plt.title(f"Mean vs Risk-Adjusted LTV ({name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "ltv_vs_price.png", dpi=160)
        plt.close()

        best_ra_plot = argmax_row(grid_df, col="risk_adj_ltv")
        print(f"\nBest price by risk-adjusted LTV ({name}, λ={lam_for_plot}):", best_ra_plot)
        print(f"Saved: {out_dir / 'ltv_vs_price.csv'}")
        print(f"Saved: {out_dir / 'ltv_vs_price.png'}")



if __name__ == "__main__":
    main()
