from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.pricing.calibration import estimate_logit_shift
from src.pricing.adjusted_ltv_pricing_simulator import predict_hazards_log, expected_remaining_lifetime_months, survival_from_hazards
from src.pricing.risk_adjusted_ltv_optimizer import ltv_mean_std_from_hazards


DATA_PATH = "data/processed/hazard_dataset.parquet"
MODEL_DIR = Path("reports/hazard_logreg_pricing")


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("month")
    months = df["month"].drop_duplicates().sort_values()
    cut_idx = int((1 - test_frac) * len(months))
    cut_month = months.iloc[cut_idx]
    train = df[df["month"] < cut_month].copy()
    test = df[df["month"] >= cut_month].copy()
    return train, test, cut_month


def sample_profiles(df: pd.DataFrame, n: int = 200, seed: int = 42) -> list[dict]:
    samp = df.sample(n=min(n, len(df)), random_state=seed)
    profiles = []
    for _, r in samp.iterrows():
        d = r.to_dict()
        d.pop("price", None)
        d.pop("log_price", None)
        profiles.append(d)
    return profiles


def simulate_population_grid(
    model,
    feature_cols: list[str],
    profiles: list[dict],
    price_grid: np.ndarray,
    horizon_months: int,
    fixed_cost: float,
    variable_cost_rate: float,
    monthly_discount_rate: float,
    risk_aversion: float,
    logit_shift: float,
) -> pd.DataFrame:
    rows = []
    H = int(horizon_months)

    for p in price_grid:
        p = float(p)
        mean_ltvs = []
        e_months = []
        haz_means = []
        S6_list = []


        for prof in profiles:
            hazards = predict_hazards_log(
                model=model,
                feature_columns=feature_cols,
                base_profile=prof,
                price=p,
                horizon_months=H,
                logit_shift=logit_shift,
            )
            S = survival_from_hazards(hazards)
            S6_list.append(float(S[6]) if len(S) > 6 else float(S[-1]))

            mean_ltv_i, _std_i = ltv_mean_std_from_hazards(
                hazards,
                price=p,
                fixed_cost=fixed_cost,
                variable_cost_rate=variable_cost_rate,
                monthly_discount_rate=monthly_discount_rate,
            )
            S = survival_from_hazards(hazards)
            S6_list.append(float(S[6]) if len(S) > 6 else float(S[-1]))


            mean_ltvs.append(mean_ltv_i)
            e_months.append(expected_remaining_lifetime_months(hazards))
            haz_means.append(float(hazards.mean()))

        mean_ltvs = np.asarray(mean_ltvs, dtype=float)

        pop_mean = float(mean_ltvs.mean())
        pop_std = float(mean_ltvs.std(ddof=1)) if len(mean_ltvs) > 1 else 0.0
        pop_ra = pop_mean - float(risk_aversion) * pop_std

        rows.append({
            "price": p,
            "pop_mean_ltv": pop_mean,
            "pop_std_ltv": pop_std,
            "pop_risk_adj_ltv": pop_ra,
            "avg_E_months": float(np.mean(e_months)),
            "avg_hazard_mean": float(np.mean(haz_means)),
            "avg_S6": float(np.mean(S6_list)),
            "n_profiles": int(len(profiles)),
        })

    return pd.DataFrame(rows).sort_values("price").reset_index(drop=True)


def argmax_row(df: pd.DataFrame, col: str) -> dict:
    i = int(df[col].idxmax())
    return df.loc[i].to_dict()


def main():
    model = joblib.load(MODEL_DIR / "model.joblib")
    feature_cols = json.loads((MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))

    df = pd.read_parquet(DATA_PATH)
    df["month"] = pd.to_datetime(df["month"])

    #estimate calibration shift on time-split test set
    train_df, test_df, cut_month = time_split(df, test_frac=0.2)

    target = "y_churn"
    X_test = test_df[feature_cols]
    y_test = test_df[target].astype(int)

    p_test = model.predict_proba(X_test)[:, 1]
    shift = estimate_logit_shift(p_test, target_mean=float(y_test.mean()))

    print("Cut month:", str(pd.Timestamp(cut_month).date()))
    print("Target mean hazard (event rate test):", float(y_test.mean()))
    print("Estimated logit shift:", shift)

    #population sample
    profiles = sample_profiles(df, n=200, seed=42)
    print("n_profiles:", len(profiles))

    #policy settings
    price_grid = np.arange(10, 401, 5)
    H = 48
    lam = 0.25

    fixed_cost = 5.0
    variable_cost_rate = 0.15
    monthly_discount_rate = 0.01

    out_dir = Path("reports/pricing_policy_population_calibrated")
    out_dir.mkdir(parents=True, exist_ok=True)

    pop_df = simulate_population_grid(
        model=model,
        feature_cols=feature_cols,
        profiles=profiles,
        price_grid=price_grid,
        horizon_months=H,
        fixed_cost=fixed_cost,
        variable_cost_rate=variable_cost_rate,
        monthly_discount_rate=monthly_discount_rate,
        risk_aversion=lam,
        logit_shift=shift,
    )

    best_mean = argmax_row(pop_df, "pop_mean_ltv")
    best_ra = argmax_row(pop_df, "pop_risk_adj_ltv")

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    frontier = []

    for thr in thresholds:
        feasible = pop_df[pop_df["avg_S6"] >= thr]
        if len(feasible) == 0:
            frontier.append({"S6_threshold": thr, "feasible": False})
            continue
        best = feasible.loc[feasible["pop_risk_adj_ltv"].idxmax()].to_dict()
        frontier.append({
            "S6_threshold": thr,
            "feasible": True,
            "best_price": best["price"],
            "best_pop_mean_ltv": best["pop_mean_ltv"],
            "best_pop_risk_adj_ltv": best["pop_risk_adj_ltv"],
            "avg_S6": best["avg_S6"],
            "avg_E_months": best["avg_E_months"],
            "avg_hazard_mean": best["avg_hazard_mean"],
        })

    frontier_df = pd.DataFrame(frontier)
    print("\nRetention frontier:\n", frontier_df)
    frontier_df.to_csv(out_dir / "retention_frontier_S6.csv", index=False)





    print("\nBest price (population mean):", best_mean)
    print(f"Best price (population risk-adjusted, λ={lam}):", best_ra)

    pop_df.to_csv(out_dir / "population_ltv_vs_price.csv", index=False)
    
    fd = frontier_df[frontier_df["feasible"] == True].copy()

    plt.figure()
    plt.plot(fd["S6_threshold"], fd["best_price"])
    plt.xlabel("S6 retention threshold")
    plt.ylabel("Optimal price (risk-adjusted)")
    plt.title("Retention constraint tightens → optimal price decreases")
    plt.tight_layout()
    plt.savefig(out_dir / "frontier_price_vs_S6.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(fd["S6_threshold"], fd["best_pop_risk_adj_ltv"])
    plt.xlabel("S6 retention threshold")
    plt.ylabel("Best feasible risk-adjusted LTV")
    plt.title("Retention constraint tightens → value decreases")
    plt.tight_layout()
    plt.savefig(out_dir / "frontier_value_vs_S6.png", dpi=160)
    plt.close()
    """
    plt.figure()
    plt.plot(pop_df["price"], pop_df["pop_mean_ltv"], label="pop_mean_ltv")
    plt.plot(pop_df["price"], pop_df["pop_risk_adj_ltv"], label=f"pop_risk_adj_ltv (λ={lam})")
    plt.xlabel("Price")
    plt.ylabel("Population LTV")
    plt.title("Population Mean vs Risk-Adjusted LTV (Calibrated)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "population_ltv_vs_price.png", dpi=160)
    plt.close()
    """
    print(f"\nSaved: {out_dir / 'population_ltv_vs_price.csv'}")
    print(f"Saved: {out_dir / 'population_ltv_vs_price.png'}")


if __name__ == "__main__":
    main()
