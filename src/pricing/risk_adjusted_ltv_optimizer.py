from __future__ import annotations

import numpy as np
import pandas as pd

from src.pricing.ltv_pricing_simulator import (
    predict_hazards,
    survival_from_hazards,
    expected_remaining_lifetime_months,
)

def churn_pmf_from_hazards(hazards: np.ndarray):
    H = len(hazards)
    S = survival_from_hazards(hazards)
    pmf = np.empty(H + 1, dtype=float)
    pmf[:H] = S[:-1] * hazards
    pmf[H] = S[H]
    pmf = pmf / pmf.sum()
    T_vals = np.arange(1, H + 2, dtype=int)
    return T_vals, pmf

def discounted_cashflow_given_T(
    T: int,
    margin: float,
    horizon_months: int,
    monthly_discount_rate: float = 0.0,
) -> float:
    H = horizon_months
    months_earned = H if T >= H + 1 else T
    if months_earned <= 0:
        return 0.0
    if monthly_discount_rate > 0:
        df = 1.0 / np.power(1.0 + monthly_discount_rate, np.arange(months_earned, dtype=float))
        return float(margin * df.sum())
    return float(margin * months_earned)

def ltv_mean_std_from_hazards(
    hazards: np.ndarray,
    price: float,
    fixed_cost: float = 5.0,
    variable_cost_rate: float = 0.15,
    monthly_discount_rate: float = 0.01,
) -> tuple[float, float]:
    H = len(hazards)
    margin = (1.0 - variable_cost_rate) * float(price) - float(fixed_cost)

    T_vals, pmf = churn_pmf_from_hazards(hazards)
    cf = np.array(
        [discounted_cashflow_given_T(int(T), margin, H, monthly_discount_rate) for T in T_vals],
        dtype=float,
    )

    mean = float(np.sum(pmf * cf))
    var = float(np.sum(pmf * (cf - mean) ** 2))
    std = float(np.sqrt(max(var, 0.0)))
    return mean, std

def simulate_price_grid_risk(
    model,
    feature_columns,
    base_profile: dict,
    price_grid: np.ndarray,
    horizon_months: int = 48,
    fixed_cost: float = 5.0,
    variable_cost_rate: float = 0.15,
    monthly_discount_rate: float = 0.01,
    risk_aversion: float = 0.25,
) -> pd.DataFrame:
    rows = []
    for p in price_grid:
        p = float(p)
        hazards = predict_hazards(model, feature_columns, base_profile, price=p, horizon_months=horizon_months)

        mean_ltv, std_ltv = ltv_mean_std_from_hazards(
            hazards, price=p,
            fixed_cost=fixed_cost,
            variable_cost_rate=variable_cost_rate,
            monthly_discount_rate=monthly_discount_rate,
        )
        ra_ltv = mean_ltv - risk_aversion * std_ltv

        elt = expected_remaining_lifetime_months(hazards)
        margin = (1.0 - variable_cost_rate) * p - float(fixed_cost)

        rows.append({
            "price": p,
            "margin": float(margin),
            "E_months": float(elt),
            "mean_ltv": float(mean_ltv),
            "std_ltv": float(std_ltv),
            "risk_adj_ltv": float(ra_ltv),
            "hazard_mean": float(hazards.mean()),
        })

    return pd.DataFrame(rows).sort_values("price").reset_index(drop=True)

def argmax_price(df: pd.DataFrame, col: str = "mean_ltv") -> dict:
    i = int(df[col].idxmax())
    return df.loc[i].to_dict()
