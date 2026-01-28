from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

from src.pricing.adjusted_ltv_pricing_simulator import (
    predict_hazards_log,
    expected_remaining_lifetime_months,
)
from src.pricing.risk_adjusted_ltv_optimizer import ltv_mean_std_from_hazards


@dataclass
class CostSpec:
    fixed_cost: float = 5.0
    variable_cost_rate: float = 0.15
    monthly_discount_rate: float = 0.01


def sample_profiles(df: pd.DataFrame, n: int = 200, seed: int = 42) -> List[dict]:
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
    profiles: List[dict],
    price_grid: np.ndarray,
    horizon_months: int,
    cost: CostSpec,
    risk_aversion: float = 0.25,
    logit_shift: float = 0.0,
) -> pd.DataFrame:
    rows = []
    H = int(horizon_months)

    for p in price_grid:
        p = float(p)
        mean_ltvs = []
        e_months = []
        haz_means = []

        for prof in profiles:
            hazards = predict_hazards_log(
                model=model,
                feature_columns=feature_cols,
                base_profile=prof,
                price=p,
                horizon_months=H,
                logit_shift=logit_shift,   #<< calibrated if shift != 0
            )

            mean_ltv_i, _std_i = ltv_mean_std_from_hazards(
                hazards,
                price=p,
                fixed_cost=cost.fixed_cost,
                variable_cost_rate=cost.variable_cost_rate,
                monthly_discount_rate=cost.monthly_discount_rate,
            )

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
            "n_profiles": int(len(profiles)),
        })

    return pd.DataFrame(rows).sort_values("price").reset_index(drop=True)


def argmax_row(df: pd.DataFrame, col: str) -> dict:
    i = int(df[col].idxmax())
    return df.loc[i].to_dict()
