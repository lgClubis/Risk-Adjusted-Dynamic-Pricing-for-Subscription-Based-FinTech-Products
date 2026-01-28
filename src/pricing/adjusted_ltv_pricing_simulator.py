import numpy as np
import pandas as pd
from  src.pricing.risk_adjusted_ltv_optimizer import ltv_mean_std_from_hazards
from src.pricing.calibration import apply_logit_shift


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

def predict_hazards_log(model, feature_columns, base_profile: dict, price: float, horizon_months: int, logit_shift: float = 0.0) -> np.ndarray:
    X = build_rows_over_horizon(base_profile, price, horizon_months)
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X = X.loc[:, feature_columns]
    hazards = model.predict_proba(X)[:, 1].astype(float)
    if logit_shift != 0.0:
        hazards = apply_logit_shift(hazards, logit_shift)
    return np.clip(hazards, 1e-6, 1 - 1e-6)

def survival_from_hazards(hazards: np.ndarray) -> np.ndarray:
    """
    hazards[t] = P(churn in month t+1 | alive at start of that month)
    S[0] = 1
    S[t] = prod_{i=0..t-1} (1 - hazards[i])  for t>=1
    """
    H = len(hazards)
    S = np.empty(H + 1, dtype=float)
    S[0] = 1.0
    if H > 0:
        S[1:] = np.cumprod(1.0 - hazards)
    return S

def expected_remaining_lifetime_months(hazards: np.ndarray) -> float:
    """
    Expected number of paid months under 'active at start of month' convention:
      E[months] = sum_{t=1..H} P(alive at start of month t) = sum S[t-1]
    """
    S = survival_from_hazards(hazards)
    return float(np.sum(S[:-1]))

def expected_ltv(
    hazards: np.ndarray,
    price: float,
    fixed_cost: float = 5.0,
    variable_cost_rate: float = 0.15,
    monthly_discount_rate: float = 0.01,
) -> float:
    """
    LTV = sum_{t=1..H} margin(price) * S[t-1] * df[t-1]
    """
    H = len(hazards)
    S = survival_from_hazards(hazards)

    margin = (1.0 - variable_cost_rate) * float(price) - float(fixed_cost)

    if monthly_discount_rate > 0:
        df = 1.0 / np.power(1.0 + monthly_discount_rate, np.arange(H, dtype=float))
    else:
        df = np.ones(H, dtype=float)

    return float(np.sum(margin * S[:-1] * df))


def simulate_price_grid(
    model,
    feature_columns,
    base_profile: dict,
    price_grid: np.ndarray,
    horizon_months: int = 48,
    fixed_cost: float = 5.0,
    variable_cost_rate: float = 0.15,
    monthly_discount_rate: float = 0.01,
    risk_aversion: float = 0.25,  #lambda
    logit_shift: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for p in price_grid:
        p = float(p)

        hazards = predict_hazards_log(model, feature_columns, base_profile, price=p, horizon_months=horizon_months, logit_shift=logit_shift)
        S = survival_from_hazards(hazards)

        #mean/std LTV from distribution
        mean_ltv, std_ltv = ltv_mean_std_from_hazards(
            hazards,
            price=p,
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
            "S_end": float(S[-1]),
        })

    return pd.DataFrame(rows).sort_values("price").reset_index(drop=True)


def argmax_price(df: pd.DataFrame, col: str = "mean_LTV") -> dict:
    i = int(df[col].idxmax())
    return df.loc[i].to_dict()
