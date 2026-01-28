# Population Pricing Policy (Calibrated) — Retention-Constrained Frontier

This report evaluates dynamic pricing decisions using a calibrated discrete-time hazard model and a risk-adjusted LTV objective.

## What is in this folder
- `population_ltv_vs_price.csv` — population aggregates across a price grid (mean LTV, risk-adjusted LTV, avg hazard, avg E_months, avg S6).
- `population_ltv_vs_price.png` — population mean vs risk-adjusted LTV curve (unconstrained).
- `retention_frontier_S6.csv` — efficient frontier for retention constraints: best feasible price under `avg_S6 >= threshold`.
- `frontier_price_vs_S6.png` — optimal price as a function of the retention threshold.
- `frontier_value_vs_S6.png` — best feasible risk-adjusted LTV as a function of the retention threshold.

## Method summary
1) **Calibration (logit shift)**
We estimate a constant logit shift on the time-based test split so that the average predicted hazard matches the observed test event rate.
This preserves the learned feature sensitivities (including `log_price`) but corrects baseline calibration.

2) **Population simulation**
We sample `N=200` customer-month profiles from `hazard_dataset.parquet` (pricing features removed).
For each candidate price `p`, we simulate a fixed horizon `H=48` months and predict monthly churn hazards:
- `hazard_t(p) = P(churn in month t | active at start of month t, features, price=p)`

From hazards we compute:
- `E_months` (expected paid months over the horizon)
- per-profile discounted `mean_ltv(p)` and distribution-driven `std_ltv(p)` (uncertainty from churn timing)
- population aggregates per price:
  - `pop_mean_ltv(p)` = average of per-profile mean LTV
  - `pop_std_ltv(p)` = standard deviation across profiles (cross-customer variability)
  - `pop_risk_adj_ltv(p)` = `pop_mean_ltv - λ * pop_std_ltv`

3) **Retention guardrail**
Unconstrained optimization can push prices to the grid maximum when margin dominates churn sensitivity.
To produce business-feasible policies, we impose a retention constraint using:
- `avg_S6(p)` = average survival probability to month 6 across profiles

We compute an **efficient frontier**:
- For each threshold `τ`, choose the price that maximizes `pop_risk_adj_ltv(p)` subject to `avg_S6(p) >= τ`.

## Key interpretation
- **Unconstrained optimum** can be overly aggressive (high price, very low retention).
- **Retention-constrained optimum** yields interior solutions:
  tighter retention requirements → lower optimal prices → lower LTV but higher expected lifetime/retention.

## Reproducibility
- Model artifacts: `reports/hazard_logreg_pricing/model.joblib`, `feature_columns.json`
- Data: `data/processed/hazard_dataset.parquet`
- Runner: `src/pricing_population/run_population_policy.py`
- Core hazard simulation: `src/pricing/adjusted_ltv_pricing_simulator.py`
- Risk math: `src/pricing/risk_adjusted_ltv_optimizer.py`
