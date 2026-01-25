# Risk-Adjusted Pricing Simulator (One Profile)

This folder contains outputs from the **risk-adjusted dynamic pricing simulation** for a single sampled customer profile.  
The simulator uses the pricing-augmented discrete-time hazard model to evaluate **mean LTV** and **risk-adjusted LTV** across a grid of candidate prices.

## Files
- `ltv_vs_price.csv`  
  Price grid results (mean LTV, std LTV, risk-adjusted LTV, etc.).

- `ltv_vs_price.png`  
  Plot of `mean_ltv` and `risk_adj_ltv` vs `price` for a reference risk aversion λ (the one used when generating the plot).

- `lambda_sweep.csv` (if present)  
  Summary table showing how the **optimal price** changes as risk aversion λ varies.


## Method summary (what we did)

For each candidate monthly price `p` in a grid (e.g. 20…120):

1) **Forward simulation horizon**  
We simulate `H` future months. Customer features are held constant except:
- `tenure_months` increases by +1 each simulated month.

We inject the pricing features:
- `price = p`
- `log_price = log(p)`

2) **Hazard prediction (model of churn)**  
The hazard model predicts monthly churn probability:
- `hazard_t(p) = P(churn in month t | active at start of month t, features, price=p)`

3) **Survival from hazards**  
Survival is the probability of being active at the start of each month:
- `S[0] = 1`
- `S[t] = Π_{i=1..t} (1 - hazard_i)`

4) **Cashflows and margin**  
Monthly margin is a simple proxy:
- `margin(p) = (1 - variable_cost_rate) * p - fixed_cost`

Cashflow convention used:
- We earn month `t` margin if the customer is active at the **start** of month `t`.

5) **Mean LTV**  
Discounted expected LTV over horizon:
- `mean_ltv(p) = Σ_{t=1..H} margin(p) * S[t-1] * df[t-1]`
- `df[k] = 1/(1+r)^k` is the monthly discount factor.

6) **Risk (standard deviation of LTV)**  
We compute an **exact distribution** over churn time `T` induced by hazards:
- `P(T=t) = S[t-1] * hazard_t` for `t=1..H`
- `P(T=H+1) = S[H]` (survive beyond horizon)

For each possible churn time `T`, we compute total discounted cashflow `CF(T)`, then:
- `std_ltv(p) = Std[ CF(T) ]`

7) **Risk-adjusted objective**
We optimize:
- `risk_adj_ltv(p) = mean_ltv(p) - λ * std_ltv(p)`

Where:
- `λ` (lambda) is the **risk aversion** parameter.


## How to interpret outputs

### `ltv_vs_price.csv` columns
- `price`  
  Candidate monthly subscription price.

- `margin`  
  Monthly margin proxy from the cost model.

- `hazard_mean`  
  Average predicted monthly churn hazard over the simulated horizon.

- `E_months`  
  Expected number of remaining paid months within the horizon:  
  `E_months = Σ S[t-1]`

- `mean_ltv`  
  Expected discounted LTV (mean).

- `std_ltv`  
  Standard deviation of LTV due to uncertainty in churn timing.

- `risk_adj_ltv`  
  Risk-adjusted objective: `mean_ltv - λ * std_ltv` (λ set in the run that generated the file/plot).

### `lambda_sweep.csv` columns
- `lambda`  
  Risk aversion used.

- `best_mean_price`  
  Price that maximizes `mean_ltv`.

- `best_ra_price`  
  Price that maximizes `risk_adj_ltv` for that λ.

- `best_ra_mean`, `best_ra_std`, `best_ra_value`  
  The mean LTV, std LTV, and risk-adjusted value at the risk-adjusted optimum.


## Notes / caveats
- This report is for **one sampled profile**. A production-ready policy should be evaluated across:
  - segments (e.g. usage/support tiers), or
  - many sampled profiles with aggregation.
- If predicted hazards are systematically too high/low, the next step is **calibration** (e.g., logit intercept shift) so that baseline hazards align with observed event rates, while preserving the `log_price` sensitivity.


## Reproducibility
- Model: `reports/hazard_logreg_pricing/model.joblib`
- Features: `reports/hazard_logreg_pricing/feature_columns.json`
- Data: `data/processed/hazard_dataset.parquet`
- Runner: `src/pricing/run_one_profile.py`
- Core logic: `src/pricing/ltv_pricing_simulator.py`
- Risk layer: `src/pricing/risk_adjusted_ltv_optimizer.py`
