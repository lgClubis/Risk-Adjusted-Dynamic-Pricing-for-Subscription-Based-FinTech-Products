# Pricing Simulator (One Profile) — LTV vs Price

This report contains the output of a **pricing simulator** that uses the **pricing-augmented discrete-time hazard model** (`reports/hazard_logreg_pricing/model.joblib`) to evaluate **expected lifetime value (LTV)** across a grid of subscription prices.

Folder contents:
- `ltv_vs_price.csv` — simulation results (one row per candidate price)
- `ltv_vs_price.png` — LTV vs price curve for the chosen profile


## What i did

I ran an end-to-end simulation for **one representative customer-month row** sampled from:

- `data/processed/hazard_dataset.parquet`

For each candidate monthly price `p` in a grid (e.g., 20…120), I:

1. **Constructed a forward-looking horizon** of length `H = 48` months:
   - All customer features were held constant **except** `tenure_months`, which was incremented by +1 each simulated month.
   - I injected the pricing levers:
     - `price = p`
     - `log_price = log(p)`

2. **Predicted monthly churn hazards** using the trained hazard model:
   - `hazard_t(p) = P(churn in month t | active at start of month t, price = p)`

3. Converted hazards into a **survival curve**:
   - `S[0] = 1`
   - `S[t] = Π_{i=1..t} (1 - hazard_i)`
   - Interpretation: `S[t]` is the probability the customer is still active at the **start** of month `t+1`.

4. Computed expected remaining months and expected LTV:
   - Expected remaining paid months:
     - `E_months = Σ_{t=1..H} S[t-1]`
   - Monthly margin (simple proxy):
     - `margin(p) = (1 - variable_cost_rate) * p - fixed_cost`
   - Expected discounted LTV:
     - `LTV(p) = Σ_{t=1..H} margin(p) * S[t-1] * df[t-1]`
     - where `df[k] = 1 / (1 + r)^k` and `r` is the monthly discount rate.

This simulator is **decision-oriented**: it is meant to support pricing decisions by explicitly linking **price → churn risk → lifetime → value**.


## Assumptions (explicit)

**Billing convention**
- I assume we earn month `t` margin if the customer is active at the **start** of month `t`.
- Under this convention, if churn happens during month `t`, month `t` revenue is still counted.

**Feature dynamics**
- Features are treated as static over the horizon except `tenure_months`.
- This isolates the structural effect of price (via `log_price`) on churn hazard.
- Extensions can add a feature-update model later.

**Cost model**
- Fixed monthly cost proxy: `fixed_cost = 5.0`
- Variable cost proxy: `variable_cost_rate = 0.15` (15%)

**Discounting**
- Monthly discount rate used: `r = 0.01` (1% per month)

**Horizon**
- `H = 48` simulated months (finite horizon approximation)


## How to read the outputs

### `ltv_vs_price.csv` columns

- `price`  
  Candidate monthly subscription price being evaluated.

- `margin`  
  Monthly gross margin proxy from the cost model:  
  `margin = (1 - variable_cost_rate) * price - fixed_cost`

- `hazard_mean`  
  Average predicted monthly churn hazard across the simulated horizon for this price.

- `E_months`  
  Expected number of remaining paid months within the horizon:  
  `E_months = Σ S[t-1]`  
  Smaller values mean the model expects faster churn.

- `S_end`  
  Probability the customer survives beyond the horizon: `S[H]`.  
  This is typically very small when hazards are high.

- `LTV`  
  Expected discounted lifetime value over the horizon:  
  `LTV = Σ margin * S[t-1] * df[t-1]`

### `ltv_vs_price.png`

A line plot of `LTV` vs `price`.  
The maximum point corresponds to the **mean-LTV optimal price** under these assumptions.


## What the current run indicates (one sampled profile)

For the sampled profile used in `run_one_profile.py`, hazards were very high (mean hazard close to 1), implying:
- Survival drops quickly over the first few months.
- Expected remaining months is close to ~1 month order of magnitude.
- In this regime, higher prices can still win because margin increases strongly while expected lifetime decreases only slightly.

In the logged run, the mean-LTV optimum occurred at the **upper end of the tested grid (price = 120)**, meaning:
- The incremental margin dominated the incremental churn-risk increase for this particular profile and model.


## Caveats and next steps

1. **Risk-adjusted optimization**  
   Next phase adds risk measures (e.g., standard deviation, CVaR) and optimizes:
   - `risk_adj_LTV = mean_LTV - λ * std_LTV`
   This yields more conservative pricing when higher prices create more downside risk.

2. **Calibration**  
   If predicted hazards are systematically too high/low, an intercept-shift calibration can be applied so that:
   - average predicted hazard aligns with observed event rates,
   while keeping price sensitivity (the `log_price` coefficient) intact.

3. **Segmentation**  
   This folder shows **one profile**. A production decision policy should run the simulator across:
   - customer segments (e.g., usage/support tiers), or
   - many sampled profiles with aggregation.

## Reproducibility

- Model artifact: `reports/hazard_logreg_pricing/model.joblib`
- Feature list: `reports/hazard_logreg_pricing/feature_columns.json`
- Data source: `data/processed/hazard_dataset.parquet`
- Runner: `src/pricing/run_one_profile.py`
- Core functions: `src/pricing/ltv_pricing_simulator.py`
