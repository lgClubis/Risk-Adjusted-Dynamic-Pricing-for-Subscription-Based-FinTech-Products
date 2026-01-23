# Discrete-Time Hazard Model with Price Sensitivity

## Objective

This model extends the baseline hazard model by explicitly introducing
**price as a decision variable**.

The goal is not pure churn prediction, but to quantify how pricing decisions
affect customer lifetime via changes in the monthly churn hazard.

Formally, the model estimates:

h_t(p) = P(churn_in_month = 1 | active at start of month, X_t, price)

This hazard model serves as the risk layer for a
**risk-adjusted lifetime value (LTV) and dynamic pricing framework**.


## Data & Setup

Input dataset: `data/processed/hazard_dataset.parquet`

Key properties:
- Observations: account × month
- At-risk definition: account active at start of month
- Lifecycle focus: tenure ≥ 12 months
- Target: monthly churn hazard (`y_churn`)
- Price variable:
  - `price`: synthetic monthly price proxy
  - `log_price`: logarithmic transformation used in the model

The synthetic price variable is introduced transparently as a
**pricing lever for counterfactual simulation**, not as a claim about
true historical prices.


## Model Specification

- Model: Logistic Regression (class-weighted)
- Link function: logit
- Key extension: inclusion of `log_price`

Interpretation:
- Coefficient on `log_price` represents a **price elasticity of churn hazard**
- Positive coefficient implies higher prices increase churn risk
- The model enforces a monotone, economically plausible price effect


## Performance Summary

See `metrics.json` for full details.

The inclusion of price does not aim to maximize predictive performance.
Instead, it provides a **structural relationship** between pricing and risk
that can be used for lifetime and pricing simulations.


## Role in the Pricing Framework

This hazard model is used to:
1. Compute monthly hazard rates as a function of price
2. Derive survival curves S(t | p)
3. Estimate expected remaining lifetime
4. Compute risk-adjusted expected LTV(p)
5. Optimize pricing decisions under risk constraints

## Limitations

- Price variable is synthetic and used for simulation purposes
- Model calibration is approximate
- Results should be interpreted as directional and comparative,
  not as precise point forecasts


## Next Steps

- Convert hazard predictions into expected lifetime
- Compute LTV(p) across a price grid
- Derive an optimal dynamic pricing policy
