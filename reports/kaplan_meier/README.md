# Kaplan–Meier Retention Baseline (Subscription Lifetime)

## Objective

This report provides a **non-parametric retention baseline** using a Kaplan–Meier estimator.
The goal is to understand the **subscription lifetime distribution** before introducing
feature-driven churn models or pricing policies.

In a subscription-based FinTech context, this baseline helps quantify:
- expected customer longevity
- long-run retention levels
- whether churn is an early- or late-lifecycle phenomenon


## Data & Definitions

Input dataset: `data/processed/survival_dataset.parquet`

- `duration` (months): observed customer lifetime in months
- `event`: 1 if churn occurred during the observation window, 0 if right-censored

The Kaplan–Meier curve estimates the survival function:

S(t) = P(customer remains active beyond month t)


## Key Results (from `summary.json`)

- Median lifetime: **24 months**
- Survival at 6 months: **1.00**
- Survival at 12 months: **1.00**

Interpretation:
- There is **no observed churn in the first 12 months**.
- Churn appears to be a **late-lifecycle event** rather than a short-term behavior-driven outcome.


## Implications for Modeling & Pricing

These results explain why short-horizon churn classification can fail:
- If churn is structurally absent in early months, month-ahead churn labels provide limited signal.
- Pricing decisions should be framed around **lifetime value** and **time-to-event dynamics** rather than
  short-term churn probability.

Planned next steps:
- model churn with discrete-time hazard / survival modeling (starting from later tenures)
- integrate survival curves into a risk-adjusted LTV framework
- optimize pricing based on expected LTV rather than short-term churn classification


## Files

- `survival_curve.csv`: Kaplan–Meier survival curve (timeline, survival probability)
- `summary.json`: key retention statistics (median lifetime and survival at selected horizons)
