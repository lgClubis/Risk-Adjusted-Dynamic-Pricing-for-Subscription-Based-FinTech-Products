# Discrete-Time Hazard Model (Logistic Regression) – Results & Interpretation

## Objective

This model estimates a **monthly churn hazard** for subscription accounts:

h_t = P(churn_in_month = 1 | active at start of month, X_t)

Given earlier analysis (Kaplan–Meier), churn is a **late-lifecycle phenomenon** in this dataset.
Therefore, the hazard model is trained only on observations where `tenure_months >= 12`.

The purpose of this model is to provide a **risk layer** that can be integrated into
a **risk-adjusted LTV framework** and, ultimately, a dynamic pricing policy.


## Data & Labeling

Input dataset: `data/processed/hazard_dataset.parquet`

Construction rules (high level):
- Each row represents an `account_id × month` observation.
- "At-risk" months are months where the account has **not churned prior to the month**.
- Target: `y_churn = churn_in_month` (churn occurs in this month).
- Filtering: `tenure_months >= 12` to focus on the lifecycle region where churn occurs.


## Model Setup

- Model: Logistic Regression (class-weighted)
- Split: Time-based split
  - Training data: months before cutoff (see `metrics.json`)
  - Test data: months from cutoff onward
- Features:
  - Engagement: usage counts/duration/errors
  - Support: ticket count, resolution time, satisfaction score, escalation flag
  - Tenure: `tenure_months`
  - Subscription indicator: `is_subscribed`


## Performance Summary (from `metrics.json`)

Key metrics on the test set:
- ROC-AUC: ~0.52
- Average Precision: ~0.24
- Brier Score: ~0.36
- Test event rate: ~0.23

Interpretation:
- The model shows **weak but non-zero ranking power** (AUC slightly above 0.5).
- Average Precision is above the base event rate, indicating **some predictive signal**.
- Calibration is not ideal (Brier is high), but the model is **usable as a baseline risk layer**
  and can be improved via calibration and/or additional features.


## Decile Analysis

See `deciles.csv`.

Predicted hazard increases across deciles, while realized hazard rates vary due to limited sample size.
Overall, the decile view suggests the model provides **coarse risk segmentation** rather than
high-precision churn forecasts.

This is aligned with the project’s decision goal:
- Not to perfectly predict churn events,
- but to obtain a **risk-sensitive lifetime model** that supports pricing decisions.


## Limitations

- Limited feature set and synthetic nature of the dataset constrain predictive performance.
- Hazard rates are modeled at monthly granularity and may miss within-month dynamics.
- The model currently does not include an explicit price variable; pricing effects will be introduced next.


## Next Steps

1) Introduce a price variable and estimate/assume a **monotone price → hazard sensitivity**.
2) Convert hazard predictions into survival curves and expected remaining lifetime.
3) Compute risk-adjusted expected LTV and optimize prices under constraints.
