# Baseline Churn Risk Model – Interpretation

## Objective

The goal of this baseline model was to estimate the **short-term churn risk**
of subscription accounts in order to serve as a foundation for
risk-adjusted pricing decisions.

Formally, the model estimates:

P(churn_next_month = 1 | customer behavior and support signals in month t)

This baseline is intentionally simple and interpretable and serves as a
reference point for more advanced approaches.


## Model Setup

- Model: Logistic Regression (class-weighted)
- Split: Time-based split
  - Training data: months before 2024-08-01
  - Test data: months from 2024-08-01 onward
- Target: `churn_next_month`
- Features:
  - Engagement: usage counts, usage duration, errors
  - Support: ticket volume, resolution time, satisfaction score
  - Account state: tenure, subscription status


## Performance Summary

Metric | Value 

ROC-AUC | ~0.47 
Average Precision | ~0.10 
Brier Score | ~0.60 

Key observations:
- The ROC-AUC is below 0.50, indicating worse-than-random ranking.
- Average Precision is close to the base churn rate.
- The Brier score indicates severe miscalibration.


## Decile Analysis

Predicted churn probabilities increase monotonically across deciles
(from ~0.59 to ~0.92), but **actual churn rates remain flat at ~10–13%**
and even decline in higher-risk deciles.

This indicates:
- No meaningful ranking of customers by churn risk
- Severe overestimation of absolute churn probabilities
- No decision-usable risk segmentation


## Interpretation & Diagnosis

The results suggest that **short-horizon churn classification is not a suitable
formulation for this problem**, given the available data.

Likely reasons include:
- Churn events are sparse and occur as single terminal events
- Behavioral and support features show limited short-term signal
- Churn appears to be driven more by lifecycle timing or exogenous factors
  than by immediate usage or support patterns
- Monthly binary labels are too coarse to capture gradual risk dynamics

As a result, the model produces misleading probability estimates that
would be harmful if used for pricing or retention decisions.


## Implications for Pricing Decisions

Using this model to drive pricing or discount decisions would lead to:
- Incorrect identification of high-risk customers
- Systematic misallocation of discounts
- Significant value destruction due to poor risk calibration

Therefore, this model is **not suitable for decision-making** and is retained
solely as a documented baseline.
