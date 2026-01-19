from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#Saving results
from pathlib import Path
import json


DATA_PATH = "data/processed/customer_month.parquet"


def time_split(df: pd.DataFrame, date_col: str = "month", test_frac: float = 0.2):
    df = df.sort_values(date_col)
    months = df[date_col].drop_duplicates().sort_values()
    cut_idx = int((1 - test_frac) * len(months))
    cut_month = months.iloc[cut_idx]
    train = df[df[date_col] < cut_month].copy()
    test = df[df[date_col] >= cut_month].copy()
    return train, test, cut_month


def main():
    df = pd.read_parquet(DATA_PATH)

    #Basic cleaning
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["account_id", "month"])

    target = "churn_next_month"

    #Feature set
    feature_cols = [
        "is_subscribed",
        "tenure_months",
        "usage_count",
        "usage_duration_secs",
        "error_count",
        "is_beta_feature",
        "usage_events",
        "ticket_count",
        "resolution_time_hours",
        "first_response_time_minutes",
        "satisfaction_score",
        "escalation_flag",
    ]

    #Some of these might be missing depending on aggregation; keep only existing
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Drop rows with missing target (shouldn't happen, but safe)
    df = df.dropna(subset=[target])

    #Time split
    train_df, test_df, cut_month = time_split(df, "month", test_frac=0.2)
    print(f"Time split cutoff month: {cut_month.date()}")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    X_train = train_df[feature_cols]
    y_train = train_df[target].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[target].astype(int)

    #Baseline model Logistic Regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)

    #Predictions
    p_test = model.predict_proba(X_test)[:, 1]

    #Metrics
    auc = roc_auc_score(y_test, p_test) if y_test.nunique() > 1 else float("nan")
    ap = average_precision_score(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Avg Precision (PR-AUC-ish): {ap:.4f}")
    print(f"Brier (calibration): {brier:.4f}")

    #Quick risk segmentation check
    test_df = test_df.copy()
    test_df["p_churn_next"] = p_test
    test_df["decile"] = pd.qcut(test_df["p_churn_next"], 10, labels=False, duplicates="drop")
    grp = test_df.groupby("decile").agg(
        n=("account_id", "size"),
        avg_pred=("p_churn_next", "mean"),
        actual_rate=(target, "mean"),
    ).reset_index()
    print("\nDecile table (higher decile = higher predicted risk):")
    print(grp)


    #Save baseline report
    REPORT_DIR = Path("reports/baseline_logreg")

    #Save metrics
    metrics = {
        "cutoff_month": str(cut_month.date()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "roc_auc": float(auc) if auc == auc else None,
        "avg_precision": float(ap),
        "brier": float(brier),
        "features": feature_cols,
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    #Save deciles
    grp.to_csv(REPORT_DIR / "deciles.csv", index=False)

    #Save coefficients (for logistic regression)
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]

    coef = pd.DataFrame({
        "feature": feature_cols,
        "coef": clf.coef_.ravel(),
    })
    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.sort_values("abs_coef", ascending=False)
    coef.to_csv(REPORT_DIR / "coefficients.csv", index=False)

    print(f"\nSaved report artifacts to: {REPORT_DIR}")

if __name__ == "__main__":
    main()
