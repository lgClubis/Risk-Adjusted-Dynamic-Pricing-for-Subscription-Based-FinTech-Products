from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DATA_PATH = "data/processed/hazard_dataset.parquet"
REPORT_DIR = Path("reports/hazard_logreg")

def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("month")
    months = df["month"].drop_duplicates().sort_values()
    cut_idx = int((1 - test_frac) * len(months))
    cut_month = months.iloc[cut_idx]
    train = df[df["month"] < cut_month].copy()
    test = df[df["month"] >= cut_month].copy()
    return train, test, cut_month

def main():

    df = pd.read_parquet(DATA_PATH)
    df["month"] = pd.to_datetime(df["month"])

    target = "y_churn"

    #Features: exclude identifiers & leakage
    exclude = {"account_id", "month", "subscription_id", "churn_in_month", target}
    feature_cols = [c for c in df.columns if c not in exclude]

    #Train/test time split
    train_df, test_df, cut_month = time_split(df, test_frac=0.2)

    X_train, y_train = train_df[feature_cols], train_df[target].astype(int)
    X_test, y_test = test_df[feature_cols], test_df[target].astype(int)

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, p) if y_test.nunique() > 1 else float("nan")
    ap = average_precision_score(y_test, p)
    brier = brier_score_loss(y_test, p)

    #Deciles
    out = test_df[["account_id", "month"]].copy()
    out["p_hazard"] = p
    out["y"] = y_test.values
    out["decile"] = pd.qcut(out["p_hazard"], 10, labels=False, duplicates="drop")
    dec = out.groupby("decile").agg(
        n=("y", "size"),
        avg_pred=("p_hazard", "mean"),
        actual_rate=("y", "mean"),
    ).reset_index()

    #Coefficients
    clf = model.named_steps["clf"]
    coef = pd.DataFrame({"feature": feature_cols, "coef": clf.coef_.ravel()})
    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.sort_values("abs_coef", ascending=False)

    metrics = {
        "cutoff_month": str(cut_month.date()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "event_rate_test": float(y_test.mean()),
        "roc_auc": float(auc) if auc == auc else None,
        "avg_precision": float(ap),
        "brier": float(brier),
        "features": feature_cols,
    }

    (REPORT_DIR / "deciles.csv").write_text(dec.to_csv(index=False), encoding="utf-8")
    (REPORT_DIR / "coefficients.csv").write_text(coef.to_csv(index=False), encoding="utf-8")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("\nDeciles:\n", dec.head(10))
    print(f"\nSaved to: {REPORT_DIR}")

if __name__ == "__main__":
    main()
