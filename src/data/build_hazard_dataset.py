from __future__ import annotations
import pandas as pd

IN_PATH = "data/processed/customer_month.parquet"
OUT_PATH = "data/processed/hazard_dataset.parquet"

def main():
    df = pd.read_parquet(IN_PATH)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["account_id", "month"])

    #cumulative churn up to current month
    df["cum_churn"] = df.groupby("account_id")["churn_in_month"].cumsum()

    #cumulative churn strictly BEFORE current month (i.e., at start of month)
    df["cum_churn_prior"] = df.groupby("account_id")["cum_churn"].shift(1).fillna(0).astype(int)

    #At risk if:
    #- subscribed this month
    #- not churned before this month
    #- tenure >= 12 (based on KM: no churn before 20 months)
    at_risk = df[
        (df["is_subscribed"] == 1) &
        (df["cum_churn_prior"] == 0) &
        (df["tenure_months"] >= 12)
    ].copy()

    #Target: churn happens in THIS month
    at_risk["y_churn"] = at_risk["churn_in_month"].astype(int)

    #Drop helper columns
    drop_cols = ["churn_next_month", "cum_churn", "cum_churn_prior"]
    at_risk = at_risk.drop(columns=[c for c in drop_cols if c in at_risk.columns])

    at_risk.to_parquet(OUT_PATH, index=False)

    print(f"Wrote hazard dataset: {OUT_PATH} | rows={len(at_risk):,} cols={at_risk.shape[1]}")
    print("Churn rate in hazard dataset:", at_risk["y_churn"].mean())
    print("Number of churn events:", int(at_risk["y_churn"].sum()))

if __name__ == "__main__":
    main()
