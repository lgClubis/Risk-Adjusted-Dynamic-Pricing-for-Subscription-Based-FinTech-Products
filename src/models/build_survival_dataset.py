from __future__ import annotations
import pandas as pd

DATA_PATH = "data/processed/customer_month.parquet"
OUT_PATH = "data/processed/survival_dataset.parquet"

def main():
    df = pd.read_parquet(DATA_PATH)
    df["month"] = pd.to_datetime(df["month"])

    #Sort
    df = df.sort_values(["account_id", "month"])

    rows = []

    for acc_id, g in df.groupby("account_id"):
        g = g.sort_values("month")

        #Duration= last viewed tenure + 1
        duration = g["tenure_months"].max() + 1

        #Event= ever churned?
        event = int(g["churn_in_month"].max() == 1)

        #Features: last viewed churn
        if event == 1:
            last = g[g["churn_in_month"] == 1].iloc[0]
        else:
            last = g.iloc[-1]

        row = {
            "account_id": acc_id,
            "duration": duration,
            "event": event,
            "tenure_months": last["tenure_months"],
            "usage_count": last.get("usage_count", 0),
            "usage_duration_secs": last.get("usage_duration_secs", 0),
            "error_count": last.get("error_count", 0),
            "usage_events": last.get("usage_events", 0),
            "ticket_count": last.get("ticket_count", 0),
            "resolution_time_hours": last.get("resolution_time_hours", 0),
            "first_response_time_minutes": last.get("first_response_time_minutes", 0),
            "satisfaction_score": last.get("satisfaction_score", 0),
            "escalation_flag": last.get("escalation_flag", 0),
        }

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)

    print(f"Wrote survival dataset: {OUT_PATH} | rows={len(out):,}")

if __name__ == "__main__":
    main()
