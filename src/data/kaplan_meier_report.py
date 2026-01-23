from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from lifelines import KaplanMeierFitter

DATA_PATH = "data/processed/survival_dataset.parquet"
REPORT_DIR = Path("reports/kaplan_meier")

def main():

    df = pd.read_parquet(DATA_PATH)

    durations = df["duration"].astype(float)
    events = df["event"].astype(int)

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)

    # Survival curve as a tidy CSV
    surv = kmf.survival_function_.reset_index()
    surv.columns = ["timeline", "survival_prob"]
    surv.to_csv(REPORT_DIR / "survival_curve.csv", index=False)

    # Summary numbers
    def s_at(t: int) -> float:
        return float(kmf.survival_function_at_times(t).values[0])

    summary = {
        "n_accounts": int(len(df)),
        "event_rate": float(events.mean()),
        "median_lifetime_months": float(kmf.median_survival_time_),
        "survival_at_6_months": s_at(6),
        "survival_at_12_months": s_at(12),
        "survival_at_22_months": s_at(22),
        "survival_at_24_months": s_at(24),
    }

    with open(REPORT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved Kaplan–Meier report to: {REPORT_DIR}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
