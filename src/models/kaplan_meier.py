from lifelines import KaplanMeierFitter
import pandas as pd

DATA_PATH = "data/processed/survival_dataset.parquet"

def main():
    df = pd.read_parquet(DATA_PATH)

    kmf = KaplanMeierFitter()
    kmf.fit(df["duration"], event_observed=df["event"])

    print("Median lifetime:", kmf.median_survival_time_)
    print("Survival at 6 months:", kmf.survival_function_at_times(6).values)
    print("Survival at 12 months:", kmf.survival_function_at_times(12).values)
    print("Survival at 12 months:", kmf.survival_function_at_times(16).values)
    print("Survival at 12 months:", kmf.survival_function_at_times(20).values)

if __name__ == "__main__":
    main()
