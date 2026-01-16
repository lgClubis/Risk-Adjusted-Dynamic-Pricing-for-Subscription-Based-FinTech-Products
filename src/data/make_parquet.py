from __future__ import annotations
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/raw_parquet")

def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    for f in csv_files:
        df = pd.read_csv(f)
        out_path = OUT_DIR / f.with_suffix(".parquet").name

        # Keep index out of the file; parquet is columnar
        df.to_parquet(out_path, index=False)

        print(f"wrote: {out_path}  rows={len(df):,}  cols={df.shape[1]}")

if __name__ == "__main__":
    main()
