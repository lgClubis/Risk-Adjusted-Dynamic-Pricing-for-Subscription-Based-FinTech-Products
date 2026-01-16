from __future__ import annotations
from pathlib import Path
import pandas as pd

PQ_DIR = Path("data/processed/raw_parquet")

def load_table(name: str) -> pd.DataFrame:
    path = PQ_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)

def list_tables() -> list[str]:
    return sorted([p.stem for p in PQ_DIR.glob("*.parquet")])
