"""Utilities for loading and validating the raw S&P 500 dataset.

 The idea is to keep the data-loading logic in one place
and import it from notebooks, scripts or the future Flask app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# Project paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_FILE = ROOT_DIR / "data" / "raw" / "sp500.csv"


def load_raw_data(csv_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the historical S&P 500 dataset.

    Parameters
    ----------
    csv_path : Path, optional
        Location of the CSV file. Defaults to ``data/raw/sp500.csv``.

    Returns
    -------
    pandas.DataFrame
        Data indexed by datetime. Columns keep their original names.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find the raw dataset at {csv_path}. "
            "Did you download it or move it somewhere else?"
        )

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the CSV. Please double-check the source file.")

    # Basic cleaning so downstream notebooks can assume a tidy datetime index.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
    df = df.set_index("Date")
    return df


def dataset_summary(df: pd.DataFrame) -> Tuple[str, str]:
    """Return human-readable strings describing the dataframe.

    This keeps printing logic in a single place so tests or notebooks can call
    it and capture the same output.
    """
    n_rows, n_cols = df.shape
    date_min = df.index.min()
    date_max = df.index.max()

    meta = (
        f"Rows: {n_rows:,} | Columns: {n_cols} | Date range: "
        f"{date_min.date()} → {date_max.date()}"
    )
    missing = df.isna().sum().to_frame("missing").T
    return meta, missing.to_string()


def main() -> None:
    """Entry point for quick manual checks (``python -m src.data.load_data``)."""
    df = load_raw_data()
    meta, missing = dataset_summary(df)

    print("=== Dataset overview ===")
    print(meta)
    print("\n=== Column dtypes ===")
    print(df.dtypes)
    print("\n=== Missing value count ===")
    print(missing)


if __name__ == "__main__":
    main()
