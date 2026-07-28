from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_series_csv(path: Path, date_col: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"missing date column {date_col!r} in {path}")
    if value_col not in df.columns:
        raise ValueError(f"missing value column {value_col!r} in {path}")
    dates = pd.to_datetime(df[date_col])
    values = pd.to_numeric(df[value_col], errors="coerce")
    series = pd.Series(values.values, index=dates).sort_index()
    return series.dropna()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
