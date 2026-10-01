from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _sniff_delimiter(path: str | Path, default: str = ",") -> str:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except csv.Error:
        return ";" if sample.count(";") > sample.count(",") else default


def load_finam_export(path: str | Path) -> pd.DataFrame:
    """Load a Finam-style daily export and return a clean DateTime-indexed frame."""

    path = Path(path)
    delimiter = _sniff_delimiter(path)
    df = pd.read_csv(path, sep=delimiter, engine="python")
    df.columns = [re.sub(r"[<>]", "", str(c)).strip().upper() for c in df.columns]

    required = {"DATE", "CLOSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}; got {list(df.columns)}")

    date_raw = df["DATE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    time_raw = (
        df["TIME"].astype(str).str.strip().str.replace(r"\D", "", regex=True).replace("", "0").str.zfill(6)
        if "TIME" in df.columns
        else pd.Series("000000", index=df.index)
    )

    date6 = date_raw.str.zfill(6)
    dt = pd.to_datetime(date6 + time_raw, format="%y%m%d%H%M%S", errors="coerce")

    if dt.isna().any():
        date8 = date_raw.str.zfill(8)
        dt8 = pd.to_datetime(date8 + time_raw, format="%Y%m%d%H%M%S", errors="coerce")
        dt = dt.fillna(dt8)

    df.insert(0, "DATETIME", dt)

    for column in ["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["DATETIME", "CLOSE"]).sort_values("DATETIME")

    dedupe_cols = [c for c in ["TICKER", "PER", "DATETIME"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")

    return df.set_index("DATETIME")


def close_prices(frame: pd.DataFrame) -> pd.Series:
    prices = frame["CLOSE"].astype(float).dropna().rename("close")
    if prices.empty:
        raise ValueError("No valid close prices are available.")
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive before taking logarithms.")
    return prices


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Continuously compounded returns: log(P_t) - log(P_{t-1})."""

    prices = prices.astype(float).dropna()
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive before taking logarithms.")
    returns = np.log(prices).diff().dropna()
    returns.name = "log_return"
    return returns
