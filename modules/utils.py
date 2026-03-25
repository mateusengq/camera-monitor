"""
utils.py
--------
Shared utility functions used across the analytical pipeline.

Centralising these here avoids duplication across modules and ensures
that any logic change (e.g. ISO week edge cases) propagates everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loader import DATE_COL


# ---------------------------------------------------------------------------
# ISO week helpers
# ---------------------------------------------------------------------------

def last_complete_iso_week(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return (monday, sunday) of the last complete ISO week in the dataframe.

    A week is considered complete if it contains a Sunday (dayofweek == 6).
    Falls back to the last 7 days if no Sunday is present.

    ISO week edge case (year boundary):
    ISO weeks can start in one calendar year and end in another.
    This function derives monday as sunday - 6 days, which is always correct
    regardless of year boundary.
    """
    sundays = df.loc[df[DATE_COL].dt.dayofweek == 6, DATE_COL]

    if sundays.empty:
        week_end   = df[DATE_COL].max()
        week_start = week_end - pd.Timedelta(days=6)
        return week_start, week_end

    week_end   = sundays.max()
    week_start = week_end - pd.Timedelta(days=6)
    return week_start, week_end


def iso_week_mask(
    df: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> pd.Series:
    """Return a boolean mask for rows within the given ISO week boundaries."""
    return (df[DATE_COL] >= week_start) & (df[DATE_COL] <= week_end)


# ---------------------------------------------------------------------------
# Week-level signal extractors
# ---------------------------------------------------------------------------

def week_bool(df: pd.DataFrame, mask: pd.Series, col: str) -> bool:
    """
    Return True if the column fired (any True value) within the masked rows.
    Returns False if the column doesn't exist or has no valid data.
    """
    if col not in df.columns:
        return False
    vals = df.loc[mask, col]
    return bool(vals.any()) if vals.notna().any() else False


def week_scalar(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    """
    Return the median value of a column within the masked rows.
    Returns NaN if the column doesn't exist or has no valid data.
    """
    if col not in df.columns:
        return np.nan
    vals = df.loc[mask, col].dropna()
    return float(vals.median()) if len(vals) > 0 else np.nan
