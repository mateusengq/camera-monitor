"""
baseline.py
-----------
Computes the expected baseline for each camera column using:

    1. Day-of-week (DOW) segmentation  — each weekday gets its own baseline
    2. Recency-weighted median          — recent weeks contribute more
    3. Exclusion of event weeks         — rows where _has_event is True are dropped
    4. Exclusion of zero rows           — rows where the camera value is 0 are dropped
    5. Configurable lookback window     — 3, 6, or 12 months

Output
------
For every (date, camera) pair the module returns:
    baseline_median   : recency-weighted pseudo-median for that DOW
    baseline_mean     : simple mean over the filtered window (kept for reference)
    residual          : observed - baseline_median
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from loader import DATE_COL, EVENT_COL, TOTAL_COL


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

LookbackMonths = Literal[3, 6, 12]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cutoff_date(df: pd.DataFrame, months: LookbackMonths) -> pd.Timestamp:
    """Return the earliest date to include given the lookback window."""
    max_date = df[DATE_COL].max()
    return max_date - pd.DateOffset(months=months)


def _recency_weights(dates: pd.Series, decay: float) -> np.ndarray:
    """
    Compute exponential recency weights for a series of dates.

    The most recent date receives weight 1.0; earlier dates are discounted
    exponentially based on their distance (in days) from the most recent date.

    Parameters
    ----------
    dates : pd.Series of datetime
        Dates of the observations being weighted.
    decay : float in [0.0, 1.0]
        Controls how fast older observations are discounted.
        0.0  → all weights equal (no recency preference)
        1.0  → maximum discount (only the most recent observation matters)

    Returns
    -------
    np.ndarray
        Normalised weights summing to 1.0.
    """
    if decay == 0.0 or len(dates) <= 1:
        return np.ones(len(dates)) / len(dates)

    max_date  = dates.max()
    days_back = (max_date - dates).dt.days.values.astype(float)

    # daily decay rate derived from the decay parameter:
    #   weight = exp(-rate * days_back)
    # decay=1 → rate chosen so the oldest obs in a 365d window gets weight ~0.01
    rate    = decay * (np.log(100) / 365)
    weights = np.exp(-rate * days_back)
    return weights / weights.sum()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the weighted median.

    Sort values, accumulate normalised weights, return the value at the
    50 % cumulative weight threshold.
    """
    if len(values) == 0:
        return np.nan

    sort_idx         = np.argsort(values)
    sorted_values    = values[sort_idx]
    sorted_weights   = weights[sort_idx]
    cumulative       = np.cumsum(sorted_weights)
    median_idx       = np.searchsorted(cumulative, 0.5)
    median_idx       = min(median_idx, len(sorted_values) - 1)
    return float(sorted_values[median_idx])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_baseline(
    df: pd.DataFrame,
    camera_cols: list[str],
    lookback_months: LookbackMonths = 12,
    decay: float = 0.3,
) -> pd.DataFrame:
    """
    Compute baseline values for every camera column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from loader.load_csv.
    camera_cols : list[str]
        Camera column names to process.
    lookback_months : {3, 6, 12}
        Historical window used for baseline computation.
    decay : float in [0.0, 1.0]
        Recency decay factor. 0 = flat weights, 1 = heavy recency preference.

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional columns per camera:
            {camera}_baseline_median
            {camera}_baseline_mean
            {camera}_residual
    """
    cutoff = _cutoff_date(df, lookback_months)
    result = df.copy()

    for cam in camera_cols:
        medians: list[float] = []
        means:   list[float] = []

        for idx, row in df.iterrows():
            target_dow  = row["_dow"]
            target_date = row[DATE_COL]

            # Filter: same DOW, within lookback window, ISO week has no event, no zero
            mask = (
                (df["_dow"]                 == target_dow)  &
                (df[DATE_COL]               >= cutoff)       &
                (df[DATE_COL]               <  target_date)  &
                (~df["_iso_week_has_event"])                 &  # entire ISO week clean
                (df[cam]                    >  0)
            )

            pool = df.loc[mask, [DATE_COL, cam]].dropna(subset=[cam])

            if pool.empty:
                # Fallback: relax the event exclusion but keep the zero exclusion
                mask_fallback = (
                    (df["_dow"]   == target_dow) &
                    (df[DATE_COL] >= cutoff)      &
                    (df[DATE_COL] <  target_date) &
                    (df[cam]      >  0)
                )
                pool = df.loc[mask_fallback, [DATE_COL, cam]].dropna(subset=[cam])

            if pool.empty:
                medians.append(np.nan)
                means.append(np.nan)
                continue

            values  = pool[cam].values.astype(float)
            weights = _recency_weights(pool[DATE_COL], decay)

            medians.append(_weighted_median(values, weights))
            means.append(float(np.average(values, weights=weights)))

        result[f"{cam}_baseline_median"] = medians
        result[f"{cam}_baseline_mean"]   = means
        result[f"{cam}_residual"]        = result[cam] - result[f"{cam}_baseline_median"]

    return result
