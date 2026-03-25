"""
signals.py
----------
Computes all analytical signals per camera column over the full time series.

Signals produced
----------------
For each camera:

    EWMA
    ----
    {cam}_ewma          : exponentially weighted moving average of residuals
    {cam}_ewma_alarm    : bool — EWMA crossed its control limit

    CUSUM
    -----
    {cam}_cusum_pos     : cumulative sum tracking upward deviations
    {cam}_cusum_neg     : cumulative sum tracking downward deviations (stored as positive)
    {cam}_cusum_alarm   : bool — either CUSUM arm crossed threshold h

    Slope
    -----
    {cam}_slope_90d     : OLS slope of observed values over the last 90 days

    Zeros
    -----
    {cam}_zeros_pct_12m : fraction of non-event rows where camera = 0, last 12 months
    {cam}_zeros_week    : count of zero readings in the current analysis week

    Weekly drop
    -----------
    {cam}_week_pct_chg  : % change vs. the immediately preceding week (by DOW median)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from loader import DATE_COL, TOTAL_COL
from utils import last_complete_iso_week, week_scalar


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_signals(
    df: pd.DataFrame,
    camera_cols: list[str],
    ewma_lambda: float = 0.2,
    cusum_k: float = 1.0,
    cusum_h: float = 5.0,
) -> pd.DataFrame:
    """
    Compute all signals for every camera column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with baseline columns already attached (output of baseline.compute_baseline).
    camera_cols : list[str]
        Camera column names.
    ewma_lambda : float in (0, 1]
        Smoothing parameter for EWMA. Higher = more weight on recent residuals.
    cusum_k : float
        CUSUM slack (allowance). Controls sensitivity to small shifts.
        Typical range: 0.5 – 3.0 (in units of residual std).
    cusum_h : float
        CUSUM decision threshold. Alarm fires when cumsum exceeds h * residual_std.
        Typical range: 2.0 – 10.0.
    analysis_week_start : pd.Timestamp, optional
        Start of the week being analysed. If None, uses the last 7 days.
    analysis_week_end : pd.Timestamp, optional
        End of the week being analysed. If None, uses the latest date in df.

    Returns
    -------
    pd.DataFrame
        Input dataframe with signal columns appended.
    """
    result = df.copy().sort_values(DATE_COL).reset_index(drop=True)

    # Resolve analysis week boundaries from last complete ISO week
    analysis_week_start, analysis_week_end = last_complete_iso_week(result)

    for cam in camera_cols:
        residual_col = f"{cam}_residual"

        if residual_col not in result.columns:
            raise ValueError(
                f"Residual column '{residual_col}' not found. "
                "Run baseline.compute_baseline before compute_signals."
            )

        residuals = result[residual_col].values.astype(float)

        # Residual std used to scale CUSUM thresholds
        # Use nanstd over the full series; robust to a handful of NaNs
        sigma = float(np.nanstd(residuals)) or 1.0   # guard against zero std

        # ------------------------------------------------------------------
        # EWMA
        # ------------------------------------------------------------------
        ewma_vals  = _compute_ewma(residuals, ewma_lambda)
        ewma_limit = 3.0 * sigma * np.sqrt(ewma_lambda / (2 - ewma_lambda))

        result[f"{cam}_ewma"]       = ewma_vals
        result[f"{cam}_ewma_limit"] = ewma_limit
        result[f"{cam}_ewma_alarm"] = np.abs(ewma_vals) > ewma_limit

        # ------------------------------------------------------------------
        # CUSUM
        # ------------------------------------------------------------------
        k_scaled = cusum_k * sigma
        h_scaled = cusum_h * sigma

        cusum_pos, cusum_neg = _compute_cusum(residuals, k_scaled)

        result[f"{cam}_cusum_pos"]   = cusum_pos
        result[f"{cam}_cusum_neg"]   = cusum_neg
        result[f"{cam}_cusum_limit"] = h_scaled
        result[f"{cam}_cusum_alarm"] = (cusum_pos > h_scaled) | (cusum_neg > h_scaled)

        # ------------------------------------------------------------------
        # Slope over last 90 days
        # ------------------------------------------------------------------
        result[f"{cam}_slope_90d"] = _rolling_slope(
            result[DATE_COL], result[cam], window_days=90
        )

        # ------------------------------------------------------------------
        # Zeros % over last 12 months (excluding event ISO weeks)
        # ------------------------------------------------------------------
        cutoff_12m = result[DATE_COL].max() - pd.DateOffset(months=12)
        mask_12m   = (
            (result[DATE_COL]               >= cutoff_12m) &
            (~result["_iso_week_has_event"])
        )
        pool_12m   = result.loc[mask_12m, cam]

        zeros_pct = float((pool_12m == 0).sum() / len(pool_12m)) if len(pool_12m) > 0 else 0.0
        result[f"{cam}_zeros_pct_12m"] = zeros_pct

        # Zeros in analysis ISO week
        mask_week  = (
            (result[DATE_COL] >= analysis_week_start) &
            (result[DATE_COL] <= analysis_week_end)
        )
        zeros_week = int((result.loc[mask_week, cam] == 0).sum())
        result[f"{cam}_zeros_week"] = zeros_week

        # ------------------------------------------------------------------
        # Weekly % change — ISO week vs. previous ISO week
        # included even for event weeks (growth metrics keep event weeks)
        # ------------------------------------------------------------------
        result[f"{cam}_week_pct_chg"] = _weekly_pct_change(
            result, cam, analysis_week_start, analysis_week_end
        )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_ewma(residuals: np.ndarray, lam: float) -> np.ndarray:
    """
    Compute EWMA of a residual series.

    S_t = λ * r_t + (1 - λ) * S_{t-1}
    Initialised at 0 (expected residual under in-control process).
    NaNs in residuals are propagated — the EWMA holds its last value.
    """
    n      = len(residuals)
    ewma   = np.zeros(n)
    prev   = 0.0

    for i, r in enumerate(residuals):
        if np.isnan(r):
            ewma[i] = prev
        else:
            ewma[i] = lam * r + (1 - lam) * prev
            prev    = ewma[i]

    return ewma


def _compute_cusum(
    residuals: np.ndarray, k: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-sided CUSUM with slack k.

    C+_t = max(0, C+_{t-1} + r_t - k)   tracks upward shifts
    C-_t = max(0, C-_{t-1} - r_t - k)   tracks downward shifts

    Returns (cusum_pos, cusum_neg) — both stored as non-negative values.
    NaNs in residuals reset the accumulator to 0.
    """
    n         = len(residuals)
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)

    cp, cn = 0.0, 0.0

    for i, r in enumerate(residuals):
        if np.isnan(r):
            cp, cn = 0.0, 0.0
        else:
            cp = max(0.0, cp + r - k)
            cn = max(0.0, cn - r - k)

        cusum_pos[i] = cp
        cusum_neg[i] = cn

    return cusum_pos, cusum_neg


def _rolling_slope(
    dates: pd.Series,
    values: pd.Series,
    window_days: int = 90,
) -> pd.Series:
    """
    Compute the OLS slope of `values` over the trailing `window_days` for each row.

    Returns a Series aligned with the input index.
    Slope is in units of [value units / day].
    """
    slopes     = np.full(len(dates), np.nan)
    dates_arr  = dates.values
    values_arr = values.values.astype(float)

    for i in range(len(dates_arr)):
        cutoff   = dates_arr[i] - np.timedelta64(window_days, "D")
        mask     = (dates_arr >= cutoff) & (dates_arr <= dates_arr[i])
        x        = (dates_arr[mask] - dates_arr[mask].min()).astype("timedelta64[D]").astype(float)
        y        = values_arr[mask]

        valid    = ~np.isnan(y)
        if valid.sum() >= 3:
            slope, *_ = stats.linregress(x[valid], y[valid])
            slopes[i] = slope

    return pd.Series(slopes, index=dates.index)


def _weekly_pct_change(
    df: pd.DataFrame,
    cam: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> pd.Series:
    """
    Compute % change between current week and previous week for each row
    in the analysis window, based on the DOW median of each week.

    Rows outside the current analysis week get NaN.
    """
    pct_chg = pd.Series(np.nan, index=df.index)

    prev_start = week_start - pd.Timedelta(days=7)
    prev_end   = week_end   - pd.Timedelta(days=7)

    mask_curr = (df[DATE_COL] >= week_start) & (df[DATE_COL] <= week_end)
    mask_prev = (df[DATE_COL] >= prev_start) & (df[DATE_COL] <= prev_end)

    curr_median = df.loc[mask_curr, cam].median()
    prev_median = df.loc[mask_prev, cam].median()

    if pd.notna(prev_median) and prev_median != 0:
        change = (curr_median - prev_median) / abs(prev_median)
        pct_chg.loc[mask_curr] = change

    return pct_chg
