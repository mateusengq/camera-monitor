"""
scorer.py
---------
Computes a composite health score (0–100) for each camera in the analysis
ISO week, combining all analytical signals.

Score logic
-----------
The score starts at 100 and accumulates penalties from five independent
dimensions. Each dimension has a configurable weight (0–1) that scales
its maximum penalty contribution.

    Dimension               Max penalty     Signal used
    ─────────────────────── ─────────────── ──────────────────────────────
    1. CUSUM alarm          25 pts          cusum_alarm (bool)
    2. EWMA alarm           20 pts          ewma_alarm (bool)
    3. Weekly drop          20 pts          week_pct_chg  (continuous)
    4. CV (volatility)      20 pts          zeros_pct_12m + CV proxy
    5. Isolation            15 pts          isolation score

    Total max penalty: 100 pts → score floor = 0

Penalty scaling
---------------
- CUSUM / EWMA: binary (alarm fired = full penalty, else 0)
- Weekly drop:  linear from 0% (no penalty) to drop_alert_threshold (full penalty)
- CV:           linear from CV=0 (no penalty) to CV=1.5 (full penalty)
                combined with zeros_pct: penalty = max(cv_penalty, zeros_penalty)
- Isolation:    linear from 0 (no penalty) to isolation=3 (full penalty)

All penalties are capped at their maximum before applying weights.

Output columns
--------------
    {cam}_score              : float [0, 100]
    {cam}_penalty_cusum      : float contribution from CUSUM
    {cam}_penalty_ewma       : float contribution from EWMA
    {cam}_penalty_drop       : float contribution from weekly drop
    {cam}_penalty_cv         : float contribution from CV / zeros
    {cam}_penalty_isolation  : float contribution from isolation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loader import DATE_COL
from utils import last_complete_iso_week, week_bool, week_scalar


# ---------------------------------------------------------------------------
# Defaults (overridable via sidebar)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "cusum":     1.0,   # weight for CUSUM penalty
    "ewma":      1.0,   # weight for EWMA penalty
    "drop":      1.0,   # weight for weekly drop penalty
    "cv":        1.0,   # weight for CV / zeros penalty
    "isolation": 1.0,   # weight for isolation penalty
}

DEFAULT_DROP_THRESHOLD = 0.20   # 20% drop triggers full drop penalty


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_scores(
    df: pd.DataFrame,
    camera_cols: list[str],
    weights: dict[str, float] | None = None,
    drop_alert_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute composite health scores for each camera.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all signal columns attached.
    camera_cols : list[str]
        Camera column names.
    weights : dict, optional
        Override default penalty weights. Keys: cusum, ewma, drop, cv, isolation.
        Values in [0, 1].
    drop_alert_threshold : float
        Fractional drop (e.g. 0.20 = 20%) that maps to full drop penalty.

    Returns
    -------
    pd.DataFrame
        Input dataframe with score and penalty breakdown columns appended.
    """
    w      = {**DEFAULT_WEIGHTS, **(weights or {})}
    result = df.copy()

    week_start, week_end = last_complete_iso_week(result)
    mask_week = (
        (result[DATE_COL] >= week_start) &
        (result[DATE_COL] <= week_end)
    )

    for cam in camera_cols:
        # ------------------------------------------------------------------
        # Extract scalar signals for the analysis week
        # ------------------------------------------------------------------
        cusum_alarm = week_bool(result, mask_week, f"{cam}_cusum_alarm")
        ewma_alarm  = week_bool(result, mask_week, f"{cam}_ewma_alarm")
        week_drop   = week_scalar(result, mask_week, f"{cam}_week_pct_chg")
        zeros_pct   = week_scalar(result, mask_week, f"{cam}_zeros_pct_12m")
        isolation   = week_scalar(result, mask_week, f"{cam}_isolation")

        # CV proxy: use the coefficient of variation of the camera over the
        # last 12 months as a volatility measure
        cv_12m = _compute_cv_12m(result, cam, week_end)

        # ------------------------------------------------------------------
        # Penalties (0 → max_penalty before weight scaling)
        # ------------------------------------------------------------------

        # 1. CUSUM (binary, max 25)
        p_cusum = 25.0 if cusum_alarm else 0.0

        # 2. EWMA (binary, max 20)
        p_ewma = 20.0 if ewma_alarm else 0.0

        # 3. Weekly drop (linear, max 20)
        #    Only fires on negative drops; ignore positive changes
        if pd.isna(week_drop) or week_drop >= 0:
            p_drop = 0.0
        else:
            drop_ratio = min(abs(week_drop) / drop_alert_threshold, 1.0)
            p_drop     = 20.0 * drop_ratio

        # 4. CV / zeros (linear, max 20)
        #    Take the worse of the two signals
        p_cv_raw    = min(cv_12m / 1.5, 1.0) * 20.0 if not np.isnan(cv_12m) else 0.0
        p_zeros_raw = min(zeros_pct / 0.30, 1.0) * 20.0 if not np.isnan(zeros_pct) else 0.0
        p_cv        = max(p_cv_raw, p_zeros_raw)

        # 5. Isolation (linear, max 15)
        if pd.isna(isolation):
            p_isolation = 0.0
        else:
            p_isolation = min(isolation / 3.0, 1.0) * 15.0

        # ------------------------------------------------------------------
        # Apply weights and sum penalties
        # ------------------------------------------------------------------
        total_penalty = (
            p_cusum     * w["cusum"]     +
            p_ewma      * w["ewma"]      +
            p_drop      * w["drop"]      +
            p_cv        * w["cv"]        +
            p_isolation * w["isolation"]
        )

        # Normalise: weights can inflate total beyond 100; cap at 100
        max_possible = (
            25.0 * w["cusum"]     +
            20.0 * w["ewma"]      +
            20.0 * w["drop"]      +
            20.0 * w["cv"]        +
            15.0 * w["isolation"]
        )
        if max_possible > 0:
            total_penalty = (total_penalty / max_possible) * 100.0

        score = max(0.0, 100.0 - total_penalty)

        # ------------------------------------------------------------------
        # Write results — scalar values broadcast to all rows in analysis week
        # ------------------------------------------------------------------
        result[f"{cam}_score"]             = np.nan
        result[f"{cam}_penalty_cusum"]     = np.nan
        result[f"{cam}_penalty_ewma"]      = np.nan
        result[f"{cam}_penalty_drop"]      = np.nan
        result[f"{cam}_penalty_cv"]        = np.nan
        result[f"{cam}_penalty_isolation"] = np.nan

        result.loc[mask_week, f"{cam}_score"]             = round(score, 1)
        result.loc[mask_week, f"{cam}_penalty_cusum"]     = round(p_cusum     * w["cusum"],     2)
        result.loc[mask_week, f"{cam}_penalty_ewma"]      = round(p_ewma      * w["ewma"],      2)
        result.loc[mask_week, f"{cam}_penalty_drop"]      = round(p_drop      * w["drop"],      2)
        result.loc[mask_week, f"{cam}_penalty_cv"]        = round(p_cv        * w["cv"],        2)
        result.loc[mask_week, f"{cam}_penalty_isolation"] = round(p_isolation * w["isolation"], 2)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_cv_12m(
    df: pd.DataFrame,
    cam: str,
    reference_date: pd.Timestamp,
) -> float:
    """
    Coefficient of variation over the last 12 months, excluding event ISO weeks.
    CV = std / mean (only for non-zero, non-event rows).
    """
    cutoff = reference_date - pd.DateOffset(months=12)
    mask   = (
        (df[DATE_COL]               >= cutoff)          &
        (~df["_iso_week_has_event"])                     &
        (df[cam]                    >  0)
    )
    vals = df.loc[mask, cam].dropna()

    if len(vals) < 4:
        return np.nan

    mean = float(vals.mean())
    std  = float(vals.std())

    return (std / mean) if mean > 0 else np.nan
