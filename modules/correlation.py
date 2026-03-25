"""
correlation.py
--------------
Computes the intra-week correlation index between cameras.

The goal is to distinguish two failure modes:
    (A) Camera problem   → one camera drops while others stay normal
    (B) Real flow drop   → all cameras drop together (correlated)

Logic
-----
For each analysis ISO week we compute:

    1. weekly_delta per camera
       Normalised deviation from its own baseline median for that week:
           delta_i = (observed_i - baseline_median_i) / baseline_median_i

       Using relative (%) deltas instead of raw residuals makes cameras
       with very different absolute volumes comparable.

    2. correlation_index  (scalar, range [-1, 1])
       Mean pairwise Pearson correlation of daily observed values across
       all camera pairs during the analysis week.
       High positive value → cameras moving together → likely real flow event.
       Low / negative value → cameras diverging → likely equipment issue.

    3. per-camera isolation_score  (range [0, 1])
       How much a single camera diverges from the group median delta.
       High isolation → that camera behaved differently from the rest.
           isolation_i = |delta_i - median(delta_all)| / (mad(delta_all) + ε)
       Normalised with MAD (robust to outliers from broken cameras).

Output columns added to the dataframe
--------------------------------------
    _corr_index          : float, scalar repeated for every row in the analysis week
    {cam}_isolation      : float, isolation score for the camera in the analysis week
                           NaN outside the analysis week
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

from loader import DATE_COL
from utils import last_complete_iso_week


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_correlation(
    df: pd.DataFrame,
    camera_cols: list[str],
) -> pd.DataFrame:
    """
    Compute intra-week correlation index and per-camera isolation scores.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with baseline columns attached (output of baseline.compute_baseline).
    camera_cols : list[str]
        Camera column names.

    Returns
    -------
    pd.DataFrame
        Input dataframe with added columns:
            _corr_index          : mean pairwise correlation in the analysis ISO week
            {cam}_isolation      : isolation score per camera
    """
    result = df.copy()

    # Identify analysis ISO week (last complete ISO week — same logic as signals.py)
    week_start, week_end = last_complete_iso_week(result)

    mask_week = (
        (result[DATE_COL] >= week_start) &
        (result[DATE_COL] <= week_end)
    )
    week_df = result.loc[mask_week, [DATE_COL] + camera_cols].copy()

    # ------------------------------------------------------------------
    # 1. Correlation index — mean pairwise Pearson over daily values
    # ------------------------------------------------------------------
    corr_index = _mean_pairwise_correlation(week_df, camera_cols)
    result["_corr_index"] = np.nan
    result.loc[mask_week, "_corr_index"] = corr_index

    # ------------------------------------------------------------------
    # 2. Weekly delta per camera (relative deviation from baseline median)
    # ------------------------------------------------------------------
    deltas: dict[str, float] = {}

    for cam in camera_cols:
        baseline_col = f"{cam}_baseline_median"

        if baseline_col not in result.columns:
            deltas[cam] = np.nan
            continue

        obs_median      = week_df[cam].median()
        baseline_median = result.loc[mask_week, baseline_col].median()

        if pd.isna(baseline_median) or baseline_median == 0:
            deltas[cam] = np.nan
        else:
            deltas[cam] = (obs_median - baseline_median) / abs(baseline_median)

    # ------------------------------------------------------------------
    # 3. Isolation score per camera
    #    How much each camera diverges from the group median delta
    # ------------------------------------------------------------------
    valid_deltas = np.array([v for v in deltas.values() if not np.isnan(v)])

    if len(valid_deltas) >= 2:
        group_median = float(np.median(valid_deltas))
        mad          = float(np.median(np.abs(valid_deltas - group_median))) + 1e-9
    else:
        group_median = np.nan
        mad          = 1.0

    for cam in camera_cols:
        result[f"{cam}_isolation"] = np.nan

        if pd.isna(deltas.get(cam, np.nan)) or pd.isna(group_median):
            continue

        isolation = abs(deltas[cam] - group_median) / mad
        # Cap at 10 to avoid extreme values from near-zero MAD edge cases
        isolation = min(float(isolation), 10.0)
        result.loc[mask_week, f"{cam}_isolation"] = isolation

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mean_pairwise_correlation(
    week_df: pd.DataFrame,
    camera_cols: list[str],
) -> float:
    """
    Compute the mean Pearson correlation across all camera pairs
    using daily observed values within the analysis week.

    Returns NaN if fewer than 2 cameras have valid data or if the
    week has fewer than 3 data points (correlation is unreliable).
    """
    # Drop cameras with all-NaN or constant values in this week
    valid_cols = [
        c for c in camera_cols
        if week_df[c].notna().sum() >= 3 and week_df[c].std() > 0
    ]

    if len(valid_cols) < 2:
        return np.nan

    pairs        = list(combinations(valid_cols, 2))
    correlations = []

    for c1, c2 in pairs:
        pair_df = week_df[[c1, c2]].dropna()
        if len(pair_df) < 3:
            continue
        r = float(pair_df[c1].corr(pair_df[c2]))
        if not np.isnan(r):
            correlations.append(r)

    return float(np.mean(correlations)) if correlations else np.nan
