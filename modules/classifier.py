"""
classifier.py
-------------
Maps the composite score to a health status and assigns a signal type label
for each camera in the analysis ISO week.

Status thresholds (score-based)
--------------------------------
    Score       Status
    ─────────── ──────────
    90 – 100    Saudável
    75 –  89    Atenção
    50 –  74    Alerta
     0 –  49    Crítico

The thresholds are fixed — they are not configurable, as they represent
operational severity levels agreed upon with stakeholders.

Signal type label
-----------------
The label describes *why* the camera is flagged, derived from which
penalty dimension dominated the score reduction:

    Label                       Condition
    ─────────────────────────── ──────────────────────────────────────────
    Falha Operacional           zeros_week > 0 AND cusum_alarm
    Drift / Mudança de Regime   cusum_alarm AND NOT zeros_week
    Queda Abrupta               drop penalty is dominant AND drop < threshold
    Alta Volatilidade           cv penalty is dominant
    Isolamento de Câmera        isolation penalty is dominant
    Sem Sinal Relevante         no alarm fired (score >= 90 or no dominant signal)

Output columns
--------------
    {cam}_status        : str  — Saudável / Atenção / Alerta / Crítico
    {cam}_signal_type   : str  — signal label
    {cam}_week_pct_str  : str  — formatted % change string for display (e.g. "-5.2%")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loader import DATE_COL
from utils import last_complete_iso_week, week_bool, week_scalar


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_THRESHOLDS = [
    (90, "Saudável"),
    (75, "Atenção"),
    (50, "Alerta"),
    (0,  "Crítico"),
]

STATUS_ORDER = ["Saudável", "Atenção", "Alerta", "Crítico"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_classification(
    df: pd.DataFrame,
    camera_cols: list[str],
) -> pd.DataFrame:
    """
    Assign health status and signal type label to each camera.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with score and penalty columns attached.
    camera_cols : list[str]
        Camera column names.

    Returns
    -------
    pd.DataFrame
        Input dataframe with classification columns appended.
    """
    result = df.copy()

    week_start, week_end = last_complete_iso_week(result)
    mask_week = (
        (result[DATE_COL] >= week_start) &
        (result[DATE_COL] <= week_end)
    )

    for cam in camera_cols:
        score       = week_scalar(result, mask_week, f"{cam}_score")
        zeros_week  = week_scalar(result, mask_week, f"{cam}_zeros_week")
        cusum_alarm = week_bool(result,   mask_week, f"{cam}_cusum_alarm")
        week_drop   = week_scalar(result, mask_week, f"{cam}_week_pct_chg")

        # Penalty breakdown for dominant signal detection
        p_cusum     = week_scalar(result, mask_week, f"{cam}_penalty_cusum")
        p_ewma      = week_scalar(result, mask_week, f"{cam}_penalty_ewma")
        p_drop      = week_scalar(result, mask_week, f"{cam}_penalty_drop")
        p_cv        = week_scalar(result, mask_week, f"{cam}_penalty_cv")
        p_isolation = week_scalar(result, mask_week, f"{cam}_penalty_isolation")

        # ------------------------------------------------------------------
        # Status
        # ------------------------------------------------------------------
        status = _score_to_status(score)

        # ------------------------------------------------------------------
        # Signal type
        # ------------------------------------------------------------------
        signal_type = _derive_signal_type(
            score        = score,
            zeros_week   = zeros_week,
            cusum_alarm  = cusum_alarm,
            week_drop    = week_drop,
            p_cusum      = p_cusum,
            p_ewma       = p_ewma,
            p_drop       = p_drop,
            p_cv         = p_cv,
            p_isolation  = p_isolation,
        )

        # ------------------------------------------------------------------
        # Formatted % change string
        # ------------------------------------------------------------------
        if pd.isna(week_drop):
            pct_str = "N/A"
        else:
            sign    = "+" if week_drop > 0 else ""
            pct_str = f"{sign}{week_drop * 100:.1f}%"

        # ------------------------------------------------------------------
        # Write to dataframe
        # ------------------------------------------------------------------
        result[f"{cam}_status"]       = np.nan
        result[f"{cam}_signal_type"]  = np.nan
        result[f"{cam}_week_pct_str"] = np.nan

        result.loc[mask_week, f"{cam}_status"]       = status
        result.loc[mask_week, f"{cam}_signal_type"]  = signal_type
        result.loc[mask_week, f"{cam}_week_pct_str"] = pct_str

    return result


def build_ranking(
    df: pd.DataFrame,
    camera_cols: list[str],
) -> pd.DataFrame:
    """
    Build a summary ranking DataFrame with one row per camera,
    sorted by score ascending (worst first).

    Columns returned
    ----------------
    camera, score, status, signal_type, week_pct_chg, week_pct_str,
    zeros_week, zeros_pct_12m, cusum_alarm, ewma_alarm,
    slope_90d, isolation, corr_index
    """
    week_start, week_end = last_complete_iso_week(df)
    mask_week = (
        (df[DATE_COL] >= week_start) &
        (df[DATE_COL] <= week_end)
    )

    rows = []
    for cam in camera_cols:
        row = {"camera": cam}

        for col_suffix in [
            "score", "status", "signal_type", "week_pct_chg", "week_pct_str",
            "zeros_week", "zeros_pct_12m", "cusum_alarm", "ewma_alarm",
            "slope_90d", "isolation",
        ]:
            full_col = f"{cam}_{col_suffix}"
            if full_col in df.columns:
                val = df.loc[mask_week, full_col].dropna()
                row[col_suffix] = val.iloc[0] if len(val) > 0 else np.nan
            else:
                row[col_suffix] = np.nan

        # corr_index is not per-camera — same for all
        if "_corr_index" in df.columns:
            val = df.loc[mask_week, "_corr_index"].dropna()
            row["corr_index"] = float(val.iloc[0]) if len(val) > 0 else np.nan
        else:
            row["corr_index"] = np.nan

        rows.append(row)

    ranking = pd.DataFrame(rows)

    # Sort: worst score first
    if "score" in ranking.columns:
        ranking = ranking.sort_values("score", ascending=True).reset_index(drop=True)

    return ranking


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _score_to_status(score: float) -> str:
    if pd.isna(score):
        return "Sem Dados"
    for threshold, label in STATUS_THRESHOLDS:
        if score >= threshold:
            return label
    return "Crítico"


def _derive_signal_type(
    score: float,
    zeros_week: float,
    cusum_alarm: bool,
    week_drop: float,
    p_cusum: float,
    p_ewma: float,
    p_drop: float,
    p_cv: float,
    p_isolation: float,
) -> str:
    # No meaningful signal
    if pd.isna(score) or score >= 90:
        return "Sem Sinal Relevante"

    # Operational failure: zeros recorded AND CUSUM fired
    if (not pd.isna(zeros_week)) and zeros_week > 0 and cusum_alarm:
        return "Falha Operacional"

    # Find dominant penalty dimension
    penalties = {
        "cusum":     p_cusum     if not pd.isna(p_cusum)     else 0.0,
        "ewma":      p_ewma      if not pd.isna(p_ewma)      else 0.0,
        "drop":      p_drop      if not pd.isna(p_drop)      else 0.0,
        "cv":        p_cv        if not pd.isna(p_cv)        else 0.0,
        "isolation": p_isolation if not pd.isna(p_isolation) else 0.0,
    }

    dominant = max(penalties, key=penalties.get)

    if dominant in ("cusum", "ewma"):
        return "Drift / Mudança de Regime"
    if dominant == "drop":
        return "Queda Abrupta"
    if dominant == "cv":
        return "Alta Volatilidade"
    if dominant == "isolation":
        return "Isolamento de Câmera"

    return "Sem Sinal Relevante"
