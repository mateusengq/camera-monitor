"""
views/detail.py
---------------
Renders the per-camera detail section, replicating the three-panel layout
from the original PDF report:

    Panel 1 — Observed vs Baseline (last 60 days)
    Panel 2 — Residuals vs baseline median (last 60 days)
    Panel 3 — Drift: EWMA + CUSUM (last 90 days)

Cameras are ordered by score ascending (worst first), matching the ranking.
Only cameras with status != "Saudável" are expanded by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))

from loader import DATE_COL
from utils  import last_complete_iso_week, iso_week_mask
from dashboard import STATUS_COLORS


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_detail(df: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """
    Render per-camera detail panels.

    Parameters
    ----------
    df      : full processed dataframe
    ranking : one-row-per-camera summary (sorted worst first)
    """
    st.subheader("🔬 Detalhamento — Câmeras Prioritárias")
    st.caption(
        "As linhas tracejadas verticais delimitam a semana analisada. "
        "Gráficos: (1) Observado vs Baseline, (2) Resíduos, (3) Drift (EWMA + CUSUM)."
    )

    week_start, week_end = last_complete_iso_week(df)

    for _, row in ranking.iterrows():
        cam    = str(row["camera"])
        status = str(row.get("status", "Sem Dados"))
        score  = row.get("score", np.nan)
        score_str = f"{score:.1f}" if not pd.isna(score) else "—"

        color  = STATUS_COLORS.get(status, "#95a5a6")
        label  = (
            f"**{cam.replace('_', ' ').title()}** — "
            f"Score: {score_str} | Status: :{_status_color(status)}[{status}] | "
            f"Tipo: {row.get('signal_type', '—')}"
        )

        # Non-healthy cameras expand by default
        expanded = status != "Saudável"

        with st.expander(label, expanded=expanded):
            _render_camera_detail(df, cam, week_start, week_end, row)


# ---------------------------------------------------------------------------
# Per-camera rendering
# ---------------------------------------------------------------------------

def _render_camera_detail(
    df: pd.DataFrame,
    cam: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    summary_row: pd.Series,
) -> None:
    """Render the three-panel chart + metadata for a single camera."""

    # Metadata strip
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score",         f"{summary_row.get('score', np.nan):.1f}" if not pd.isna(summary_row.get('score')) else "—")
    c2.metric("Var. Semana",   str(summary_row.get("week_pct_str", "N/A")))
    c3.metric("Zeros (sem.)",  int(summary_row.get("zeros_week", 0)) if not pd.isna(summary_row.get("zeros_week")) else 0)
    c4.metric("Zeros % 12m",   f"{float(summary_row['zeros_pct_12m']) * 100:.1f}%" if not pd.isna(summary_row.get("zeros_pct_12m")) else "—")
    c5.metric("Slope 90d",     f"{float(summary_row['slope_90d']):+.2f}" if not pd.isna(summary_row.get("slope_90d")) else "—")

    st.markdown("")

    # Filter to last 60 / 90 days for charts
    ref_date   = week_end
    df_60      = df[df[DATE_COL] >= ref_date - pd.Timedelta(days=60)].copy()
    df_90      = df[df[DATE_COL] >= ref_date - pd.Timedelta(days=90)].copy()

    required_cols = [cam, f"{cam}_baseline_median", f"{cam}_baseline_mean", f"{cam}_residual"]
    signal_cols   = [f"{cam}_ewma", f"{cam}_ewma_limit", f"{cam}_cusum_pos", f"{cam}_cusum_neg", f"{cam}_cusum_limit"]

    if not all(c in df.columns for c in required_cols):
        st.warning(f"Colunas de baseline não encontradas para '{cam}'.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.45)

    _plot_observed_vs_baseline(axes[0], df_60, cam, week_start, week_end)
    _plot_residuals(axes[1], df_60, cam, week_start, week_end)

    if all(c in df.columns for c in signal_cols):
        _plot_drift(axes[2], df_90, cam, week_start, week_end)
    else:
        axes[2].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart panels
# ---------------------------------------------------------------------------

def _plot_observed_vs_baseline(
    ax: plt.Axes,
    df: pd.DataFrame,
    cam: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> None:
    dates    = df[DATE_COL]
    observed = df[cam]
    b_median = df[f"{cam}_baseline_median"]
    b_mean   = df[f"{cam}_baseline_mean"]

    ax.plot(dates, observed, color="#3498db", lw=1.4, label="Observado")
    ax.plot(dates, b_mean,   color="#e67e22", lw=1.2, ls="--", label="Esperado (média DOW)")
    ax.plot(dates, b_median, color="#27ae60", lw=1.2, ls="--", label="Esperado (mediana DOW)")

    _add_week_vlines(ax, week_start, week_end)
    _fmt_date_axis(ax)

    ax.set_title(f"{cam} — Observado vs Baseline (últimos 60 dias)", fontsize=10)
    ax.set_ylabel("Fluxo")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def _plot_residuals(
    ax: plt.Axes,
    df: pd.DataFrame,
    cam: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> None:
    dates    = df[DATE_COL]
    residual = df[f"{cam}_residual"]

    ax.plot(dates, residual, color="#3498db", lw=1.2)
    ax.axhline(0, color="#7f8c8d", lw=0.8, ls="-")

    # Shade negative residuals
    ax.fill_between(dates, residual, 0,
                    where=(residual < 0), alpha=0.25, color="#e74c3c", label="Abaixo do esperado")
    ax.fill_between(dates, residual, 0,
                    where=(residual >= 0), alpha=0.15, color="#2ecc71", label="Acima do esperado")

    _add_week_vlines(ax, week_start, week_end)
    _fmt_date_axis(ax)

    ax.set_title(f"{cam} — Resíduos vs baseline mediana (últimos 60 dias)", fontsize=10)
    ax.set_ylabel("Resíduo")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def _plot_drift(
    ax: plt.Axes,
    df: pd.DataFrame,
    cam: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> None:
    dates      = df[DATE_COL]
    ewma       = df[f"{cam}_ewma"]
    ewma_lim   = float(df[f"{cam}_ewma_limit"].iloc[-1])
    cusum_pos  = df[f"{cam}_cusum_pos"]
    cusum_neg  = df[f"{cam}_cusum_neg"]
    cusum_lim  = float(df[f"{cam}_cusum_limit"].iloc[-1])

    # EWMA on left axis
    color_ewma = "#3498db"
    ax.plot(dates, ewma, color=color_ewma, lw=1.3, label="EWMA (resíduo)")
    ax.axhline( ewma_lim, color=color_ewma, lw=0.8, ls="--", alpha=0.7, label="Limite EWMA")
    ax.axhline(-ewma_lim, color=color_ewma, lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel("EWMA", color=color_ewma)
    ax.tick_params(axis="y", labelcolor=color_ewma)

    # CUSUM on right axis
    ax2 = ax.twinx()
    ax2.plot(dates, cusum_pos, color="#e74c3c", lw=1.1, label="CUSUM +")
    ax2.plot(dates, cusum_neg, color="#e67e22", lw=1.1, label="CUSUM - (abs)")
    ax2.axhline(cusum_lim, color="#e74c3c", lw=0.8, ls="--", alpha=0.6, label="Limite CUSUM")
    ax2.set_ylabel("CUSUM", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    _add_week_vlines(ax, week_start, week_end)
    _fmt_date_axis(ax)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    ax.set_title(f"{cam} — Drift (EWMA + CUSUM) — últimos 90 dias", fontsize=10)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Shared chart helpers
# ---------------------------------------------------------------------------

def _add_week_vlines(
    ax: plt.Axes,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> None:
    for d in [week_start, week_end]:
        ax.axvline(d, color="#2c3e50", lw=0.9, ls="--", alpha=0.6)


def _fmt_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)


def _status_color(status: str) -> str:
    """Map status to Streamlit markdown color name."""
    mapping = {
        "Saudável": "green",
        "Atenção":  "orange",
        "Alerta":   "red",
        "Crítico":  "violet",
    }
    return mapping.get(status, "gray")
