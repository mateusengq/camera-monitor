"""
views/ranking.py
----------------
Renders the camera health ranking table, sorted worst → best score.
Includes status badge, signal type, score, week % change, zeros, slope,
and CUSUM / EWMA alarm indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard import STATUS_COLORS


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_ranking(ranking: pd.DataFrame) -> None:
    """
    Render the camera ranking section.

    Parameters
    ----------
    ranking : pd.DataFrame
        One-row-per-camera summary from classifier.build_ranking,
        sorted worst score first.
    """
    st.subheader("📋 Ranking de Saúde das Câmeras")
    st.caption("Ordenado do pior para o melhor score. Clique no cabeçalho para reordenar.")

    if ranking.empty:
        st.warning("Nenhuma câmera encontrada.")
        return

    display = _build_display_df(ranking)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=_column_config(),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_display_df(ranking: pd.DataFrame) -> pd.DataFrame:
    """Reshape and format ranking into a display-ready DataFrame."""
    rows = []

    for _, row in ranking.iterrows():
        status  = str(row.get("status", "Sem Dados"))
        score   = row.get("score", np.nan)
        cam     = str(row.get("camera", ""))

        rows.append({
            "Status":         _status_badge(status),
            "Câmera":         _fmt_camera_name(cam),
            "Score":          round(float(score), 1) if not pd.isna(score) else None,
            "Tipo de Sinal":  str(row.get("signal_type", "—")),
            "Var. Semana":    str(row.get("week_pct_str", "N/A")),
            "Zeros (sem.)":   int(row["zeros_week"]) if not pd.isna(row.get("zeros_week")) else 0,
            "Zeros % (12m)":  f"{float(row['zeros_pct_12m']) * 100:.1f}%" if not pd.isna(row.get("zeros_pct_12m")) else "—",
            "Slope 90d":      f"{float(row['slope_90d']):+.2f}" if not pd.isna(row.get("slope_90d")) else "—",
            "CUSUM ⚠":        "🔴" if row.get("cusum_alarm") else "🟢",
            "EWMA ⚠":         "🔴" if row.get("ewma_alarm")  else "🟢",
            "Isolamento":     f"{float(row['isolation']):.2f}" if not pd.isna(row.get("isolation")) else "—",
        })

    return pd.DataFrame(rows)


def _status_badge(status: str) -> str:
    icons = {
        "Saudável":  "🟢 Saudável",
        "Atenção":   "🟡 Atenção",
        "Alerta":    "🔴 Alerta",
        "Crítico":   "🟣 Crítico",
        "Sem Dados": "⚪ Sem Dados",
    }
    return icons.get(status, status)


def _fmt_camera_name(name: str) -> str:
    """Title-case camera name for display."""
    return name.replace("_", " ").title()


def _column_config() -> dict:
    return {
        "Score": st.column_config.ProgressColumn(
            "Score",
            help="Score de saúde (0 = crítico, 100 = perfeito)",
            min_value=0,
            max_value=100,
            format="%.1f",
        ),
        "Var. Semana": st.column_config.TextColumn(
            "Var. Semana",
            help="Variação % da semana atual vs. semana anterior (inclui semanas de evento)",
        ),
        "CUSUM ⚠": st.column_config.TextColumn(
            "CUSUM",
            help="🔴 = alarme CUSUM disparado na semana",
            width="small",
        ),
        "EWMA ⚠": st.column_config.TextColumn(
            "EWMA",
            help="🔴 = alarme EWMA disparado na semana",
            width="small",
        ),
    }
