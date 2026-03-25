"""
views/dashboard.py
------------------
Renders the global diagnostics section:
    - Total flow comparison (current week / previous week / same week LY)
    - 90-day trend slope for total
    - Camera health summary (counts by status)
    - Correlation index interpretation
    - Signal type breakdown
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))

from loader import DATE_COL, TOTAL_COL
from utils  import last_complete_iso_week, iso_week_mask
from classifier import STATUS_ORDER


# Status colour palette (matches report)
STATUS_COLORS = {
    "Saudável":  "#2ecc71",
    "Atenção":   "#f39c12",
    "Alerta":    "#e74c3c",
    "Crítico":   "#7d3c98",
    "Sem Dados": "#95a5a6",
}


def render_dashboard(df: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """
    Render the global diagnostics section.

    Parameters
    ----------
    df      : full processed dataframe
    ranking : one-row-per-camera summary from classifier.build_ranking
    """
    week_start, week_end = last_complete_iso_week(df)
    mask_curr            = iso_week_mask(df, week_start, week_end)

    prev_start = week_start - pd.Timedelta(days=7)
    prev_end   = week_end   - pd.Timedelta(days=7)
    mask_prev  = iso_week_mask(df, prev_start, prev_end)

    ly_start   = week_start - pd.DateOffset(years=1)
    ly_end     = week_end   - pd.DateOffset(years=1)
    mask_ly    = iso_week_mask(df, ly_start, ly_end)

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    total_curr = int(df.loc[mask_curr, TOTAL_COL].sum())
    total_prev = int(df.loc[mask_prev, TOTAL_COL].sum()) if mask_prev.any() else None
    total_ly   = int(df.loc[mask_ly,   TOTAL_COL].sum()) if mask_ly.any()   else None

    pct_prev = _pct_change(total_curr, total_prev)
    pct_ly   = _pct_change(total_curr, total_ly)

    iso_label = f"Semana {week_end.isocalendar().week:02d} / {week_end.isocalendar().year}"
    st.subheader(f"📊 Diagnóstico Global — {iso_label}")
    st.caption(f"{week_start.strftime('%d/%m/%Y')} → {week_end.strftime('%d/%m/%Y')}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fluxo Total — Semana Atual", f"{total_curr:,}")
    c2.metric(
        "Semana Anterior",
        f"{total_prev:,}" if total_prev else "—",
        delta=pct_prev,
        delta_color="normal",
    )
    c3.metric(
        "Mesma Semana Ano Anterior",
        f"{total_ly:,}" if total_ly else "—",
        delta=pct_ly,
        delta_color="normal",
    )

    # ------------------------------------------------------------------
    # 90-day slope for total
    # ------------------------------------------------------------------
    slope_col = f"{TOTAL_COL}_slope_90d" if f"{TOTAL_COL}_slope_90d" in df.columns else None
    if slope_col:
        slope_val = df.loc[mask_curr, slope_col].dropna()
        if not slope_val.empty:
            slope = slope_val.iloc[-1]
            direction = "📈" if slope > 0 else "📉"
            st.caption(
                f"Tendência 90 dias (total): {direction} "
                f"{slope:+.2f} visitantes/dia"
            )

    st.divider()

    # ------------------------------------------------------------------
    # Health summary
    # ------------------------------------------------------------------
    st.subheader("🩺 Resumo de Saúde das Câmeras")

    col_counts, col_chart = st.columns([1, 1])

    status_counts = (
        ranking["status"]
        .value_counts()
        .reindex(STATUS_ORDER, fill_value=0)
    )

    with col_counts:
        for status in STATUS_ORDER:
            count = int(status_counts.get(status, 0))
            color = STATUS_COLORS.get(status, "#95a5a6")
            st.markdown(
                f"<span style='color:{color}; font-weight:600'>● {status}</span>: "
                f"**{count}** câmera(s)",
                unsafe_allow_html=True,
            )

    with col_chart:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        labels  = [s for s in STATUS_ORDER if status_counts.get(s, 0) > 0]
        sizes   = [status_counts[s] for s in labels]
        colors  = [STATUS_COLORS[s] for s in labels]

        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax.set_title("Distribuição de Status", fontsize=10, pad=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.divider()

    # ------------------------------------------------------------------
    # Correlation index interpretation
    # ------------------------------------------------------------------
    st.subheader("🔗 Correlação entre Câmeras")

    corr_val = ranking["corr_index"].dropna()
    if not corr_val.empty:
        corr = float(corr_val.iloc[0])
        _render_corr_indicator(corr)
    else:
        st.caption("Correlação não disponível para esta semana.")

    st.divider()

    # ------------------------------------------------------------------
    # Signal type breakdown
    # ------------------------------------------------------------------
    st.subheader("🔍 Tipos de Sinal Detectados")

    signal_counts = (
        ranking
        .loc[ranking["status"] != "Saudável", "signal_type"]
        .value_counts()
    )

    if signal_counts.empty:
        st.success("Nenhum sinal relevante detectado nesta semana.")
    else:
        for signal, count in signal_counts.items():
            st.markdown(f"- **{signal}**: {count} câmera(s)")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pct_change(current: int, previous: int | None) -> str | None:
    if previous is None or previous == 0:
        return None
    pct = (current - previous) / abs(previous) * 100
    return f"{pct:+.1f}%"


def _render_corr_indicator(corr: float) -> None:
    """Render a simple colour-coded correlation indicator with interpretation."""
    if corr >= 0.7:
        color = "#e74c3c"
        label = "Alta correlação — queda provavelmente real (fluxo do shopping)"
    elif corr >= 0.3:
        color = "#f39c12"
        label = "Correlação moderada — investigar contexto"
    else:
        color = "#2ecc71"
        label = "Baixa correlação — câmeras divergindo (possível falha de equipamento)"

    st.markdown(
        f"<div style='padding:10px; border-left: 4px solid {color}; "
        f"background:#f9f9f9; border-radius:4px'>"
        f"<b>Índice de correlação:</b> {corr:.2f}<br>"
        f"<span style='color:{color}'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
