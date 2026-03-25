"""
views/sidebar.py
----------------
Renders all configurable parameters in the Streamlit sidebar.

Returns a single `params` dict consumed by every pipeline module,
keeping app.py clean and sidebar logic isolated.

Params dict schema
------------------
{
    "lookback_months":      int,    # 3 | 6 | 12
    "decay":                float,  # baseline recency decay
    "ewma_lambda":          float,
    "cusum_k":              float,
    "cusum_h":              float,
    "drop_alert_threshold": float,
    "weights": {
        "cusum":     float,
        "ewma":      float,
        "drop":      float,
        "cv":        float,
        "isolation": float,
    }
}
"""

from __future__ import annotations

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import defaults as D


def render_sidebar() -> dict:
    """
    Render all sidebar controls and return the params dict.
    Call this once at the top of app.py.
    """
    st.sidebar.title("⚙️ Parâmetros")

    # ------------------------------------------------------------------
    # 1. Dados
    # ------------------------------------------------------------------
    st.sidebar.header("Dados")

    lookback_months = st.sidebar.selectbox(
        label="Janela histórica",
        options=[3, 6, 12],
        index=[3, 6, 12].index(D.BASELINE_LOOKBACK_MONTHS),
        format_func=lambda x: f"{x} meses",
        help="Período usado para calcular o baseline de cada câmera.",
    )

    # ------------------------------------------------------------------
    # 2. Baseline
    # ------------------------------------------------------------------
    st.sidebar.header("Baseline")

    decay = st.sidebar.slider(
        label="Fator de recência (decay)",
        min_value=0.0,
        max_value=1.0,
        value=D.BASELINE_DECAY,
        step=0.05,
        help=(
            "Controla o peso das semanas mais recentes no baseline. "
            "0 = pesos iguais para todo o período; "
            "1 = semanas recentes têm peso máximo."
        ),
    )

    # ------------------------------------------------------------------
    # 3. EWMA
    # ------------------------------------------------------------------
    st.sidebar.header("EWMA")

    ewma_lambda = st.sidebar.slider(
        label="λ (lambda)",
        min_value=0.05,
        max_value=0.50,
        value=D.EWMA_LAMBDA,
        step=0.05,
        help=(
            "Fator de suavização do EWMA. "
            "Valores altos = mais peso nos resíduos recentes (mais sensível). "
            "Valores baixos = mais inércia (menos alarmes falsos)."
        ),
    )

    # ------------------------------------------------------------------
    # 4. CUSUM
    # ------------------------------------------------------------------
    st.sidebar.header("CUSUM")

    cusum_k = st.sidebar.slider(
        label="k (slack)",
        min_value=0.5,
        max_value=3.0,
        value=D.CUSUM_K,
        step=0.1,
        help=(
            "Tolerância do CUSUM antes de acumular desvio. "
            "Valores altos = menos sensível a pequenas quedas."
        ),
    )

    cusum_h = st.sidebar.slider(
        label="h (threshold)",
        min_value=2.0,
        max_value=10.0,
        value=D.CUSUM_H,
        step=0.5,
        help=(
            "Limiar de disparo do alarme CUSUM. "
            "Valores altos = alarme só dispara em desvios persistentes grandes."
        ),
    )

    # ------------------------------------------------------------------
    # 5. Score — pesos por dimensão
    # ------------------------------------------------------------------
    st.sidebar.header("Pesos do Score")

    with st.sidebar.expander("Ajustar pesos", expanded=False):
        w_cusum = st.slider(
            "Peso CUSUM",
            0.0, 1.0, D.WEIGHT_CUSUM, 0.1,
            help="Contribuição do alarme CUSUM no score composto.",
        )
        w_ewma = st.slider(
            "Peso EWMA",
            0.0, 1.0, D.WEIGHT_EWMA, 0.1,
            help="Contribuição do alarme EWMA no score composto.",
        )
        w_drop = st.slider(
            "Peso Queda Semanal",
            0.0, 1.0, D.WEIGHT_DROP, 0.1,
            help="Contribuição da queda % semana vs. semana anterior.",
        )
        w_cv = st.slider(
            "Peso Volatilidade (CV)",
            0.0, 1.0, D.WEIGHT_CV, 0.1,
            help="Contribuição da volatilidade histórica (CV + zeros %).",
        )
        w_isolation = st.slider(
            "Peso Isolamento",
            0.0, 1.0, D.WEIGHT_ISOLATION, 0.1,
            help="Contribuição do isolamento da câmera vs. grupo.",
        )

    # ------------------------------------------------------------------
    # 6. Alertas
    # ------------------------------------------------------------------
    st.sidebar.header("Alertas")

    drop_threshold_pct = st.sidebar.slider(
        label="Threshold de queda semanal (%)",
        min_value=5,
        max_value=50,
        value=int(D.DROP_ALERT_THRESHOLD * 100),
        step=5,
        format="%d%%",
        help=(
            "Queda percentual semanal que dispara penalidade máxima no score. "
            "Ex: 20% significa que uma queda de 20% ou mais gera penalidade total."
        ),
    )

    # ------------------------------------------------------------------
    # Assemble and return
    # ------------------------------------------------------------------
    return {
        "lookback_months":      int(lookback_months),
        "decay":                float(decay),
        "ewma_lambda":          float(ewma_lambda),
        "cusum_k":              float(cusum_k),
        "cusum_h":              float(cusum_h),
        "drop_alert_threshold": float(drop_threshold_pct) / 100.0,
        "weights": {
            "cusum":     float(w_cusum),
            "ewma":      float(w_ewma),
            "drop":      float(w_drop),
            "cv":        float(w_cv),
            "isolation": float(w_isolation),
        },
    }
