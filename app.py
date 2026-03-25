"""
app.py
------
Streamlit entry point.

Run with:
    streamlit run app.py

Pipeline order:
    upload → sidebar params → loader → baseline → signals →
    correlation → scorer → classifier → views
"""

from __future__ import annotations

import sys
import os

# Make modules and views importable regardless of run location
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "modules"))
sys.path.insert(0, os.path.join(ROOT, "views"))
sys.path.insert(0, os.path.join(ROOT, "config"))

import streamlit as st

# Pipeline modules
from loader      import load_csv
from baseline    import compute_baseline
from signals     import compute_signals
from correlation import compute_correlation
from scorer      import compute_scores
from classifier  import compute_classification, build_ranking

# Views
from sidebar   import render_sidebar
from dashboard import render_dashboard
from ranking   import render_ranking
from detail    import render_detail


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Monitor de Câmeras",
    page_icon="📷",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📷 Monitoramento de Câmeras")
st.caption(
    "Faça upload do CSV com os dados de fluxo. "
    "O sistema detecta automaticamente a última semana ISO completa."
)


# ---------------------------------------------------------------------------
# Sidebar — params (rendered before upload so sliders are always visible)
# ---------------------------------------------------------------------------

params = render_sidebar()


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    label="Upload do CSV de fluxo",
    type=["csv"],
    help="Colunas esperadas: data, <câmeras...>, total, evento (opcional).",
)

if uploaded is None:
    st.info("⬆️ Faça upload de um arquivo CSV para começar.")
    st.stop()


# ---------------------------------------------------------------------------
# Pipeline — cached by file content + params to avoid recomputing on every
# widget interaction. Cache key uses file name + size + param values.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Processando dados...")
def run_pipeline(file_bytes: bytes, file_name: str, params: dict):
    """
    Run the full analytical pipeline.
    Cached: only recomputes when file or params change.
    """
    import io

    result = load_csv(io.BytesIO(file_bytes))

    if not result.ok:
        return None, None, result.errors, result.warnings, []

    df           = result.df
    camera_cols  = result.camera_cols

    # 1. Baseline
    df = compute_baseline(
        df,
        camera_cols,
        lookback_months=params["lookback_months"],
        decay=params["decay"],
    )

    # 2. Signals
    df = compute_signals(
        df,
        camera_cols,
        ewma_lambda=params["ewma_lambda"],
        cusum_k=params["cusum_k"],
        cusum_h=params["cusum_h"],
    )

    # 3. Correlation
    df = compute_correlation(df, camera_cols)

    # 4. Score
    df = compute_scores(
        df,
        camera_cols,
        weights=params["weights"],
        drop_alert_threshold=params["drop_alert_threshold"],
    )

    # 5. Classification
    df = compute_classification(df, camera_cols)

    # 6. Ranking
    ranking = build_ranking(df, camera_cols)

    return df, ranking, [], result.warnings, camera_cols


# Run pipeline
file_bytes = uploaded.read()
df, ranking, errors, warnings, camera_cols = run_pipeline(
    file_bytes, uploaded.name, params
)


# ---------------------------------------------------------------------------
# Surface errors and warnings
# ---------------------------------------------------------------------------

for err in errors:
    st.error(f"❌ {err}")

for warn in warnings:
    st.warning(f"⚠️ {warn}")

if errors:
    st.stop()


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

tab_dashboard, tab_ranking, tab_detail = st.tabs(
    ["📊 Diagnóstico Global", "📋 Ranking", "🔬 Detalhamento"]
)

with tab_dashboard:
    render_dashboard(df, ranking)

with tab_ranking:
    render_ranking(ranking)

with tab_detail:
    render_detail(df, ranking)


# ---------------------------------------------------------------------------
# PDF Export — shown at the bottom after all views
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📄 Exportar Relatório")

shopping_name = st.text_input(
    "Nome do shopping (para o relatório)",
    value=uploaded.name.replace(".csv", "").replace("_", " ").title(),
)

if st.button("Gerar PDF", type="primary"):
    from exporter import generate_pdf

    with st.spinner("Gerando PDF..."):
        pdf_bytes = generate_pdf(
            df=df,
            ranking=ranking,
            camera_cols=camera_cols,
            shopping_name=shopping_name,
        )

    week_start, week_end = __import__("utils").last_complete_iso_week(df)
    iso_week = week_end.isocalendar().week
    iso_year = week_end.isocalendar().year
    file_name = f"relatorio_{shopping_name.lower().replace(' ', '_')}_{iso_year}-W{iso_week:02d}.pdf"

    st.download_button(
        label="⬇️ Baixar PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )
