"""
exporter.py
-----------
Generates a PDF report replicating the original layout:

    Page 1  — Cover: shopping name, ISO week, global diagnostics, health summary
    Page 2  — Camera ranking table
    Pages 3+ — Per-camera detail (3 charts per camera, worst-first order)

Uses matplotlib with PdfPages — no external PDF library required.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

from loader     import DATE_COL, TOTAL_COL
from utils      import last_complete_iso_week, iso_week_mask
from classifier import STATUS_ORDER


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

STATUS_COLORS = {
    "Saudável":  "#2ecc71",
    "Atenção":   "#f39c12",
    "Alerta":    "#e74c3c",
    "Crítico":   "#7d3c98",
    "Sem Dados": "#95a5a6",
}

BRAND_BLUE  = "#2c3e50"
BRAND_LIGHT = "#ecf0f1"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_pdf(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    camera_cols: list[str],
    shopping_name: str = "Shopping",
) -> bytes:
    """
    Generate the full PDF report and return it as bytes.

    Parameters
    ----------
    df            : full processed dataframe
    ranking       : one-row-per-camera summary from classifier.build_ranking
    camera_cols   : list of camera column names
    shopping_name : display name used in the report header

    Returns
    -------
    bytes
        Raw PDF bytes — pass directly to st.download_button.
    """
    buf = io.BytesIO()

    week_start, week_end = last_complete_iso_week(df)
    iso_year  = week_end.isocalendar().year
    iso_week  = week_end.isocalendar().week
    week_label = f"{week_start.strftime('%Y-%m-%d')} → {week_end.strftime('%Y-%m-%d')}"

    with PdfPages(buf) as pdf:
        _page_cover(pdf, df, ranking, shopping_name, iso_year, iso_week, week_label, week_start, week_end)
        _page_ranking(pdf, ranking, shopping_name, iso_year, iso_week)
        for _, row in ranking.iterrows():
            cam = str(row["camera"])
            if cam in camera_cols:
                _page_camera(pdf, df, row, cam, week_start, week_end, shopping_name, iso_year, iso_week)

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _page_cover(
    pdf: PdfPages,
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    shopping_name: str,
    iso_year: int,
    iso_week: int,
    week_label: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")

    # ── Header band ──────────────────────────────────────────────────────
    ax_hdr = fig.add_axes([0, 0.88, 1, 0.12])
    ax_hdr.set_facecolor(BRAND_BLUE)
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.65, "RELATÓRIO EXECUTIVO — MONITORAMENTO DE CÂMERAS",
                ha="center", va="center", color="white", fontsize=13, fontweight="bold",
                transform=ax_hdr.transAxes)
    ax_hdr.text(0.5, 0.25, f"Shopping: {shopping_name}",
                ha="center", va="center", color="#bdc3c7", fontsize=10,
                transform=ax_hdr.transAxes)

    # ── Week info ─────────────────────────────────────────────────────────
    ax_week = fig.add_axes([0.05, 0.80, 0.90, 0.07])
    ax_week.axis("off")
    ax_week.text(0, 0.7, f"Semana analisada: {week_label}",
                 fontsize=9, color=BRAND_BLUE)
    ax_week.text(0, 0.2, f"Semana ISO {iso_week:02d} / {iso_year}",
                 fontsize=9, color="#7f8c8d")

    # ── Global flow metrics ───────────────────────────────────────────────
    mask_curr = iso_week_mask(df, week_start, week_end)
    prev_s    = week_start - pd.Timedelta(days=7)
    prev_e    = week_end   - pd.Timedelta(days=7)
    mask_prev = iso_week_mask(df, prev_s, prev_e)
    ly_s      = week_start - pd.DateOffset(years=1)
    ly_e      = week_end   - pd.DateOffset(years=1)
    mask_ly   = iso_week_mask(df, ly_s, ly_e)

    t_curr = int(df.loc[mask_curr, TOTAL_COL].sum())
    t_prev = int(df.loc[mask_prev, TOTAL_COL].sum()) if mask_prev.any() else None
    t_ly   = int(df.loc[mask_ly,   TOTAL_COL].sum()) if mask_ly.any()   else None

    pct_prev = _pct_str(t_curr, t_prev)
    pct_ly   = _pct_str(t_curr, t_ly)

    metrics = [
        ("Fluxo Total — Semana Atual", f"{t_curr:,}",       ""),
        ("Semana Anterior",            f"{t_prev:,}" if t_prev else "—", pct_prev),
        ("Mesma Semana Ano Anterior",  f"{t_ly:,}"   if t_ly   else "—", pct_ly),
    ]

    for i, (label, value, delta) in enumerate(metrics):
        x = 0.05 + i * 0.32
        ax_m = fig.add_axes([x, 0.66, 0.28, 0.12])
        ax_m.set_facecolor(BRAND_LIGHT)
        ax_m.axis("off")
        ax_m.text(0.5, 0.80, label,  ha="center", fontsize=7.5, color="#555")
        ax_m.text(0.5, 0.45, value,  ha="center", fontsize=13, fontweight="bold", color=BRAND_BLUE)
        if delta:
            color = "#27ae60" if delta.startswith("+") else "#e74c3c"
            ax_m.text(0.5, 0.10, delta, ha="center", fontsize=9, color=color)

    # ── Health summary table ──────────────────────────────────────────────
    ax_t = fig.add_axes([0.05, 0.42, 0.40, 0.22])
    ax_t.axis("off")
    ax_t.text(0, 1.0, "Resumo de Saúde (câmeras)", fontsize=10,
              fontweight="bold", color=BRAND_BLUE, va="top")

    status_counts = (
        ranking["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    )
    total_cams = len(ranking)

    rows_data = [
        ["Total de câmeras", str(total_cams)],
        ["Câmeras com sinal na semana",
         str(int((ranking["status"].isin(["Atenção", "Alerta", "Crítico"])).sum()))],
    ]
    for s in STATUS_ORDER:
        rows_data.append([s, str(int(status_counts.get(s, 0)))])

    col_labels = ["Indicador", "Valor"]
    tbl = ax_t.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="upper left",
        cellLoc="left",
        bbox=[0, -0.05, 1, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd")
        if r == 0:
            cell.set_facecolor(BRAND_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 2:  # status rows
            status_name = STATUS_ORDER[r - 3]
            cell.set_facecolor(STATUS_COLORS.get(status_name, "white") + "33")

    # ── Signal type breakdown ─────────────────────────────────────────────
    ax_sig = fig.add_axes([0.52, 0.42, 0.43, 0.22])
    ax_sig.axis("off")
    ax_sig.text(0, 1.0, "Diagnóstico da Semana (tipos de sinal)",
                fontsize=10, fontweight="bold", color=BRAND_BLUE, va="top")

    sig_counts = (
        ranking.loc[ranking["status"] != "Saudável", "signal_type"]
        .value_counts()
        .reset_index()
        .values.tolist()
    )
    sig_data   = sig_counts if sig_counts else [["Sem sinal relevante", 0]]
    sig_labels = ["Tipo", "Qtde de câmeras"]
    tbl2 = ax_sig.table(
        cellText=sig_data,
        colLabels=sig_labels,
        loc="upper left",
        cellLoc="left",
        bbox=[0, -0.05, 1, 0.75],
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8.5)
    for (r, c), cell in tbl2.get_celld().items():
        cell.set_edgecolor("#ddd")
        if r == 0:
            cell.set_facecolor(BRAND_BLUE)
            cell.set_text_props(color="white", fontweight="bold")

    # ── Footer ────────────────────────────────────────────────────────────
    ax_foot = fig.add_axes([0, 0, 1, 0.04])
    ax_foot.set_facecolor(BRAND_BLUE)
    ax_foot.axis("off")
    ax_foot.text(0.5, 0.5, f"Gerado automaticamente — Semana ISO {iso_week:02d}/{iso_year}",
                 ha="center", va="center", color="#bdc3c7", fontsize=8,
                 transform=ax_foot.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_ranking(
    pdf: PdfPages,
    ranking: pd.DataFrame,
    shopping_name: str,
    iso_year: int,
    iso_week: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    _add_page_header(fig, f"Ranking de Saúde — Semana ISO {iso_week:02d}/{iso_year}", shopping_name)

    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.axis("off")

    cols = ["Câmera", "Score", "Status", "Tipo (semana)",
            "% zeros (12m)", "CV (12m)", "Zeros sem.", "EWMA", "CUSUM", "Slope 90d"]

    rows = []
    for _, row in ranking.iterrows():
        rows.append([
            str(row.get("camera", "")).replace("_", " ").title(),
            f"{row['score']:.1f}"             if not pd.isna(row.get("score"))          else "—",
            str(row.get("status", "—")),
            str(row.get("signal_type", "—")),
            f"{float(row['zeros_pct_12m'])*100:.1f}%" if not pd.isna(row.get("zeros_pct_12m")) else "—",
            "—",   # CV raw not stored in ranking — placeholder
            str(int(row["zeros_week"]))       if not pd.isna(row.get("zeros_week"))     else "0",
            "Sim" if row.get("ewma_alarm")   else "Não",
            "Sim" if row.get("cusum_alarm")  else "Não",
            f"{float(row['slope_90d']):+.2f}" if not pd.isna(row.get("slope_90d"))      else "—",
        ])

    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        loc="upper center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor(BRAND_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0:
            status = rows[r - 1][2]
            if c == 2:
                cell.set_facecolor(STATUS_COLORS.get(status, "white") + "44")
            elif r % 2 == 0:
                cell.set_facecolor("#f8f8f8")

    _add_page_footer(fig, iso_week, iso_year)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_camera(
    pdf: PdfPages,
    df: pd.DataFrame,
    summary_row: pd.Series,
    cam: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    shopping_name: str,
    iso_year: int,
    iso_week: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")

    status    = str(summary_row.get("status", "—"))
    score     = summary_row.get("score", np.nan)
    sig_type  = str(summary_row.get("signal_type", "—"))
    cam_label = cam.replace("_", " ").title()

    _add_page_header(fig, "Detalhamento — câmeras prioritárias", shopping_name)

    # Camera title strip
    ax_title = fig.add_axes([0.03, 0.84, 0.94, 0.05])
    ax_title.set_facecolor(STATUS_COLORS.get(status, "#95a5a6") + "33")
    ax_title.axis("off")
    score_str = f"{score:.1f}" if not pd.isna(score) else "—"
    pct_str   = str(summary_row.get("week_pct_str", "N/A"))
    zeros_w   = int(summary_row["zeros_week"]) if not pd.isna(summary_row.get("zeros_week")) else 0
    ewma_str  = "Sim" if summary_row.get("ewma_alarm")  else "Não"
    cusum_str = "Sim" if summary_row.get("cusum_alarm") else "Não"
    slope_str = f"{float(summary_row['slope_90d']):+.2f}" if not pd.isna(summary_row.get("slope_90d")) else "—"

    ax_title.text(0.01, 0.65, cam_label, fontsize=11, fontweight="bold", color=BRAND_BLUE)
    ax_title.text(
        0.01, 0.10,
        f"Score: {score_str}  |  Status: {status}  |  Tipo: {sig_type}  |  "
        f"Var. semana: {pct_str}  |  Zeros sem.: {zeros_w}  |  "
        f"EWMA: {ewma_str}  |  CUSUM: {cusum_str}  |  Slope 90d: {slope_str}",
        fontsize=7.5, color="#555",
    )

    # Three charts
    ref       = week_end
    df_60     = df[df[DATE_COL] >= ref - pd.Timedelta(days=60)].copy()
    df_90     = df[df[DATE_COL] >= ref - pd.Timedelta(days=90)].copy()

    required = [cam, f"{cam}_baseline_median", f"{cam}_baseline_mean", f"{cam}_residual"]
    signals  = [f"{cam}_ewma", f"{cam}_ewma_limit",
                f"{cam}_cusum_pos", f"{cam}_cusum_neg", f"{cam}_cusum_limit"]

    if not all(c in df.columns for c in required):
        plt.close(fig)
        return

    # Panel positions [left, bottom, width, height]
    panels = [
        fig.add_axes([0.08, 0.57, 0.88, 0.25]),
        fig.add_axes([0.08, 0.30, 0.88, 0.25]),
        fig.add_axes([0.08, 0.06, 0.88, 0.22]),
    ]

    _plot_observed_vs_baseline(panels[0], df_60, cam, week_start, week_end)
    _plot_residuals(panels[1], df_60, cam, week_start, week_end)

    if all(c in df.columns for c in signals):
        _plot_drift(panels[2], df_90, cam, week_start, week_end)
    else:
        panels[2].set_visible(False)

    _add_page_footer(fig, iso_week, iso_year)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart panels (shared with views/detail.py logic, standalone here for PDF)
# ---------------------------------------------------------------------------

def _plot_observed_vs_baseline(ax, df, cam, week_start, week_end):
    ax.plot(df[DATE_COL], df[cam],                      color="#3498db", lw=1.2, label="Observado")
    ax.plot(df[DATE_COL], df[f"{cam}_baseline_mean"],   color="#e67e22", lw=1.0, ls="--", label="Esperado (média DOW)")
    ax.plot(df[DATE_COL], df[f"{cam}_baseline_median"], color="#27ae60", lw=1.0, ls="--", label="Esperado (mediana DOW)")
    _vlines(ax, week_start, week_end)
    _fmt_ax(ax)
    ax.set_title(f"{cam} — Observado vs Baseline (últimos 60 dias)", fontsize=9)
    ax.set_ylabel("Fluxo", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def _plot_residuals(ax, df, cam, week_start, week_end):
    res = df[f"{cam}_residual"]
    ax.plot(df[DATE_COL], res, color="#3498db", lw=1.1)
    ax.axhline(0, color="#7f8c8d", lw=0.7)
    ax.fill_between(df[DATE_COL], res, 0, where=(res < 0),  alpha=0.25, color="#e74c3c")
    ax.fill_between(df[DATE_COL], res, 0, where=(res >= 0), alpha=0.15, color="#2ecc71")
    _vlines(ax, week_start, week_end)
    _fmt_ax(ax)
    ax.set_title(f"{cam} — Resíduos vs baseline mediana (últimos 60 dias)", fontsize=9)
    ax.set_ylabel("Resíduo", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _plot_drift(ax, df, cam, week_start, week_end):
    ewma      = df[f"{cam}_ewma"]
    ewma_lim  = float(df[f"{cam}_ewma_limit"].iloc[-1])
    cusum_pos = df[f"{cam}_cusum_pos"]
    cusum_neg = df[f"{cam}_cusum_neg"]
    cusum_lim = float(df[f"{cam}_cusum_limit"].iloc[-1])

    ax.plot(df[DATE_COL], ewma, color="#3498db", lw=1.1, label="EWMA (resíduo)")
    ax.axhline( ewma_lim, color="#3498db", lw=0.7, ls="--", alpha=0.7, label="Limite EWMA")
    ax.axhline(-ewma_lim, color="#3498db", lw=0.7, ls="--", alpha=0.7)
    ax.set_ylabel("EWMA", color="#3498db", fontsize=8)
    ax.tick_params(axis="y", labelcolor="#3498db", labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(df[DATE_COL], cusum_pos, color="#e74c3c", lw=1.0, label="CUSUM +")
    ax2.plot(df[DATE_COL], cusum_neg, color="#e67e22", lw=1.0, label="CUSUM - (abs)")
    ax2.axhline(cusum_lim, color="#e74c3c", lw=0.7, ls="--", alpha=0.6, label="Limite CUSUM")
    ax2.set_ylabel("CUSUM", color="#e74c3c", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#e74c3c", labelsize=7)

    _vlines(ax, week_start, week_end)
    _fmt_ax(ax)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper left")
    ax.set_title(f"{cam} — Drift (EWMA + CUSUM) — últimos 90 dias", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _vlines(ax, week_start, week_end):
    for d in [week_start, week_end]:
        ax.axvline(d, color=BRAND_BLUE, lw=0.8, ls="--", alpha=0.6)


def _fmt_ax(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7)


def _add_page_header(fig, title: str, shopping_name: str):
    ax = fig.add_axes([0, 0.92, 1, 0.08])
    ax.set_facecolor(BRAND_BLUE)
    ax.axis("off")
    ax.text(0.5, 0.70, title,         ha="center", color="white",   fontsize=11, fontweight="bold")
    ax.text(0.5, 0.20, shopping_name, ha="center", color="#bdc3c7", fontsize=9)


def _add_page_footer(fig, iso_week: int, iso_year: int):
    ax = fig.add_axes([0, 0, 1, 0.04])
    ax.set_facecolor(BRAND_BLUE)
    ax.axis("off")
    ax.text(0.5, 0.5, f"Semana ISO {iso_week:02d}/{iso_year}",
            ha="center", va="center", color="#bdc3c7", fontsize=8)


def _pct_str(current: int, previous: int | None) -> str:
    if previous is None or previous == 0:
        return ""
    pct = (current - previous) / abs(previous) * 100
    return f"{pct:+.1f}%"
