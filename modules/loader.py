"""
loader.py
---------
Responsible for ingesting, validating, and typing the raw CSV uploaded by the user.

Expected schema:
    data      : date column (any common format, parsed automatically)
    <cameras> : one or more numeric columns representing equipment flow
    total     : numeric column with the sum across all cameras
    evento    : string column with event name; empty = normal week
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"data", "total"}          # minimum required columns
DATE_COL      = "data"
TOTAL_COL     = "total"
EVENT_COL     = "evento"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    df: Optional[pd.DataFrame]             # cleaned dataframe (None on failure)
    camera_cols: list[str]                 # inferred camera column names
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.df is not None and len(self.errors) == 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_csv(file) -> LoadResult:
    """
    Load and validate a CSV file uploaded via Streamlit's file_uploader.

    Parameters
    ----------
    file : UploadedFile or path-like
        The file object returned by st.file_uploader.

    Returns
    -------
    LoadResult
        Dataclass with the cleaned DataFrame, inferred camera columns,
        validation errors, and warnings.
    """
    errors: list[str]   = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Read raw CSV
    # ------------------------------------------------------------------
    try:
        raw = pd.read_csv(
            file,
            sep=None,           # auto-detect separator
            engine="python",
            dtype=str,
            encoding_errors="replace",
        )
    except Exception as exc:
        return LoadResult(df=None, camera_cols=[], errors=[f"Could not read file: {exc}"])

    # Normalise column names: strip whitespace, lowercase, remove BOM
    raw.columns = [c.strip().lower().lstrip("\ufeff").lstrip("\xef\xbb\xbf") for c in raw.columns]

    # ------------------------------------------------------------------
    # 2. Check required columns
    # ------------------------------------------------------------------
    missing = REQUIRED_COLS - set(raw.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return LoadResult(df=None, camera_cols=[], errors=errors)

    # ------------------------------------------------------------------
    # 3. Parse date column
    # ------------------------------------------------------------------
    # Try multiple formats to handle diverse inputs
    date_formats = [
        None,                  # pandas auto-detect (ISO, DD/MM/YYYY, etc.)
        "%a %b %d, %Y",        # Mon Jan 01, 2024
        "%a %b %d %Y",         # Mon Jan 01 2024
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
    ]

    parsed = None
    for fmt in date_formats:
        try:
            if fmt is None:
                parsed = pd.to_datetime(raw[DATE_COL], infer_datetime_format=True)
            else:
                parsed = pd.to_datetime(raw[DATE_COL], format=fmt)
            break
        except Exception:
            continue

    if parsed is None:
        errors.append(
            f"Column '{DATE_COL}' could not be parsed as dates. "
            "Accepted formats: YYYY-MM-DD, DD/MM/YYYY, Mon Jan 01, 2024, etc."
        )
        return LoadResult(df=None, camera_cols=[], errors=errors)

    raw[DATE_COL] = parsed

    # Duplicate dates
    dupes = raw[raw.duplicated(subset=[DATE_COL], keep=False)][DATE_COL]
    if not dupes.empty:
        warnings.append(
            f"Duplicate dates found and removed (kept first): "
            f"{sorted(dupes.dt.strftime('%Y-%m-%d').unique().tolist())}"
        )
        raw = raw.drop_duplicates(subset=[DATE_COL], keep="first")

    raw = raw.sort_values(DATE_COL).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Infer camera columns
    #    Everything that is NOT data / total / evento is treated as a camera
    # ------------------------------------------------------------------
    reserved = {DATE_COL, TOTAL_COL, EVENT_COL}
    camera_cols = [c for c in raw.columns if c not in reserved]

    if len(camera_cols) == 0:
        errors.append(
            "No camera columns found. Expected at least one column besides "
            "'data', 'total', and 'evento'."
        )
        return LoadResult(df=None, camera_cols=[], errors=errors)

    # ------------------------------------------------------------------
    # 5. Convert numeric columns
    # ------------------------------------------------------------------
    numeric_cols = camera_cols + [TOTAL_COL]

    for col in numeric_cols:
        # Handle locale-formatted numbers where dot is thousands separator
        # e.g. "1.356" should be 1356, not 1.356
        cleaned = raw[col].astype(str).str.strip()

        # Detect dot-as-thousands: pattern like "1.234" with no decimal after 3 digits
        # Strategy: if comma is never present but dot always precedes exactly 3 digits → thousands
        has_comma  = cleaned.str.contains(",").any()
        dot_milhar = cleaned.str.match(r"^\d{1,3}(\.\d{3})+$").any()

        if not has_comma and dot_milhar:
            cleaned = cleaned.str.replace(".", "", regex=False)
        else:
            # Standard: comma as thousands, dot as decimal
            cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

        converted = pd.to_numeric(cleaned, errors="coerce")
        n_bad = int(converted.isna().sum()) - int(raw[col].isna().sum())
        if n_bad > 0:
            warnings.append(
                f"Column '{col}': {n_bad} non-numeric value(s) coerced to NaN."
            )
        raw[col] = converted

    # ------------------------------------------------------------------
    # 6. Handle evento column
    # ------------------------------------------------------------------
    if EVENT_COL not in raw.columns:
        warnings.append(
            f"Column '{EVENT_COL}' not found. All rows treated as normal weeks."
        )
        raw[EVENT_COL] = ""
    else:
        raw[EVENT_COL] = raw[EVENT_COL].fillna("").astype(str).str.strip()

    # ------------------------------------------------------------------
    # 7. Minimum data check
    # ------------------------------------------------------------------
    min_weeks = 12  # need at least 12 data points to compute meaningful baselines
    if len(raw) < min_weeks:
        warnings.append(
            f"Only {len(raw)} rows found. At least {min_weeks} rows are recommended "
            "for reliable baseline and signal computation."
        )

    # ------------------------------------------------------------------
    # 8. Add helper columns used downstream
    # ------------------------------------------------------------------
    iso                 = raw[DATE_COL].dt.isocalendar()
    raw["_dow"]         = raw[DATE_COL].dt.dayofweek          # 0=Monday … 6=Sunday
    raw["_iso_year"]    = iso.year.astype(int)
    raw["_iso_week"]    = iso.week.astype(int)
    # Unique string key per ISO week: "2026-W11"
    raw["_iso_week_id"] = (
        raw["_iso_year"].astype(str) + "-W" +
        raw["_iso_week"].astype(str).str.zfill(2)
    )
    raw["_has_event"]   = raw[EVENT_COL] != ""                # True when evento is non-empty

    # ISO-week-level event flag: True if ANY day in that ISO week has an event.
    # Used by baseline and signals to exclude entire weeks, not just individual days.
    week_has_event = (
        raw.groupby("_iso_week_id")["_has_event"]
        .transform("any")
    )
    raw["_iso_week_has_event"] = week_has_event

    return LoadResult(df=raw, camera_cols=camera_cols, errors=errors, warnings=warnings)
