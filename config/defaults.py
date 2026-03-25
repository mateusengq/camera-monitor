"""
config/defaults.py
------------------
Single source of truth for all configurable parameters.
Every sidebar widget pulls its default from here.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

BASELINE_LOOKBACK_MONTHS: int   = 12       # options: 3, 6, 12
BASELINE_DECAY:           float = 0.3      # recency decay [0.0 = flat, 1.0 = max recency]

# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------

EWMA_LAMBDA: float = 0.2                   # smoothing factor [0.05 – 0.50]

# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

CUSUM_K: float = 1.0                       # slack / allowance  [0.5 – 3.0]
CUSUM_H: float = 5.0                       # decision threshold [2.0 – 10.0]

# ---------------------------------------------------------------------------
# Score weights  (each in [0.0 – 1.0])
# ---------------------------------------------------------------------------

WEIGHT_CUSUM:     float = 1.0
WEIGHT_EWMA:      float = 1.0
WEIGHT_DROP:      float = 1.0
WEIGHT_CV:        float = 1.0
WEIGHT_ISOLATION: float = 1.0

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

DROP_ALERT_THRESHOLD: float = 0.20         # 20% weekly drop triggers full drop penalty

# ---------------------------------------------------------------------------
# Convenience bundle — passed directly to scorer.compute_scores
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "cusum":     WEIGHT_CUSUM,
    "ewma":      WEIGHT_EWMA,
    "drop":      WEIGHT_DROP,
    "cv":        WEIGHT_CV,
    "isolation": WEIGHT_ISOLATION,
}
