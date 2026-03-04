"""
AWPRI Pipeline — Diagnostic Report
Prints descriptive statistics (mean, std, min, max) for all 15 variables
across 25 countries, plus the correlation matrix between layer sub-scores
and the AWPRI composite.

Reads:  data/final/awpri_raw.csv
        data/final/awpri_normalized.csv
Writes: logs/diagnostic_report.txt
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH  = os.path.join(BASE_DIR, "data", "final", "awpri_raw.csv")
NORM_PATH = os.path.join(BASE_DIR, "data", "final", "awpri_normalized.csv")
LOG_PATH  = os.path.join(BASE_DIR, "logs", "diagnostic_report.txt")

raw  = pd.read_csv(RAW_PATH)
norm = pd.read_csv(NORM_PATH)

VAR_COLS_RAW = [c for c in raw.columns if c not in ("country_iso2", "country_name")]
VAR_COLS_NORM = [c for c in norm.columns
                 if c not in ("country_iso2", "country_name",
                               "L1_score", "L2_score", "L3_score", "AWPRI_score")]
SCORE_COLS = ["L1_score", "L2_score", "L3_score", "AWPRI_score"]

lines = []

def h(title, char="="):
    width = 78
    lines.append("")
    lines.append(char * width)
    lines.append(f"  {title}")
    lines.append(char * width)

def row(label, mean, std, vmin, vmax, skew):
    lines.append(
        f"  {label:<42}  mean={mean:>10.4f}  std={std:>9.4f}"
        f"  min={vmin:>10.4f}  max={vmax:>10.4f}  skew={skew:>+7.3f}"
    )

# ── Header ────────────────────────────────────────────────────────────────────
lines.append("=" * 78)
lines.append("  AWPRI DIAGNOSTIC REPORT")
lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"  Dataset:   25 countries × 15 variables")
lines.append("=" * 78)

# ── Section 1: Raw variable statistics ───────────────────────────────────────
h("SECTION 1 — RAW VARIABLE DESCRIPTIVE STATISTICS (risk-direction corrected)")
lines.append(f"  {'Variable':<42}  {'Mean':>10}  {'Std':>9}  {'Min':>10}  {'Max':>10}  {'Skew':>7}")
lines.append("  " + "-" * 74)

for col in VAR_COLS_RAW:
    s = raw[col].dropna()
    row(col, s.mean(), s.std(), s.min(), s.max(), float(s.skew()))

# ── Section 2: Normalized variable statistics ─────────────────────────────────
h("SECTION 2 — NORMALIZED VARIABLE DESCRIPTIVE STATISTICS (min-max, 0–1 scale)")
lines.append("  Note: VAR_14 (speciesist_bias_ratio) is log(x+1)-transformed before")
lines.append("  min-max scaling to correct extreme right skew (raw skew=+5.0, CN outlier).")
lines.append(f"  {'Variable':<42}  {'Mean':>10}  {'Std':>9}  {'Min':>10}  {'Max':>10}  {'Skew (norm)':>11}")
lines.append("  " + "-" * 78)

for col in VAR_COLS_NORM:
    s = norm[col].dropna()
    row(col, s.mean(), s.std(), s.min(), s.max(), float(s.skew()))

# ── Section 3: Layer & composite score statistics ────────────────────────────
h("SECTION 3 — LAYER SUB-SCORES AND AWPRI COMPOSITE STATISTICS")
lines.append(f"  {'Score':<42}  {'Mean':>10}  {'Std':>9}  {'Min':>10}  {'Max':>10}  {'Skew':>7}")
lines.append("  " + "-" * 74)

for col in SCORE_COLS:
    s = norm[col].dropna()
    row(col, s.mean(), s.std(), s.min(), s.max(), float(s.skew()))

# ── Section 4: Correlation matrix ────────────────────────────────────────────
h("SECTION 4 — CORRELATION MATRIX  (L1_score, L2_score, L3_score, AWPRI_score)")
corr = norm[SCORE_COLS].corr()
# Pretty-print
col_w = 13
header = "  " + " " * 14 + "".join(f"{c:>{col_w}}" for c in SCORE_COLS)
lines.append(header)
lines.append("  " + "-" * (14 + col_w * len(SCORE_COLS) + 2))
for idx_col in SCORE_COLS:
    vals = "".join(f"{corr.loc[idx_col, c]:>{col_w}.4f}" for c in SCORE_COLS)
    lines.append(f"  {idx_col:<14}{vals}")

# ── Section 5: Per-country scores ────────────────────────────────────────────
h("SECTION 5 — PER-COUNTRY SCORES (sorted by AWPRI_score descending)")
ranked = norm[["country_iso2", "country_name"] + SCORE_COLS] \
             .sort_values("AWPRI_score", ascending=False) \
             .reset_index(drop=True)
ranked.index += 1

lines.append(f"  {'Rank':<5} {'ISO2':<6} {'Country':<30} {'L1':>7} {'L2':>7} {'L3':>7} {'AWPRI':>8}")
lines.append("  " + "-" * 74)
for rank, r in ranked.iterrows():
    lines.append(
        f"  {rank:<5} {r['country_iso2']:<6} {r['country_name']:<30}"
        f" {r['L1_score']:>7.4f} {r['L2_score']:>7.4f}"
        f" {r['L3_score']:>7.4f} {r['AWPRI_score']:>8.4f}"
    )

# ── Section 6: Variable × score correlations ─────────────────────────────────
h("SECTION 6 — VARIABLE → AWPRI_score CORRELATIONS (Pearson, normalized values)")
var_awpri_corr = norm[VAR_COLS_NORM + ["AWPRI_score"]].corr()["AWPRI_score"] \
                     .drop("AWPRI_score") \
                     .sort_values(ascending=False)

lines.append(f"  {'Variable':<42}  {'r with AWPRI':>13}  {'Interpretation'}")
lines.append("  " + "-" * 74)
for var, r_val in var_awpri_corr.items():
    if pd.isna(r_val):
        interp = "undefined (zero variance)"
    elif r_val >= 0.6:
        interp = "strong positive driver"
    elif r_val >= 0.3:
        interp = "moderate positive driver"
    elif r_val >= 0.0:
        interp = "weak positive"
    elif r_val >= -0.3:
        interp = "weak negative"
    elif r_val >= -0.6:
        interp = "moderate negative"
    else:
        interp = "strong negative"
    lines.append(f"  {var:<42}  {r_val:>13.4f}  {interp}")

# Interpretation note
meat_r = var_awpri_corr.get("meat_consumption_per_capita_kg", float("nan"))
lines.append("")
lines.append("  INTERPRETATION NOTE — meat_consumption_per_capita_kg (r ≈ {:.2f}):".format(meat_r))
lines.append("  ─" * 38)
lines.append("  The negative correlation between meat consumption and AWPRI risk is")
lines.append("  counterintuitive but explicable by a confound with national wealth.")
lines.append("  The highest meat-consuming countries in this dataset (US, AU, AR, NZ,")
lines.append("  DK) are also wealthy, institutionally developed nations with strong")
lines.append("  animal welfare legislation (high WAP scores), active NGO ecosystems,")
lines.append("  and high political salience — factors that collectively dominate the")
lines.append("  composite and pull their AWPRI scores toward the lower-risk end.")
lines.append("  Conversely, low-consumption countries (IN, KE, NG, VN) tend to score")
lines.append("  high risk due to weak legal frameworks and low civic capacity, not")
lines.append("  because of consumption behaviour per se. This reflects a known")
lines.append("  limitation of governance-weighted composite indices: they can reward")
lines.append("  institutional quality even when absolute harm volumes are high.")
lines.append("  Analysts should interpret VAR_05 in conjunction with VAR_01 (slaughter")
lines.append("  volume per capita) rather than in isolation.")

# ── Section 7: Missing data audit ────────────────────────────────────────────
h("SECTION 7 — MISSING DATA AUDIT")
lines.append("  (All values should be 0 given hardcoded fallbacks)")
lines.append("")
any_missing = False
for col in VAR_COLS_NORM:
    n = norm[col].isna().sum()
    status = "✓ OK" if n == 0 else f"⚠ {n} MISSING"
    lines.append(f"  {col:<42}  {status}")
    if n > 0:
        any_missing = True
lines.append("")
lines.append(f"  Overall: {'✓ No missing values' if not any_missing else '⚠ Missing values detected'}")

lines.append("")
lines.append("=" * 78)
lines.append("  END OF DIAGNOSTIC REPORT")
lines.append("=" * 78)
lines.append("")

# ── Write & print ─────────────────────────────────────────────────────────────
output = "\n".join(lines)
print(output)

with open(LOG_PATH, "w") as f:
    f.write(output)

print(f"\n✓ Saved → {LOG_PATH}")
