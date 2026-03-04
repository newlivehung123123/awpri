"""
AWPRI Pipeline — Assemble & Normalize

Reads layer1.csv, layer2.csv, layer3.csv from data/processed/,
merges them, applies risk direction corrections, saves:
  data/final/awpri_raw.csv        — raw merged values (risk-direction corrected)
  data/final/awpri_normalized.csv — min-max normalized + layer scores + AWPRI composite

Columns in final CSV:
  country_iso2, country_name,
  VAR_01..VAR_15 (short names),
  L1_score, L2_score, L3_score, AWPRI_score

Prints top-10 highest-risk countries ranked by AWPRI_score.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
FINAL_DIR   = os.path.join(BASE_DIR, "data", "final")
LOG_PATH    = os.path.join(BASE_DIR, "logs", "coverage_report.txt")

os.makedirs(FINAL_DIR, exist_ok=True)

RAW_OUT     = os.path.join(FINAL_DIR, "awpri_raw.csv")
NORM_OUT    = os.path.join(FINAL_DIR, "awpri_normalized.csv")

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ── Import normalize utilities ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (
    SHORT_NAMES, COLUMN_ALIASES, INVERT_BEFORE_NORM,
    minmax_normalize, invert_series, prepare_raw_matrix, normalize_matrix
)

# ── Layer → Variable mapping ──────────────────────────────────────────────────
LAYER1_VARS = ["VAR_01", "VAR_02", "VAR_03", "VAR_04", "VAR_05"]
LAYER2_VARS = ["VAR_06", "VAR_07", "VAR_08", "VAR_09", "VAR_10"]
LAYER3_VARS = ["VAR_11", "VAR_12", "VAR_13", "VAR_14", "VAR_15"]

ALL_VAR_CODES = LAYER1_VARS + LAYER2_VARS + LAYER3_VARS

coverage_lines = []

def log(msg: str):
    logger.info(msg)
    coverage_lines.append(msg)

# ── Load processed layers ─────────────────────────────────────────────────────
def load_layers() -> pd.DataFrame:
    l1 = pd.read_csv(os.path.join(PROC_DIR, "layer1.csv"))
    l2 = pd.read_csv(os.path.join(PROC_DIR, "layer2.csv"))
    l3 = pd.read_csv(os.path.join(PROC_DIR, "layer3.csv"))

    log(f"Layer 1 shape: {l1.shape}")
    log(f"Layer 2 shape: {l2.shape}")
    log(f"Layer 3 shape: {l3.shape}")

    # Merge on country_iso2
    df = l1.merge(l2, on=["country_iso2", "country_name"], how="outer")
    df = df.merge(l3, on=["country_iso2", "country_name"], how="outer")

    log(f"Merged shape: {df.shape} — {df['country_iso2'].nunique()} countries")
    return df

# ── Build raw risk-corrected matrix ──────────────────────────────────────────
def build_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to short names and invert VAR_12/VAR_13.
    All 15 VAR columns will be risk-direction-correct (high = high risk).
    """
    raw = prepare_raw_matrix(df)

    # Ensure all 15 short columns exist
    for var_code in ALL_VAR_CODES:
        col = SHORT_NAMES[var_code]
        if col not in raw.columns:
            log(f"  ⚠ Column {col} missing from merged data — filling with NaN")
            raw[col] = np.nan

    # Keep only meta + 15 VAR columns
    keep_cols = ["country_iso2", "country_name"] + [SHORT_NAMES[v] for v in ALL_VAR_CODES]
    # Filter to only existing columns
    keep_cols = [c for c in keep_cols if c in raw.columns]
    raw = raw[keep_cols].copy()

    # Summary of missingness
    log("\nMissingness in raw matrix:")
    for var_code in ALL_VAR_CODES:
        col = SHORT_NAMES[var_code]
        if col in raw.columns:
            n = raw[col].isna().sum()
            log(f"  {var_code} ({col}): {n} missing")
        else:
            log(f"  {var_code}: COLUMN ABSENT")

    return raw

# ── Normalize and compute scores ──────────────────────────────────────────────
def build_normalized(raw: pd.DataFrame) -> pd.DataFrame:
    norm = normalize_matrix(raw)

    # Compute layer sub-scores as mean of normalized variables
    def layer_score(row, var_codes):
        vals = [row.get(SHORT_NAMES[v], np.nan) for v in var_codes]
        valid = [v for v in vals if not pd.isna(v)]
        return round(np.mean(valid), 4) if valid else np.nan

    norm["L1_score"] = norm.apply(lambda r: layer_score(r, LAYER1_VARS), axis=1)
    norm["L2_score"] = norm.apply(lambda r: layer_score(r, LAYER2_VARS), axis=1)
    norm["L3_score"] = norm.apply(lambda r: layer_score(r, LAYER3_VARS), axis=1)

    # Composite AWPRI score
    norm["AWPRI_score"] = norm.apply(
        lambda r: round(np.nanmean([r["L1_score"], r["L2_score"], r["L3_score"]]), 4),
        axis=1
    )

    # Round all VAR columns to 4 decimal places
    for var_code in ALL_VAR_CODES:
        col = SHORT_NAMES[var_code]
        if col in norm.columns:
            norm[col] = norm[col].round(4)

    return norm

# ── Print rankings ────────────────────────────────────────────────────────────
def print_rankings(norm: pd.DataFrame):
    ranked = norm[["country_iso2", "country_name", "L1_score", "L2_score", "L3_score", "AWPRI_score"]] \
                 .sort_values("AWPRI_score", ascending=False) \
                 .reset_index(drop=True)
    ranked.index += 1  # 1-based ranking

    log("\n" + "=" * 70)
    log("AWPRI — TOP 25 COUNTRIES RANKED BY COMPOSITE RISK SCORE")
    log("=" * 70)
    log(f"{'Rank':<5} {'ISO2':<6} {'Country':<30} {'L1':>6} {'L2':>6} {'L3':>6} {'AWPRI':>7}")
    log("-" * 70)
    for rank, row in ranked.iterrows():
        log(
            f"{rank:<5} {row['country_iso2']:<6} {row['country_name']:<30} "
            f"{row['L1_score']:>6.3f} {row['L2_score']:>6.3f} "
            f"{row['L3_score']:>6.3f} {row['AWPRI_score']:>7.4f}"
        )

    log("\nTop 10 highest-risk countries:")
    for i, row in ranked.head(10).iterrows():
        log(f"  {i:2d}. {row['country_name']} ({row['country_iso2']}) — AWPRI={row['AWPRI_score']:.4f}")

# ── Validate completeness ──────────────────────────────────────────────────────
def validate(norm: pd.DataFrame):
    n_countries = norm["country_iso2"].nunique()
    n_vars      = sum(1 for v in ALL_VAR_CODES if SHORT_NAMES[v] in norm.columns)
    total_cells = n_countries * n_vars
    missing_cells = sum(norm[SHORT_NAMES[v]].isna().sum() for v in ALL_VAR_CODES if SHORT_NAMES[v] in norm.columns)

    log(f"\n{'='*70}")
    log(f"VALIDATION SUMMARY")
    log(f"  Countries: {n_countries}/25")
    log(f"  Variables: {n_vars}/15")
    log(f"  Total cells: {total_cells} | Missing: {missing_cells} ({missing_cells/total_cells*100:.1f}%)")
    if n_countries < 25:
        log(f"  ⚠ Missing countries: {set(['AU','BR','CA','FR','DE','IN','IT','JP','NL','NZ','KR','ES','SE','GB','US','AR','CN','DK','KE','MX','NG','PL','ZA','TH','VN']) - set(norm['country_iso2'])}")
    if missing_cells == 0:
        log("  ✓ No missing values — all 25×15 cells populated")
    else:
        log(f"  ⚠ {missing_cells} cells are NaN (see coverage report for details)")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log(f"AWPRI ASSEMBLY — {datetime.now().isoformat()}")
    log("=" * 70)
    log("TRANSFORM NOTE — VAR_14 speciesist_bias_ratio:")
    log("  Raw values stored in awpri_raw.csv are UNTRANSFORMED (ratio scale).")
    log("  A log(x+1) transformation is applied in normalize.py BEFORE min-max")
    log("  scaling for awpri_normalized.csv. Rationale: raw skew=+4.997 driven")
    log("  by China outlier (ratio=28.5 vs median≈0.0). log1p compresses the")
    log("  tail, reduces skew to ~+1.7, and preserves rank order across all")
    log("  25 countries. No other variables are transformed.")

    # 1. Load and merge layers
    df  = load_layers()

    # 2. Build risk-corrected raw matrix
    raw = build_raw(df)

    # 3. Save raw
    raw.to_csv(RAW_OUT, index=False)
    log(f"\nRaw matrix saved → {RAW_OUT}")
    log(f"  Shape: {raw.shape}")

    # 4. Normalize and compute scores
    norm = build_normalized(raw)

    # 5. Save normalized
    norm.to_csv(NORM_OUT, index=False)
    log(f"\nNormalized matrix saved → {NORM_OUT}")
    log(f"  Shape: {norm.shape}")

    # 6. Print rankings
    print_rankings(norm)

    # 7. Validate
    validate(norm)

    # 8. Append to coverage report
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(coverage_lines) + "\n")
    log(f"\nAssembly log appended → {LOG_PATH}")

    # 9. Print final raw and normalized tables to stdout for quick review
    logger.info("\n--- RAW MATRIX (first 5 rows) ---")
    logger.info(raw.head().to_string(index=False))
    logger.info("\n--- NORMALIZED MATRIX (first 5 rows) ---")
    logger.info(norm.head().to_string(index=False))

if __name__ == "__main__":
    main()
