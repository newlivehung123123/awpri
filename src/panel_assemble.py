"""
AWPRI Panel — Assembly & Normalization

Merges panel_layer1.csv, panel_layer2.csv, panel_layer3.csv
Applies risk direction corrections and min-max normalization
within each year (cross-sectional normalization) to preserve
comparability across countries at each time point.

Outputs:
  data/final/panel_awpri_raw.csv        — merged raw values
  data/final/panel_awpri_normalized.csv — normalized + layer scores + AWPRI
  data/final/panel_awpri_wide.csv       — wide format for ML models
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import *

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")
LOG_PATH  = os.path.join(BASE_DIR, "logs", "panel_coverage.txt")
os.makedirs(FINAL_DIR, exist_ok=True)

RAW_OUT   = os.path.join(FINAL_DIR, "panel_awpri_raw.csv")
NORM_OUT  = os.path.join(FINAL_DIR, "panel_awpri_normalized.csv")
WIDE_OUT  = os.path.join(FINAL_DIR, "panel_awpri_wide.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
coverage_lines = []

def log(msg):
    logger.info(msg)
    coverage_lines.append(msg)

# ── Variable definitions ──────────────────────────────────────────────────────

LAYER1_VARS = ["VAR_01", "VAR_02", "VAR_03", "VAR_04", "VAR_05"]
LAYER2_VARS = ["VAR_06", "VAR_07", "VAR_08", "VAR_09", "VAR_10"]
LAYER3_VARS = ["VAR_11", "VAR_12", "VAR_13", "VAR_14", "VAR_15"]
ALL_VARS    = LAYER1_VARS + LAYER2_VARS + LAYER3_VARS

# Raw column names from processed CSVs
RAW_COLS = {
    "VAR_01": "VAR_01_farmed_animals_slaughtered_per_capita",
    "VAR_02": "VAR_02_aquaculture_pct_total_animal_production",
    "VAR_03": "VAR_03_animal_rights_risk",
    "VAR_04": "VAR_04_rule_of_law_risk",
    "VAR_05": "VAR_05_meat_consumption_per_capita_kg",
    "VAR_06": "VAR_06_animal_rights_delta_risk",
    "VAR_07": "VAR_07_plant_protein_ratio_risk",
    "VAR_08": "VAR_08_public_concern_risk",
    "VAR_09": "VAR_09_civic_space_risk",
    "VAR_10": "VAR_10_civil_liberties_risk",
    "VAR_11": "VAR_11_ai_governance_aw_risk",
    "VAR_12": "VAR_12_ai_aw_research_per_million",
    "VAR_13": "VAR_13_ai_sentience_research_per_million",
    "VAR_14": "VAR_14_speciesist_bias_ratio",
    "VAR_15": "VAR_15_livestock_ai_patent_intensity",
}

# Short output names
SHORT_NAMES = {
    "VAR_01": "farmed_animals_per_capita",
    "VAR_02": "aquaculture_pct",
    "VAR_03": "animal_rights_risk",
    "VAR_04": "rule_of_law_risk",
    "VAR_05": "meat_consumption_kg",
    "VAR_06": "animal_rights_delta_risk",
    "VAR_07": "plant_protein_risk",
    "VAR_08": "public_concern_risk",
    "VAR_09": "civic_space_risk",
    "VAR_10": "civil_liberties_risk",
    "VAR_11": "ai_governance_risk",
    "VAR_12": "ai_aw_research_risk",
    "VAR_13": "ai_sentience_risk",
    "VAR_14": "speciesist_bias_ratio",
    "VAR_15": "patent_intensity",
}

# Variables needing inversion before normalization
# (currently stored as higher=better, need higher=worse for AWPRI)
INVERT_VARS = ["VAR_12", "VAR_13"]

# Variables needing log1p transform before normalization (extreme skew)
LOG_TRANSFORM_VARS = ["VAR_14"]

# ── Load and merge ────────────────────────────────────────────────────────────

def load_and_merge():
    l1 = pd.read_csv(os.path.join(PROC_DIR, "panel_layer1.csv"))
    l2 = pd.read_csv(os.path.join(PROC_DIR, "panel_layer2.csv"))
    l3 = pd.read_csv(os.path.join(PROC_DIR, "panel_layer3.csv"))

    log(f"Layer 1: {l1.shape} | Layer 2: {l2.shape} | Layer 3: {l3.shape}")

    df = l1.merge(l2, on=["country_iso2", "country_name", "year"], how="outer")
    df = df.merge(l3, on=["country_iso2", "country_name", "year"], how="outer")
    df = df.sort_values(["country_iso2", "year"]).reset_index(drop=True)

    log(f"Merged: {df.shape} — {df['country_iso2'].nunique()} countries, {df['year'].nunique()} years")
    log(f"Missing cells: {df.isna().sum().sum()}")
    return df

# ── Build raw matrix ──────────────────────────────────────────────────────────

def build_raw(df):
    raw = pd.DataFrame()
    raw["country_iso2"]  = df["country_iso2"]
    raw["country_name"]  = df["country_name"]
    raw["year"]          = df["year"]

    for var, raw_col in RAW_COLS.items():
        short = SHORT_NAMES[var]
        if raw_col in df.columns:
            if var in INVERT_VARS:
                # Invert: max - value (higher research = lower risk)
                s = df[raw_col]
                raw[short] = s.max() - s
            else:
                raw[short] = df[raw_col]
        else:
            log(f"  WARNING: {raw_col} not found in merged data")
            raw[short] = np.nan

    return raw

# ── Normalize within each year (cross-sectional) ─────────────────────────────

def minmax_normalize_series(s):
    """Min-max normalize a series to [0,1]. Returns 0.5 if constant."""
    valid = s.dropna()
    if len(valid) == 0:
        return s.copy()
    vmin, vmax = valid.min(), valid.max()
    if vmax == vmin:
        return pd.Series([0.5 if not pd.isna(v) else np.nan for v in s],
                         index=s.index, name=s.name)
    return (s - vmin) / (vmax - vmin)

def normalize_panel(raw):
    """
    Apply min-max normalization within each year across all 25 countries.
    This preserves cross-sectional comparability: at each year,
    the highest-risk country = 1.0 and lowest-risk = 0.0.
    """
    norm = raw[["country_iso2", "country_name", "year"]].copy()

    var_cols = [SHORT_NAMES[v] for v in ALL_VARS]

    for col in var_cols:
        if col not in raw.columns:
            norm[col] = np.nan
            continue

        normalized_col = pd.Series(index=raw.index, dtype=float)

        for year in YEARS:
            mask = raw["year"] == year
            year_data = raw.loc[mask, col].copy()

            # Apply log1p transform for skewed variables
            var_code = [k for k, v in SHORT_NAMES.items() if v == col]
            if var_code and var_code[0] in LOG_TRANSFORM_VARS:
                year_data = np.log1p(year_data)

            normalized_col.loc[mask] = minmax_normalize_series(year_data).values

        norm[col] = normalized_col.round(4)

    return norm

# ── Compute layer scores and AWPRI ────────────────────────────────────────────

def compute_scores(norm):
    def layer_mean(row, var_codes):
        vals = [row.get(SHORT_NAMES[v], np.nan) for v in var_codes]
        valid = [v for v in vals if not pd.isna(v)]
        return round(np.mean(valid), 4) if valid else np.nan

    norm["L1_score"]    = norm.apply(lambda r: layer_mean(r, LAYER1_VARS), axis=1)
    norm["L2_score"]    = norm.apply(lambda r: layer_mean(r, LAYER2_VARS), axis=1)
    norm["L3_score"]    = norm.apply(lambda r: layer_mean(r, LAYER3_VARS), axis=1)
    norm["AWPRI_score"] = norm.apply(
        lambda r: round(np.nanmean([r["L1_score"], r["L2_score"], r["L3_score"]]), 4),
        axis=1
    )
    return norm

# ── Build wide format for ML ──────────────────────────────────────────────────

def build_wide(norm):
    """
    Pivot to wide format: one row per country, columns are var_year.
    Format: country_iso2, VAR_01_2004, VAR_01_2005, ..., AWPRI_2022
    Useful for PCA and clustering across time.
    """
    id_cols = ["country_iso2", "country_name"]
    var_cols = [SHORT_NAMES[v] for v in ALL_VARS] + ["L1_score", "L2_score", "L3_score", "AWPRI_score"]

    wide_frames = []
    for year in YEARS:
        year_df = norm[norm["year"] == year][id_cols + var_cols].copy()
        year_df.columns = id_cols + [f"{c}_{year}" for c in var_cols]
        wide_frames.append(year_df.set_index(id_cols))

    wide = pd.concat(wide_frames, axis=1).reset_index()
    return wide

# ── Print rankings ────────────────────────────────────────────────────────────

def print_rankings(norm):
    log("\n" + "=" * 70)
    log("AWPRI PANEL — RANKINGS BY YEAR (selected years)")
    log("=" * 70)

    for year in [2004, 2010, 2016, 2022]:
        year_df = norm[norm["year"] == year].sort_values("AWPRI_score", ascending=False).reset_index(drop=True)
        year_df.index += 1
        log(f"\n--- {year} Rankings ---")
        log(f"{'Rank':<5} {'ISO2':<6} {'Country':<25} {'L1':>6} {'L2':>6} {'L3':>6} {'AWPRI':>7}")
        for rank, row in year_df.iterrows():
            log(f"{rank:<5} {row['country_iso2']:<6} {row['country_name']:<25} "
                f"{row['L1_score']:>6.3f} {row['L2_score']:>6.3f} "
                f"{row['L3_score']:>6.3f} {row['AWPRI_score']:>7.4f}")

# ── Validate ──────────────────────────────────────────────────────────────────

def validate(norm):
    log("\n" + "=" * 70)
    log("VALIDATION")
    log(f"  Total rows: {len(norm)} (expected 475)")
    log(f"  Countries: {norm['country_iso2'].nunique()} (expected 25)")
    log(f"  Years: {norm['year'].nunique()} (expected 19: 2004-2022)")
    log(f"  Missing AWPRI scores: {norm['AWPRI_score'].isna().sum()}")

    var_cols = [SHORT_NAMES[v] for v in ALL_VARS]
    total_cells = len(norm) * len(var_cols)
    missing = sum(norm[c].isna().sum() for c in var_cols if c in norm.columns)
    log(f"  Missing variable cells: {missing}/{total_cells} ({missing/total_cells*100:.1f}%)")

    # Check normalization bounds
    out_of_range = []
    for c in var_cols:
        if c in norm.columns:
            mn, mx = norm[c].min(), norm[c].max()
            if mn < -0.001 or mx > 1.001:
                out_of_range.append(f"{c}: [{mn:.3f}, {mx:.3f}]")
    if out_of_range:
        log(f"  OUT OF RANGE: {out_of_range}")
    else:
        log("  All normalized values within [0,1] ✓")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log(f"PANEL ASSEMBLY — {datetime.now().isoformat()}")
    log("=" * 70)

    # 1. Load and merge
    df = load_and_merge()

    # 2. Build raw matrix
    raw = build_raw(df)
    raw.to_csv(RAW_OUT, index=False)
    log(f"\nRaw panel saved → {RAW_OUT} | Shape: {raw.shape}")

    # 3. Normalize within each year
    norm = normalize_panel(raw)

    # 4. Compute scores
    norm = compute_scores(norm)
    norm.to_csv(NORM_OUT, index=False)
    log(f"Normalized panel saved → {NORM_OUT} | Shape: {norm.shape}")

    # 5. Build wide format
    wide = build_wide(norm)
    wide.to_csv(WIDE_OUT, index=False)
    log(f"Wide format saved → {WIDE_OUT} | Shape: {wide.shape}")

    # 6. Print rankings
    print_rankings(norm)

    # 7. Validate
    validate(norm)

    with open(LOG_PATH, "a") as f:
        f.write("\n".join(coverage_lines) + "\n")

    logger.info(f"\nSample output (China 2004-2022):")
    cn = norm[norm["country_iso2"] == "CN"][["year", "L1_score", "L2_score", "L3_score", "AWPRI_score"]]
    logger.info(cn.to_string(index=False))

if __name__ == "__main__":
    main()
