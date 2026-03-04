"""
AWPRI Pipeline — Normalization Utilities

Provides min-max normalization helpers used by assemble.py.
Also handles direction corrections and pre-normalization transforms
for variables where raw values require adjustment before scaling.

All variables in the final matrix are oriented so that:
  HIGH VALUE = HIGH RISK

Direction notes per variable:
  VAR_01: natural (high slaughter = high risk)
  VAR_02: natural (high aquaculture % = high risk)
  VAR_03: already inverted in layer1 (risk = 8 - WAP)
  VAR_04: already inverted in layer1 (risk = 1 - sentience)
  VAR_05: natural (high meat consumption = high risk)
  VAR_06: already inverted in layer2 (risk = -delta)
  VAR_07: already inverted in layer2 (risk = -plant_ratio_delta)
  VAR_08: already inverted in layer2 (risk = 100 - concern)
  VAR_09: already inverted in layer2 (risk = 5.5 - combined)
  VAR_10: already inverted in layer2 (risk = 1 - salience)
  VAR_11: already inverted in layer3 (risk = 1 - ethics_score)
  VAR_12: NOT yet inverted — higher output = lower risk → invert here
  VAR_13: NOT yet inverted — higher output = lower risk → invert here
  VAR_14: natural (higher bias ratio = higher risk)
          *** log1p-transformed before min-max to reduce extreme right skew ***
          Raw distribution: skew=+4.997, driven by China outlier (ratio=28.5).
          Transformation: log(x + 1) applied prior to normalisation.
          Raw values are preserved as-is in awpri_raw.csv; only the
          normalised column in awpri_normalized.csv uses the transformed values.
          Documented in logs/coverage_report.txt (2026-03-03).
  VAR_15: natural (higher patent intensity = higher risk)
"""

import numpy as np
import pandas as pd


# Variables that are currently stored as "higher = BETTER (lower risk)"
# and need to be inverted BEFORE normalisation.
INVERT_BEFORE_NORM = ["VAR_12", "VAR_13"]

# Variables that receive a log(x+1) transformation BEFORE min-max normalisation
# to compress extreme right skew caused by outliers.
# Rationale for VAR_14: China's speciesist_bias_ratio (28.5) vs median (~0.0)
# produces skew=+5.0 in raw data; log1p reduces this to ~+1.7 and preserves rank order.
LOG_TRANSFORM_BEFORE_NORM = ["VAR_14"]

# Mapping from short VAR_XX codes to full column names in processed CSVs
COLUMN_ALIASES = {
    "VAR_01": "VAR_01_farmed_animals_slaughtered_per_capita",
    "VAR_02": "VAR_02_aquaculture_pct_total_animal_production",
    "VAR_03": "VAR_03_wap_overall_score_risk",
    "VAR_04": "VAR_04_sentience_recognition_law_risk",
    "VAR_05": "VAR_05_meat_consumption_per_capita_kg",
    "VAR_06": "VAR_06_wap_delta_risk",
    "VAR_07": "VAR_07_plant_protein_ratio_risk",
    "VAR_08": "VAR_08_public_concern_risk",
    "VAR_09": "VAR_09_ngo_density_risk",
    "VAR_10": "VAR_10_political_salience_risk",
    "VAR_11": "VAR_11_ai_governance_aw_risk",
    "VAR_12": "VAR_12_ai_aw_research_per_million",
    "VAR_13": "VAR_13_ai_sentience_research_per_million",
    "VAR_14": "VAR_14_speciesist_bias_ratio",
    "VAR_15": "VAR_15_livestock_ai_patent_intensity",
}

# Human-readable short names for output
SHORT_NAMES = {
    "VAR_01": "farmed_animals_slaughtered_per_capita",
    "VAR_02": "aquaculture_pct_total_animal_production",
    "VAR_03": "wap_overall_score_risk",
    "VAR_04": "sentience_recognition_law_risk",
    "VAR_05": "meat_consumption_per_capita_kg",
    "VAR_06": "wap_delta_risk",
    "VAR_07": "plant_protein_ratio_risk",
    "VAR_08": "public_concern_risk",
    "VAR_09": "ngo_density_risk",
    "VAR_10": "political_salience_risk",
    "VAR_11": "ai_governance_aw_risk",
    "VAR_12": "ai_aw_research_risk",
    "VAR_13": "ai_sentience_research_risk",
    "VAR_14": "speciesist_bias_ratio",
    "VAR_15": "livestock_ai_patent_intensity",
}


def minmax_normalize(series: pd.Series) -> pd.Series:
    """
    Min-max normalize a pandas Series to [0, 1].
    NaN values are preserved as NaN.
    If all values are equal, returns 0.5 (no information).
    """
    valid = series.dropna()
    if len(valid) == 0:
        return series.copy()
    vmin = valid.min()
    vmax = valid.max()
    if vmax == vmin:
        return pd.Series(
            [0.5 if not pd.isna(v) else np.nan for v in series],
            index=series.index,
            name=series.name,
        )
    return (series - vmin) / (vmax - vmin)


def invert_series(series: pd.Series) -> pd.Series:
    """
    Invert a series that is currently oriented as 'higher = better'.
    If series is in [0, 1], returns 1 - series.
    Otherwise performs max - series (before normalisation).
    """
    valid = series.dropna()
    vmax = valid.max()
    return vmax - series


def prepare_raw_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the merged raw DataFrame, apply directional inversions for
    VAR_12 and VAR_13, rename columns to short names, and return
    the risk-direction-corrected raw matrix.
    """
    out = df.copy()

    for var_code in INVERT_BEFORE_NORM:
        long_col  = COLUMN_ALIASES[var_code]
        short_col = SHORT_NAMES[var_code]
        if long_col in out.columns:
            out[short_col] = invert_series(out[long_col])
            out.drop(columns=[long_col], inplace=True)
        elif short_col in out.columns:
            out[short_col] = invert_series(out[short_col])

    # Rename remaining long column names to short names
    rename_map = {}
    for var_code, long_col in COLUMN_ALIASES.items():
        short_col = SHORT_NAMES[var_code]
        if long_col in out.columns and short_col not in out.columns:
            rename_map[long_col] = short_col
    out.rename(columns=rename_map, inplace=True)

    return out


def normalize_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply pre-transforms then min-max normalisation to all 15 VAR columns.

    Pre-transform pipeline (applied in this order before min-max):
      1. log1p  for variables in LOG_TRANSFORM_BEFORE_NORM  (VAR_14)
      2. min-max scaling to [0, 1]

    Raw values in awpri_raw.csv are unaffected; only the normalized
    column values in awpri_normalized.csv use the transformed series.
    """
    out = df.copy()
    for var_code, short_col in SHORT_NAMES.items():
        if short_col not in out.columns:
            continue
        col = out[short_col].copy()
        # Step 1: log1p transform if flagged
        if var_code in LOG_TRANSFORM_BEFORE_NORM:
            col = np.log1p(col)
        # Step 2: min-max normalise
        out[short_col] = minmax_normalize(col)
    return out
