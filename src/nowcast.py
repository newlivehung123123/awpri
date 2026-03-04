"""
AWPRI Nowcasting Engine
=======================
Given a country ISO2 code and optional year, returns:
- Current AWPRI score + confidence interval
- Risk archetype (cluster)
- Variable-by-variable risk breakdown
- Global percentile rank
- Comparison to regional and global averages
- Key risk drivers (top 3 variables above mean)
- Key strengths (top 3 variables below mean)

Usage:
    from src.nowcast import Nowcaster
    nc = Nowcaster()
    result = nc.nowcast("VN")
    print(result)
"""

import os, sys, json
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import NAMES, YEARS

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM_PATH  = os.path.join(BASE_DIR, "data", "final", "panel_awpri_normalized.csv")
ML_DIR     = os.path.join(BASE_DIR, "data", "ml")
MODEL_PATH = os.path.join(ML_DIR, "model_metadata.pkl")
RF_PATH    = os.path.join(ML_DIR, "rf_model.pkl")
CLUSTER_PATH = os.path.join(ML_DIR, "clusters_2022.csv")
FORECAST_PATH = os.path.join(ML_DIR, "forecasts_2030.csv")

ALL_VARS = [
    "farmed_animals_per_capita", "aquaculture_pct", "animal_rights_risk",
    "rule_of_law_risk", "animal_rights_delta_risk", "plant_protein_risk",
    "civic_space_risk", "civil_liberties_risk", "ai_governance_risk",
    "meat_consumption_kg", "public_concern_risk", "ai_aw_research_risk",
    "ai_sentience_risk", "speciesist_bias_ratio", "patent_intensity",
]

VAR_LABELS = {
    "farmed_animals_per_capita": "Farmed Animals (per capita)",
    "aquaculture_pct":           "Aquaculture % of Production",
    "animal_rights_risk":        "Animal Rights Framework",
    "rule_of_law_risk":          "Rule of Law",
    "animal_rights_delta_risk":  "Animal Rights Trend",
    "plant_protein_risk":        "Plant Protein Ratio",
    "civic_space_risk":          "Civic Space & NGO Freedom",
    "civil_liberties_risk":      "Civil Liberties",
    "ai_governance_risk":        "AI Governance Framework",
    "meat_consumption_kg":       "Meat Consumption (kg/capita)",
    "public_concern_risk":       "Public Concern for Animals",
    "ai_aw_research_risk":       "AI Welfare Research Output",
    "ai_sentience_risk":         "AI Sentience Research Output",
    "speciesist_bias_ratio":     "Speciesist Bias in AI Research",
    "patent_intensity":          "Livestock AI Patent Intensity",
}

VAR_LAYERS = {
    "farmed_animals_per_capita": "L1 — Current State",
    "aquaculture_pct":           "L1 — Current State",
    "animal_rights_risk":        "L1 — Current State",
    "rule_of_law_risk":          "L1 — Current State",
    "meat_consumption_kg":       "L1 — Current State",
    "animal_rights_delta_risk":  "L2 — Policy Trajectory",
    "plant_protein_risk":        "L2 — Policy Trajectory",
    "civic_space_risk":          "L2 — Policy Trajectory",
    "civil_liberties_risk":      "L2 — Policy Trajectory",
    "public_concern_risk":       "L2 — Policy Trajectory",
    "ai_governance_risk":        "L3 — AI Amplification",
    "ai_aw_research_risk":       "L3 — AI Amplification",
    "ai_sentience_risk":         "L3 — AI Amplification",
    "speciesist_bias_ratio":     "L3 — AI Amplification",
    "patent_intensity":          "L3 — AI Amplification",
}

RISK_TIER_LABELS = {
    "Critical Risk":  ("🔴", "Critical", "#d32f2f"),
    "High Risk":      ("🟠", "High",     "#f57c00"),
    "Moderate Risk":  ("🟡", "Moderate", "#fbc02d"),
    "Lower Risk":     ("🟢", "Lower",    "#388e3c"),
    "Minimal Risk":   ("🟢", "Minimal",  "#1b5e20"),
}

REGIONS = {
    "AU": "Oceania",   "NZ": "Oceania",
    "GB": "Europe",    "DE": "Europe",    "FR": "Europe",
    "NL": "Europe",    "SE": "Europe",    "DK": "Europe",
    "IT": "Europe",    "ES": "Europe",    "PL": "Europe",
    "CA": "N.America", "US": "N.America",
    "BR": "L.America", "AR": "L.America", "MX": "L.America",
    "CN": "Asia",      "JP": "Asia",      "KR": "Asia",
    "IN": "Asia",      "TH": "Asia",      "VN": "Asia",
    "KE": "Africa",    "NG": "Africa",    "ZA": "Africa",
}


class Nowcaster:
    def __init__(self):
        self.norm     = pd.read_csv(NORM_PATH)
        self.clusters = pd.read_csv(CLUSTER_PATH)
        self.forecasts = pd.read_csv(FORECAST_PATH)
        self.metadata = joblib.load(MODEL_PATH)
        self.rf       = joblib.load(RF_PATH)
        self.latest_year = self.norm["year"].max()
        print(f"Nowcaster ready — {len(self.norm['country_iso2'].unique())} countries, "
              f"latest year: {self.latest_year}")

    def nowcast(self, iso2: str, year: int = None) -> dict:
        """
        Returns full risk profile for a country at a given year.
        Defaults to most recent year if year not specified.
        """
        iso2 = iso2.upper()
        if year is None:
            year = self.latest_year

        row = self.norm[
            (self.norm["country_iso2"] == iso2) &
            (self.norm["year"] == year)
        ]

        if row.empty:
            return {"error": f"No data for {iso2} in {year}"}

        row = row.iloc[0]
        awpri = float(row["AWPRI_score"])
        l1    = float(row["L1_score"])
        l2    = float(row["L2_score"])
        l3    = float(row["L3_score"])

        # Global rank at this year
        year_df = self.norm[self.norm["year"] == year].sort_values(
            "AWPRI_score", ascending=False).reset_index(drop=True)
        year_df.index += 1
        rank = int(year_df[year_df["country_iso2"] == iso2].index[0])
        n_countries = len(year_df)
        percentile = round((1 - rank / n_countries) * 100, 1)

        # Global and regional averages
        global_avg = float(year_df["AWPRI_score"].mean())
        region = REGIONS.get(iso2, "Unknown")
        region_countries = [c for c, r in REGIONS.items() if r == region]
        region_df = year_df[year_df["country_iso2"].isin(region_countries)]
        region_avg = float(region_df["AWPRI_score"].mean())

        # Variable breakdown
        var_scores = {}
        for var in ALL_VARS:
            if var in row.index:
                var_scores[var] = {
                    "score":       round(float(row[var]), 4),
                    "label":       VAR_LABELS[var],
                    "layer":       VAR_LAYERS[var],
                    "global_mean": round(float(self.norm[
                        self.norm["year"] == year][var].mean()), 4),
                    "risk_level":  (
                        "High"     if float(row[var]) > 0.66 else
                        "Moderate" if float(row[var]) > 0.33 else
                        "Low"
                    ),
                }

        # Key risk drivers (vars most above global mean)
        drivers = sorted(
            [(v, d["score"] - d["global_mean"]) for v, d in var_scores.items()],
            key=lambda x: x[1], reverse=True
        )[:3]

        # Key strengths (vars most below global mean)
        strengths = sorted(
            [(v, d["global_mean"] - d["score"]) for v, d in var_scores.items()],
            key=lambda x: x[1], reverse=True
        )[:3]

        # Cluster / archetype
        cluster_row = self.clusters[self.clusters["country_iso2"] == iso2]
        archetype = cluster_row["risk_archetype"].values[0] if not cluster_row.empty else "Unknown"

        # Confidence interval (simple ±5% based on panel std)
        country_hist = self.norm[self.norm["country_iso2"] == iso2]["AWPRI_score"]
        ci_half = max(0.03, float(country_hist.std()) * 1.96)
        ci_lower = round(max(0.0, awpri - ci_half), 4)
        ci_upper = round(min(1.0, awpri + ci_half), 4)

        # Historical trajectory
        hist = self.norm[self.norm["country_iso2"] == iso2].sort_values("year")
        trajectory = {
            int(r["year"]): round(float(r["AWPRI_score"]), 4)
            for _, r in hist.iterrows()
        }

        # 2030 forecast
        fc_row = self.forecasts[
            (self.forecasts["country_iso2"] == iso2) &
            (self.forecasts["year"] == 2030)
        ]
        forecast_2030 = None
        if not fc_row.empty:
            forecast_2030 = {
                "score":    round(float(fc_row["AWPRI_forecast"].values[0]), 4),
                "lower_95": round(float(fc_row["lower_95"].values[0]), 4),
                "upper_95": round(float(fc_row["upper_95"].values[0]), 4),
                "trend":    fc_row["trend"].values[0],
            }

        tier = RISK_TIER_LABELS.get(archetype, ("⚪", "Unknown", "#9e9e9e"))

        return {
            "country_iso2":   iso2,
            "country_name":   NAMES.get(iso2, iso2),
            "year":           year,
            "awpri_score":    round(awpri, 4),
            "ci_lower":       ci_lower,
            "ci_upper":       ci_upper,
            "rank":           rank,
            "n_countries":    n_countries,
            "percentile":     percentile,
            "risk_archetype": archetype,
            "risk_tier":      tier,
            "region":         region,
            "layer_scores": {
                "L1_current_state":    round(l1, 4),
                "L2_policy_trajectory": round(l2, 4),
                "L3_ai_amplification": round(l3, 4),
            },
            "global_avg":     round(global_avg, 4),
            "region_avg":     round(region_avg, 4),
            "var_scores":     var_scores,
            "key_risk_drivers": [
                {"var": v, "label": VAR_LABELS[v], "above_mean": round(d, 4)}
                for v, d in drivers
            ],
            "key_strengths": [
                {"var": v, "label": VAR_LABELS[v], "below_mean": round(d, 4)}
                for v, d in strengths
            ],
            "trajectory":    trajectory,
            "forecast_2030": forecast_2030,
        }

    def compare(self, iso2_list: list, year: int = None) -> pd.DataFrame:
        """Compare multiple countries side by side."""
        if year is None:
            year = self.latest_year
        rows = []
        for iso2 in iso2_list:
            r = self.nowcast(iso2, year)
            if "error" not in r:
                rows.append({
                    "country":   r["country_name"],
                    "iso2":      r["country_iso2"],
                    "AWPRI":     r["awpri_score"],
                    "rank":      r["rank"],
                    "archetype": r["risk_archetype"],
                    "L1":        r["layer_scores"]["L1_current_state"],
                    "L2":        r["layer_scores"]["L2_policy_trajectory"],
                    "L3":        r["layer_scores"]["L3_ai_amplification"],
                    "forecast_2030": r["forecast_2030"]["score"] if r["forecast_2030"] else None,
                })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    nc = Nowcaster()

    # Test single country
    print("\n=== VIETNAM NOWCAST ===")
    result = nc.nowcast("VN")
    print(f"Country: {result['country_name']}")
    print(f"AWPRI: {result['awpri_score']} [{result['ci_lower']}-{result['ci_upper']}]")
    print(f"Rank: {result['rank']}/{result['n_countries']} (top {100-result['percentile']:.0f}%)")
    print(f"Archetype: {result['risk_tier'][0]} {result['risk_archetype']}")
    print(f"Layers: L1={result['layer_scores']['L1_current_state']} "
          f"L2={result['layer_scores']['L2_policy_trajectory']} "
          f"L3={result['layer_scores']['L3_ai_amplification']}")
    print(f"2030 Forecast: {result['forecast_2030']}")
    print(f"\nKey Risk Drivers:")
    for d in result["key_risk_drivers"]:
        print(f"  ↑ {d['label']}: +{d['above_mean']:.3f} above global mean")
    print(f"\nKey Strengths:")
    for s in result["key_strengths"]:
        print(f"  ↓ {s['label']}: -{s['below_mean']:.3f} below global mean")

    # Test comparison
    print("\n=== COMPARISON: CN vs GB vs VN ===")
    comp = nc.compare(["CN", "GB", "VN"])
    print(comp.to_string(index=False))
