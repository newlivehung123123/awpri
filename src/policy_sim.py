"""
AWPRI Policy Simulation Engine
================================
Given a country + one or more policy interventions,
simulates the impact on AWPRI score and projects trajectory to 2030.

Interventions are defined as changes to specific normalized variables.
The simulation recomputes layer scores and AWPRI from modified variables,
then projects the trajectory forward.

Usage:
    from src.policy_sim import PolicySimulator
    ps = PolicySimulator()
    result = ps.simulate("VN", ["ai_governance_framework", "strengthen_civic_space"])
    print(result)
"""

import os, sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import NAMES
from nowcast import Nowcaster

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORECAST_PATH = os.path.join(BASE_DIR, "data", "ml", "forecasts_2030.csv")

# ── Policy intervention definitions ──────────────────────────────────────────
# Each policy specifies:
#   variables: which normalized variables it affects
#   effect:    multiplicative reduction (0.5 = reduce by 50%)
#   timeline:  years to full effect
#   description: human-readable label

POLICIES = {
    "ai_governance_framework": {
        "label":       "Adopt National AI Governance Framework",
        "description": "Implement a national AI ethics framework that explicitly covers "
                       "livestock, agriculture, and animal welfare applications of AI.",
        "variables":   {"ai_governance_risk": 0.50},
        "timeline":    2,
        "cost":        "Medium",
        "feasibility": "High",
        "examples":    "EU AI Act animal welfare provisions, OECD AI Principles",
    },
    "strengthen_rule_of_law": {
        "label":       "Strengthen Rule of Law & Judicial Independence",
        "description": "Institutional reforms to strengthen rule of law, judicial "
                       "independence, and enforcement of existing animal welfare legislation.",
        "variables":   {
            "rule_of_law_risk":      0.70,
            "civil_liberties_risk":  0.80,
        },
        "timeline":    7,
        "cost":        "High",
        "feasibility": "Medium",
        "examples":    "EU accession reforms, anti-corruption frameworks",
    },
    "strengthen_civic_space": {
        "label":       "Protect & Expand Civic Space for Animal Advocacy",
        "description": "Legal protections for animal welfare NGOs, whistleblowers, "
                       "and investigative journalists covering agricultural practices.",
        "variables":   {"civic_space_risk": 0.65},
        "timeline":    3,
        "cost":        "Low",
        "feasibility": "High",
        "examples":    "NGO registration reform, anti-SLAPP legislation",
    },
    "dietary_transition": {
        "label":       "National Dietary Transition Strategy",
        "description": "Government-backed strategy to shift dietary patterns toward "
                       "plant proteins, including subsidies, school meals, and labelling.",
        "variables":   {
            "plant_protein_risk":          0.70,
            "meat_consumption_kg":         0.80,
            "farmed_animals_per_capita":   0.85,
        },
        "timeline":    10,
        "cost":        "Medium",
        "feasibility": "Medium",
        "examples":    "Denmark protein strategy, EAT-Lancet aligned policies",
    },
    "aquaculture_welfare_standards": {
        "label":       "Mandatory Aquaculture Welfare Standards",
        "description": "Binding welfare standards for aquaculture operations including "
                       "stocking density limits, slaughter methods, and monitoring.",
        "variables":   {"aquaculture_pct": 0.75},
        "timeline":    4,
        "cost":        "Medium",
        "feasibility": "Medium",
        "examples":    "Norwegian salmon welfare regulations, ASC certification",
    },
    "ai_welfare_research_investment": {
        "label":       "National AI Animal Welfare Research Programme",
        "description": "Dedicated public funding for AI applications that improve "
                       "animal welfare monitoring, sentience research, and pain detection.",
        "variables":   {
            "ai_aw_research_risk":  0.60,
            "ai_sentience_risk":    0.65,
            "speciesist_bias_ratio": 0.75,
        },
        "timeline":    5,
        "cost":        "Medium",
        "feasibility": "High",
        "examples":    "UK Turing Institute animal sentience programme",
    },
    "comprehensive_welfare_reform": {
        "label":       "Comprehensive Animal Welfare Reform Package",
        "description": "Full package combining AI governance, civic space protections, "
                       "dietary transition, and research investment.",
        "variables":   {
            "ai_governance_risk":          0.50,
            "civic_space_risk":            0.65,
            "plant_protein_risk":          0.75,
            "meat_consumption_kg":         0.85,
            "farmed_animals_per_capita":   0.85,
            "ai_aw_research_risk":         0.65,
            "ai_sentience_risk":           0.70,
        },
        "timeline":    8,
        "cost":        "High",
        "feasibility": "Low",
        "examples":    "Green New Deal for Animals concept",
    },
}


class PolicySimulator:
    def __init__(self):
        self.nc   = Nowcaster()
        self.norm = self.nc.norm
        self.forecasts = pd.read_csv(FORECAST_PATH)
        print("PolicySimulator ready")

    def _recompute_awpri(self, var_scores: dict) -> tuple:
        """Recompute layer scores and AWPRI from a dict of variable scores."""
        L1_VARS = ["farmed_animals_per_capita", "aquaculture_pct", "animal_rights_risk",
                   "rule_of_law_risk", "meat_consumption_kg"]
        L2_VARS = ["animal_rights_delta_risk", "plant_protein_risk", "civic_space_risk",
                   "civil_liberties_risk", "public_concern_risk"]
        L3_VARS = ["ai_governance_risk", "ai_aw_research_risk", "ai_sentience_risk",
                   "speciesist_bias_ratio", "patent_intensity"]

        l1 = np.mean([var_scores.get(v, 0.5) for v in L1_VARS])
        l2 = np.mean([var_scores.get(v, 0.5) for v in L2_VARS])
        l3 = np.mean([var_scores.get(v, 0.5) for v in L3_VARS])
        awpri = np.mean([l1, l2, l3])
        return round(l1, 4), round(l2, 4), round(l3, 4), round(awpri, 4)

    def simulate(self, iso2: str, policy_ids: list, year: int = 2022) -> dict:
        """
        Simulate the impact of one or more policies on a country's AWPRI.

        Returns:
            baseline:  current risk profile
            simulated: risk profile after policies applied
            impact:    difference and % change per variable and composite
            trajectory: year-by-year path to full effect
            comparison: how simulated country compares to peers
        """
        iso2 = iso2.upper()

        # Get baseline
        baseline = self.nc.nowcast(iso2, year)
        if "error" in baseline:
            return baseline

        # Validate policy IDs
        valid_policies = {pid: POLICIES[pid] for pid in policy_ids if pid in POLICIES}
        if not valid_policies:
            return {"error": f"No valid policies in {policy_ids}. "
                             f"Available: {list(POLICIES.keys())}"}

        # Get baseline variable scores
        base_vars = {v: baseline["var_scores"][v]["score"]
                     for v in baseline["var_scores"]}

        # Apply all policies (multiplicative)
        sim_vars = base_vars.copy()
        for pid, policy in valid_policies.items():
            for var, multiplier in policy["variables"].items():
                if var in sim_vars:
                    sim_vars[var] = round(sim_vars[var] * multiplier, 4)

        # Recompute scores
        sim_l1, sim_l2, sim_l3, sim_awpri = self._recompute_awpri(sim_vars)

        # Impact calculation
        awpri_change = sim_awpri - baseline["awpri_score"]
        awpri_pct = round((awpri_change / baseline["awpri_score"]) * 100, 1)

        var_impacts = {}
        for var in base_vars:
            if base_vars[var] != sim_vars[var]:
                var_impacts[var] = {
                    "label":    baseline["var_scores"][var]["label"],
                    "before":   base_vars[var],
                    "after":    sim_vars[var],
                    "change":   round(sim_vars[var] - base_vars[var], 4),
                    "pct_change": round(
                        ((sim_vars[var] - base_vars[var]) / max(base_vars[var], 0.001)) * 100, 1
                    ),
                }

        # Timeline trajectory (linear interpolation to full effect)
        max_timeline = max(p["timeline"] for p in valid_policies.values())
        trajectory = {}
        for t in range(0, max_timeline + 1):
            frac = t / max_timeline
            t_vars = {}
            for var in base_vars:
                t_vars[var] = base_vars[var] + frac * (sim_vars[var] - base_vars[var])
            _, _, _, t_awpri = self._recompute_awpri(t_vars)
            trajectory[year + t] = round(t_awpri, 4)

        # Extend trajectory to 2030 if needed
        last_year = max(trajectory.keys())
        last_val  = trajectory[last_year]
        fc_row = self.forecasts[
            (self.forecasts["country_iso2"] == iso2) &
            (self.forecasts["year"] == 2030)
        ]
        fc_2030 = float(fc_row["AWPRI_forecast"].values[0]) if not fc_row.empty else last_val
        baseline_change_per_year = (fc_2030 - baseline["awpri_score"]) / 8

        for future_year in range(last_year + 1, 2031):
            trajectory[future_year] = round(
                max(0, min(1, last_val + baseline_change_per_year * (future_year - last_year))), 4
            )

        # Global rank after simulation
        year_df = self.norm[self.norm["year"] == year].copy()
        year_df_scores = year_df["AWPRI_score"].tolist()
        year_df_scores_sim = [
            s if c != iso2 else sim_awpri
            for s, c in zip(year_df_scores, year_df["country_iso2"])
        ]
        sim_rank = sorted(year_df_scores_sim, reverse=True).index(sim_awpri) + 1

        # Policy summary
        policy_summary = []
        for pid, policy in valid_policies.items():
            policy_summary.append({
                "id":          pid,
                "label":       policy["label"],
                "description": policy["description"],
                "timeline":    f"{policy['timeline']} years to full effect",
                "cost":        policy["cost"],
                "feasibility": policy["feasibility"],
                "examples":    policy["examples"],
                "variables":   policy.get("variables", {}),
            })

        return {
            "country_iso2":  iso2,
            "country_name":  NAMES.get(iso2, iso2),
            "year":          year,
            "policies_applied": policy_summary,
            "baseline": {
                "awpri_score": baseline["awpri_score"],
                "rank":        baseline["rank"],
                "L1": baseline["layer_scores"]["L1_current_state"],
                "L2": baseline["layer_scores"]["L2_policy_trajectory"],
                "L3": baseline["layer_scores"]["L3_ai_amplification"],
            },
            "simulated": {
                "awpri_score": sim_awpri,
                "rank":        sim_rank,
                "L1":          sim_l1,
                "L2":          sim_l2,
                "L3":          sim_l3,
            },
            "impact": {
                "awpri_change":     round(awpri_change, 4),
                "awpri_pct_change": awpri_pct,
                "rank_change":      baseline["rank"] - sim_rank,
                "variables":        var_impacts,
            },
            "trajectory": trajectory,
            "max_timeline_years": max_timeline,
        }

    def list_policies(self) -> pd.DataFrame:
        """Return a DataFrame of all available policies."""
        rows = []
        for pid, p in POLICIES.items():
            rows.append({
                "id":          pid,
                "label":       p["label"],
                "affects":     ", ".join(p["variables"].keys()),
                "timeline":    f"{p['timeline']} years",
                "cost":        p["cost"],
                "feasibility": p["feasibility"],
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    ps = PolicySimulator()

    print("\n=== AVAILABLE POLICIES ===")
    print(ps.list_policies().to_string(index=False))

    print("\n=== SIMULATE: Vietnam + AI Governance Framework ===")
    r = ps.simulate("VN", ["ai_governance_framework"])
    print(f"Baseline AWPRI: {r['baseline']['awpri_score']} (rank {r['baseline']['rank']})")
    print(f"Simulated AWPRI: {r['simulated']['awpri_score']} (rank {r['simulated']['rank']})")
    print(f"Change: {r['impact']['awpri_change']:+.4f} ({r['impact']['awpri_pct_change']:+.1f}%)")
    print(f"Rank improvement: {r['impact']['rank_change']:+d} places")
    print(f"\nVariable impacts:")
    for var, imp in r["impact"]["variables"].items():
        print(f"  {imp['label']}: {imp['before']:.3f} → {imp['after']:.3f} ({imp['pct_change']:+.1f}%)")
    print(f"\nTrajectory 2022-2030:")
    for yr, score in r["trajectory"].items():
        print(f"  {yr}: {score:.4f}")

    print("\n=== SIMULATE: China + Comprehensive Reform ===")
    r2 = ps.simulate("CN", ["comprehensive_welfare_reform"])
    print(f"Baseline: {r2['baseline']['awpri_score']} → Simulated: {r2['simulated']['awpri_score']}")
    print(f"Change: {r2['impact']['awpri_pct_change']:+.1f}%")
    print(f"Rank: {r2['baseline']['rank']} → {r2['simulated']['rank']}")
