"""
AWPRI Dashboard — Streamlit App
================================
Animal Welfare and Policy Risk Index
AI/ML-driven prototype for the Futurekind AI Fellowship

Pages:
  1. Global Overview    — world map + rankings
  2. Country Deep-Dive  — variable breakdown + radar chart + trajectory
  3. Policy Simulator   — intervention sliders + impact projection
  4. Forecasts 2030     — trajectory charts per country
  5. Methodology        — variable definitions + data sources
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from nowcast import Nowcaster, RISK_TIER_LABELS, VAR_LABELS, VAR_LAYERS, REGIONS
from policy_sim import PolicySimulator, POLICIES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AWPRI — Animal Welfare & Policy Risk Index",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load engines (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_engines():
    nc = Nowcaster()
    ps = PolicySimulator()
    return nc, ps

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    norm      = pd.read_csv(os.path.join(base, "data/final/panel_awpri_normalized.csv"))
    forecasts = pd.read_csv(os.path.join(base, "data/ml/forecasts_2030.csv"))
    clusters  = pd.read_csv(os.path.join(base, "data/ml/clusters_2022.csv"))
    fi        = pd.read_csv(os.path.join(base, "data/ml/feature_importance.csv"))
    return norm, forecasts, clusters, fi

nc, ps   = load_engines()
norm, forecasts, clusters, fi = load_data()

ALL_COUNTRIES = sorted(norm["country_iso2"].unique())
COUNTRY_NAMES = {r["country_iso2"]: r["country_name"] for _, r in
                 norm[["country_iso2","country_name"]].drop_duplicates().iterrows()}
YEARS_HIST    = sorted(norm["year"].unique())

# ── Colour palette ────────────────────────────────────────────────────────────
RISK_COLORS = {
    "Critical Risk": "#d32f2f",
    "High Risk":     "#f57c00",
    "Moderate Risk": "#fbc02d",
    "Lower Risk":    "#388e3c",
    "Minimal Risk":  "#1b5e20",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐾 Animal Welfare & Policy Risk Index")
    st.caption("AI/ML-driven risk assessment across 25 countries, 2004–2030")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🌍 Global Overview",
         "🔍 Country Deep-Dive",
         "⚙️ Policy Simulator",
         "📈 Forecasts 2030",
         "📖 Methodology"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Data: V-Dem v15, FAOSTAT, OpenAlex, Google Trends")
    st.caption("Model: PCA + K-Means + Random Forest + ARIMA")
    st.caption("© 2025 AWPRI Project")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — GLOBAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🌍 Global Overview":
    st.title("🌍 Global Animal Welfare Risk Overview")
    st.caption("Animal Welfare and Policy Risk Index (AWPRI) — higher scores indicate greater risk")

    # Year selector
    selected_year = st.select_slider(
        "Select year", options=YEARS_HIST, value=2022
    )

    year_df = norm[norm["year"] == selected_year].merge(
        clusters[["country_iso2","risk_archetype"]], on="country_iso2", how="left"
    ).sort_values("AWPRI_score", ascending=False).reset_index(drop=True)
    year_df.index += 1
    year_df["country_name"] = year_df["country_iso2"].map(COUNTRY_NAMES)

    # ── Headline metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Countries", "25")
    col2.metric("Years of Data", "2004–2022")
    col3.metric("Global Avg AWPRI", f"{year_df['AWPRI_score'].mean():.3f}")
    col4.metric("Highest Risk", f"{year_df.iloc[0]['country_iso2']} ({year_df.iloc[0]['AWPRI_score']:.3f})")
    col5.metric("Lowest Risk",  f"{year_df.iloc[-1]['country_iso2']} ({year_df.iloc[-1]['AWPRI_score']:.3f})")

    st.divider()

    col_map, col_rank = st.columns([2, 1])

    with col_map:
        st.subheader(f"AWPRI World Map — {selected_year}")
        ISO2_TO_ISO3 = {
            "AR":"ARG","AU":"AUS","BR":"BRA","CA":"CAN","CN":"CHN",
            "DE":"DEU","DK":"DNK","ES":"ESP","FR":"FRA","GB":"GBR",
            "IN":"IND","IT":"ITA","JP":"JPN","KE":"KEN","KR":"KOR",
            "MX":"MEX","NG":"NGA","NL":"NLD","NZ":"NZL","PL":"POL",
            "SE":"SWE","TH":"THA","US":"USA","VN":"VNM","ZA":"ZAF",
        }
        year_df["iso3"] = year_df["country_iso2"].map(ISO2_TO_ISO3)

        fig_map = px.choropleth(
            year_df,
            locations="iso3",
            locationmode="ISO-3",
            color="AWPRI_score",
            hover_name="country_name",
            hover_data={
                "country_iso2": False,
                "AWPRI_score":  ":.3f",
                "L1_score":     ":.3f",
                "L2_score":     ":.3f",
                "L3_score":     ":.3f",
            },
            color_continuous_scale=["#388e3c","#fbc02d","#f57c00","#d32f2f"],
            range_color=[0.25, 0.85],
            labels={"AWPRI_score": "AWPRI Risk Score"},
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(showframe=False, showcoastlines=True,
                     projection_type="natural earth"),
            coloraxis_colorbar=dict(title="Risk Score", thickness=12),
            height=420,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_rank:
        st.subheader(f"Risk Rankings — {selected_year}")
        for i, row in year_df.iterrows():
            archetype = row.get("risk_archetype", "Moderate Risk")
            color = RISK_COLORS.get(archetype, "#9e9e9e")
            emoji = RISK_TIER_LABELS.get(archetype, ("⚪",))[0]
            st.markdown(
                f"{emoji} **{i}. {row['country_iso2']}** — {row['country_name']}  \n"
                f"<span style='font-size:12px;color:{color}'>AWPRI: {row['AWPRI_score']:.3f} | {archetype}</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Layer breakdown bar chart ──
    st.subheader(f"Layer Score Breakdown — {selected_year}")
    fig_layers = go.Figure()
    for layer, col, label in [
        ("L1_score", "#42a5f5", "L1: Current State"),
        ("L2_score", "#66bb6a", "L2: Policy Trajectory"),
        ("L3_score", "#ffa726", "L3: AI Amplification"),
    ]:
        fig_layers.add_trace(go.Bar(
            name=label,
            x=year_df["country_iso2"],
            y=year_df[layer],
            marker_color=col,
        ))
    fig_layers.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis=dict(title="Country"),
    )
    st.plotly_chart(fig_layers, use_container_width=True)

    # ── Trend: worsening vs improving ──
    st.subheader("Risk Trajectory 2004 → 2022")
    df_2004 = norm[norm["year"]==2004][["country_iso2","AWPRI_score"]].rename(columns={"AWPRI_score":"s2004"})
    df_2022 = norm[norm["year"]==2022][["country_iso2","AWPRI_score"]].rename(columns={"AWPRI_score":"s2022"})
    trend_df = df_2004.merge(df_2022, on="country_iso2")
    trend_df["change"] = trend_df["s2022"] - trend_df["s2004"]
    trend_df["direction"] = trend_df["change"].apply(lambda x: "Worsening ↑" if x > 0 else "Improving ↓")
    trend_df["country_name"] = trend_df["country_iso2"].map(COUNTRY_NAMES)
    trend_df = trend_df.sort_values("change", ascending=False)

    fig_trend = px.bar(
        trend_df, x="country_iso2", y="change",
        color="direction",
        color_discrete_map={"Worsening ↑": "#d32f2f", "Improving ↓": "#388e3c"},
        hover_data={"country_name": True, "s2004": ":.3f", "s2022": ":.3f"},
        labels={"change": "AWPRI Change", "country_iso2": "Country"},
        height=300,
    )
    fig_trend.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COUNTRY DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Country Deep-Dive":
    st.title("🔍 Country Deep-Dive")

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_country = st.selectbox(
            "Select country",
            ALL_COUNTRIES,
            index=ALL_COUNTRIES.index("VN"),
            format_func=lambda x: f"{x} — {COUNTRY_NAMES.get(x, x)}",
        )
    with col_sel2:
        selected_year = st.selectbox("Year", YEARS_HIST, index=len(YEARS_HIST)-1)

    result = nc.nowcast(selected_country, selected_year)
    if "error" in result:
        st.error(result["error"])
        st.stop()

    tier_emoji, tier_label, tier_color = result["risk_tier"]

    # ── Header ──
    st.markdown(f"## {tier_emoji} {result['country_name']} — {selected_year}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AWPRI Score",   f"{result['awpri_score']:.3f}",
                delta=f"95% CI: [{result['ci_lower']:.3f}–{result['ci_upper']:.3f}]")
    col2.metric("Global Rank",   f"#{result['rank']} / {result['n_countries']}")
    col3.metric("Risk Archetype", tier_label)
    col4.metric("vs Global Avg", f"{result['awpri_score'] - result['global_avg']:+.3f}",
                delta_color="inverse")
    col5.metric("2030 Forecast",
                f"{result['forecast_2030']['score']:.3f}" if result['forecast_2030'] else "N/A",
                delta=result['forecast_2030']['trend'] if result['forecast_2030'] else "")

    st.divider()

    col_radar, col_vars = st.columns([1, 1])

    with col_radar:
        st.subheader("Variable Risk Profile")
        var_names  = [VAR_LABELS[v] for v in list(result["var_scores"].keys())]
        var_values = [result["var_scores"][v]["score"] for v in result["var_scores"]]
        global_avg = [result["var_scores"][v]["global_mean"] for v in result["var_scores"]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=var_values + [var_values[0]],
            theta=var_names + [var_names[0]],
            fill="toself", name=result["country_name"],
            line_color=tier_color, fillcolor=tier_color,
            opacity=0.6,
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=global_avg + [global_avg[0]],
            theta=var_names + [var_names[0]],
            fill="toself", name="Global Average",
            line_color="#9e9e9e", fillcolor="#9e9e9e",
            opacity=0.3,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=True, height=420,
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_vars:
        st.subheader("Variable Breakdown")
        vs = result["var_scores"]
        var_df = pd.DataFrame([{
            "Variable":  vs[v]["label"],
            "Layer":     vs[v]["layer"],
            "Score":     vs[v]["score"],
            "Global Avg": vs[v]["global_mean"],
            "Risk Level": vs[v]["risk_level"],
        } for v in vs]).sort_values("Score", ascending=False)

        for _, row in var_df.iterrows():
            risk_color = (
                "#d32f2f" if row["Risk Level"] == "High" else
                "#f57c00" if row["Risk Level"] == "Moderate" else
                "#388e3c"
            )
            bar_pct = int(row["Score"] * 100)
            avg_pct = int(row["Global Avg"] * 100)
            st.markdown(
                f"**{row['Variable']}** <span style='font-size:11px;color:#666'>{row['Layer']}</span>  \n"
                f"<div style='background:#f0f0f0;border-radius:4px;height:8px;width:100%'>"
                f"<div style='background:{risk_color};width:{bar_pct}%;height:8px;border-radius:4px'></div>"
                f"</div>"
                f"<span style='font-size:11px'>{row['Score']:.3f} &nbsp;|&nbsp; "
                f"Global avg: {row['Global Avg']:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.write("")

    st.divider()

    # ── Historical trajectory ──
    st.subheader(f"Historical Trajectory — {result['country_name']}")
    traj = result["trajectory"]
    fc   = result["forecast_2030"]

    fig_traj = go.Figure()
    # Historical
    fig_traj.add_trace(go.Scatter(
        x=list(traj.keys()), y=list(traj.values()),
        mode="lines+markers", name="Historical AWPRI",
        line=dict(color=tier_color, width=2),
        marker=dict(size=5),
    ))
    # Forecast
    if fc:
        fc_years  = list(forecasts[forecasts["country_iso2"]==selected_country]["year"])
        fc_scores = list(forecasts[forecasts["country_iso2"]==selected_country]["AWPRI_forecast"])
        fc_lower  = list(forecasts[forecasts["country_iso2"]==selected_country]["lower_95"])
        fc_upper  = list(forecasts[forecasts["country_iso2"]==selected_country]["upper_95"])
        fig_traj.add_trace(go.Scatter(
            x=[2022] + fc_years, y=[traj[2022]] + fc_scores,
            mode="lines+markers", name="ARIMA Forecast",
            line=dict(color=tier_color, width=2, dash="dash"),
            marker=dict(size=5),
        ))
        fig_traj.add_trace(go.Scatter(
            x=fc_years + fc_years[::-1],
            y=fc_upper + fc_lower[::-1],
            fill="toself", fillcolor=tier_color,
            opacity=0.15, line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
        ))

    fig_traj.add_vline(x=2022, line_dash="dot", line_color="gray",
                       annotation_text="Now", annotation_position="top right")
    fig_traj.update_layout(
        height=320, yaxis=dict(range=[0,1], title="AWPRI Score"),
        xaxis=dict(title="Year"),
        margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # ── Key drivers and strengths ──
    col_d, col_s = st.columns(2)
    with col_d:
        st.subheader("⚠️ Key Risk Drivers")
        for d in result["key_risk_drivers"]:
            st.markdown(f"🔴 **{d['label']}** — +{d['above_mean']:.3f} above global mean")
    with col_s:
        st.subheader("✅ Key Strengths")
        for s in result["key_strengths"]:
            st.markdown(f"🟢 **{s['label']}** — -{s['below_mean']:.3f} below global mean")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — POLICY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Policy Simulator":
    st.title("⚙️ Policy Impact Simulator")
    st.caption("Simulate the effect of policy interventions on animal welfare risk")

    col_sel1, col_sel2 = st.columns([1, 1])
    with col_sel1:
        sim_country = st.selectbox(
            "Select country to simulate",
            ALL_COUNTRIES,
            index=ALL_COUNTRIES.index("VN"),
            format_func=lambda x: f"{x} — {COUNTRY_NAMES.get(x, x)}",
        )
    with col_sel2:
        selected_policies = st.multiselect(
            "Select policy interventions",
            options=list(POLICIES.keys()),
            default=["ai_governance_framework"],
            format_func=lambda x: POLICIES[x]["label"],
        )

    if not selected_policies:
        st.info("Select at least one policy intervention above.")
        st.stop()

    # Run simulation
    sim_result = ps.simulate(sim_country, selected_policies)
    if "error" in sim_result:
        st.error(sim_result["error"])
        st.stop()

    bl = sim_result["baseline"]
    sm = sim_result["simulated"]
    imp = sim_result["impact"]

    # ── Impact summary ──
    st.divider()
    st.subheader(f"Impact Summary — {sim_result['country_name']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline AWPRI",   f"{bl['awpri_score']:.3f}")
    col2.metric("Simulated AWPRI",  f"{sm['awpri_score']:.3f}",
                delta=f"{imp['awpri_pct_change']:+.1f}%",
                delta_color="inverse")
    col3.metric("Rank Change",
                f"#{sm['rank']}",
                delta=f"{imp['rank_change']:+d} places",
                delta_color="inverse")
    col4.metric("Max. Timeline",
                f"{sim_result['max_timeline_years']} years to full effect")

    st.divider()

    col_chart, col_detail = st.columns([3, 2])

    with col_chart:
        st.subheader("Trajectory: Baseline vs. With Policy")
        traj = sim_result["trajectory"]

        # Get historical AWPRI
        hist = norm[norm["country_iso2"]==sim_country].sort_values("year")
        hist_years  = list(hist["year"])
        hist_scores = list(hist["AWPRI_score"])

        # Get baseline forecast
        fc_base = forecasts[forecasts["country_iso2"]==sim_country].sort_values("year")
        fc_years  = list(fc_base["year"])
        fc_scores = list(fc_base["AWPRI_forecast"])

        fig_sim = go.Figure()
        # Historical
        fig_sim.add_trace(go.Scatter(
            x=hist_years, y=hist_scores,
            mode="lines+markers", name="Historical",
            line=dict(color="#555", width=2),
            marker=dict(size=4),
        ))
        # Baseline forecast
        fig_sim.add_trace(go.Scatter(
            x=[2022] + fc_years, y=[hist_scores[-1]] + fc_scores,
            mode="lines", name="Baseline (no policy)",
            line=dict(color="#f57c00", width=2, dash="dash"),
        ))
        # Policy trajectory
        sim_traj_years  = list(traj.keys())
        sim_traj_scores = list(traj.values())
        fig_sim.add_trace(go.Scatter(
            x=sim_traj_years, y=sim_traj_scores,
            mode="lines+markers", name="With Policy",
            line=dict(color="#388e3c", width=2),
            marker=dict(size=4),
        ))
        fig_sim.add_vline(x=2022, line_dash="dot", line_color="gray",
                          annotation_text="Policy Enacted", annotation_position="top right")
        fig_sim.update_layout(
            height=350, yaxis=dict(range=[0,1], title="AWPRI Score"),
            xaxis=dict(title="Year"),
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    with col_detail:
        st.subheader("Layer Impact")
        layers_df = pd.DataFrame([
            {"Layer": "L1 — Current State",     "Baseline": bl["L1"], "Simulated": sm["L1"]},
            {"Layer": "L2 — Policy Trajectory", "Baseline": bl["L2"], "Simulated": sm["L2"]},
            {"Layer": "L3 — AI Amplification",  "Baseline": bl["L3"], "Simulated": sm["L3"]},
            {"Layer": "AWPRI Composite",         "Baseline": bl["awpri_score"], "Simulated": sm["awpri_score"]},
        ])
        for _, row in layers_df.iterrows():
            diff = row["Simulated"] - row["Baseline"]
            color = "#388e3c" if diff < 0 else "#d32f2f"
            st.markdown(
                f"**{row['Layer']}**  \n"
                f"{row['Baseline']:.3f} → **{row['Simulated']:.3f}** "
                f"<span style='color:{color}'>{diff:+.3f}</span>",
                unsafe_allow_html=True,
            )
            st.write("")

    # ── Variable impact table ──
    if imp["variables"]:
        st.subheader("Variable-Level Impact")
        var_imp_df = pd.DataFrame([{
            "Variable":   v["label"],
            "Before":     v["before"],
            "After":      v["after"],
            "Change":     v["change"],
            "% Change":   v["pct_change"],
        } for v in imp["variables"].values()])
        st.dataframe(
            var_imp_df.style.format({
                "Before": "{:.3f}", "After": "{:.3f}",
                "Change": "{:+.3f}", "% Change": "{:+.1f}%",
            }).background_gradient(subset=["Change"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

    # ── Policy details ──
    st.subheader("Policy Details")
    for pol in sim_result["policies_applied"]:
        with st.expander(f"📋 {pol['label']}"):
            st.markdown(f"**Description:** {pol['description']}")
            st.markdown(f"**Timeline:** {pol['timeline']}")
            st.markdown(f"**Cost:** {pol['cost']} &nbsp;&nbsp; **Feasibility:** {pol['feasibility']}")
            st.markdown(f"**Real-world examples:** {pol['examples']}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FORECASTS 2030
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecasts 2030":
    st.title("📈 AWPRI Forecasts to 2030")
    st.caption("ARIMA time-series forecasts — 95% confidence intervals shown")

    col_all, col_single = st.columns([2, 1])

    with col_all:
        st.subheader("2030 Projected Rankings")
        f2030 = forecasts[forecasts["year"]==2030].copy()
        f2030["country_name"] = f2030["country_iso2"].map(COUNTRY_NAMES)
        f2030 = f2030.sort_values("AWPRI_forecast", ascending=False).reset_index(drop=True)
        f2030.index += 1

        f2030["color"] = f2030["trend"].apply(
            lambda x: "#d32f2f" if "WORSENING" in str(x).upper() or "↑" in str(x)
                      else "#388e3c"
        )

        fig_f2030 = go.Figure()
        # 2022 baseline
        baseline_2022 = norm[norm["year"]==2022][["country_iso2","AWPRI_score"]]
        f2030_merged = f2030.merge(baseline_2022, on="country_iso2")
        f2030_merged = f2030_merged.sort_values("AWPRI_forecast", ascending=True)

        fig_f2030.add_trace(go.Bar(
            y=f2030_merged["country_iso2"],
            x=f2030_merged["AWPRI_score"],
            name="2022 (actual)",
            orientation="h",
            marker_color="#90a4ae",
        ))
        fig_f2030.add_trace(go.Bar(
            y=f2030_merged["country_iso2"],
            x=f2030_merged["AWPRI_forecast"],
            name="2030 (forecast)",
            orientation="h",
            marker_color=f2030_merged["color"],
            opacity=0.85,
        ))
        fig_f2030.update_layout(
            barmode="overlay", height=600,
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(range=[0,1], title="AWPRI Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
        )
        st.plotly_chart(fig_f2030, use_container_width=True)

    with col_single:
        st.subheader("Country Trajectory")
        fc_country = st.selectbox(
            "Select country",
            ALL_COUNTRIES,
            format_func=lambda x: f"{x} — {COUNTRY_NAMES.get(x, x)}",
            key="fc_country",
        )
        hist = norm[norm["country_iso2"]==fc_country].sort_values("year")
        fc_c = forecasts[forecasts["country_iso2"]==fc_country].sort_values("year")

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=list(hist["year"]), y=list(hist["AWPRI_score"]),
            mode="lines+markers", name="Historical",
            line=dict(color="#1565c0", width=2), marker=dict(size=5),
        ))
        fig_fc.add_trace(go.Scatter(
            x=[2022]+list(fc_c["year"]),
            y=[float(hist[hist["year"]==2022]["AWPRI_score"])]+list(fc_c["AWPRI_forecast"]),
            mode="lines+markers", name="Forecast",
            line=dict(color="#1565c0", width=2, dash="dash"), marker=dict(size=5),
        ))
        fig_fc.add_trace(go.Scatter(
            x=list(fc_c["year"])+list(fc_c["year"])[::-1],
            y=list(fc_c["upper_95"])+list(fc_c["lower_95"])[::-1],
            fill="toself", fillcolor="rgba(21,101,192,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))
        fig_fc.add_vline(x=2022, line_dash="dot", line_color="gray")
        fig_fc.update_layout(
            height=380, yaxis=dict(range=[0,1], title="AWPRI"),
            xaxis=dict(title="Year"),
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # 2030 summary for this country
        fc_2030_row = fc_c[fc_c["year"]==2030]
        if not fc_2030_row.empty:
            score_2030 = fc_2030_row["AWPRI_forecast"].values[0]
            score_2022 = float(hist[hist["year"]==2022]["AWPRI_score"])
            change = score_2030 - score_2022
            direction = "↑ Worsening" if change > 0 else "↓ Improving"
            color = "#d32f2f" if change > 0 else "#388e3c"
            st.markdown(
                f"**2022:** {score_2022:.3f}  \n"
                f"**2030 forecast:** {score_2030:.3f}  \n"
                f"**Change:** <span style='color:{color}'>{change:+.3f} ({direction})</span>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 Methodology":
    st.title("📖 Methodology")

    st.markdown("""
    ## Overview
    The **Animal Welfare and Policy Risk Index (AWPRI)** is an AI/ML-driven composite index
    measuring the risk that animals face inadequate welfare protections across 25 countries
    from 2004 to 2022, with forecasts to 2030.

    It was developed for the **Futurekind AI Fellowship** as a prototype demonstrating
    how machine learning can be used to assess and forecast animal welfare policy risk.
    """)

    st.divider()

    st.markdown("## Index Structure")
    st.markdown("""
    AWPRI is composed of **15 variables** across **3 layers**, each weighted equally:

    | Layer | Weight | Description |
    |---|---|---|
    | **L1 — Current State** | 33% | Agricultural intensity and baseline governance |
    | **L2 — Policy Trajectory** | 33% | Direction of policy change and civil society capacity |
    | **L3 — AI Amplification** | 33% | How AI adoption and governance gaps amplify risk |
    """)

    st.divider()

    st.markdown("## Variables")
    var_table = pd.DataFrame([{
        "Variable": VAR_LABELS[v],
        "Layer": VAR_LAYERS[v],
        "Source": (
            "FAOSTAT" if v in ["farmed_animals_per_capita","aquaculture_pct","meat_consumption_kg","plant_protein_risk"] else
            "V-Dem v15" if v in ["animal_rights_risk","rule_of_law_risk","civic_space_risk","civil_liberties_risk","animal_rights_delta_risk"] else
            "Google Trends" if v == "public_concern_risk" else
            "AWPRI / Binary" if v == "ai_governance_risk" else
            "OpenAlex / PATSTAT"
        ),
        "Time Coverage": "2004–2022 annual",
    } for v in VAR_LABELS])
    st.dataframe(var_table, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("## ML Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **1. PCA (Principal Component Analysis)**
        - Reduces 15 variables to principal components
        - PC1 (51.6% variance): Governance risk dimension
        - PC2 (17.8% variance): Agricultural intensity dimension
        - 4 components explain >85% of variance

        **2. K-Means Clustering**
        - Identifies country risk archetypes
        - Optimal k=3 (silhouette score optimised)
        - Critical Risk / High Risk / Moderate Risk
        """)
    with col2:
        st.markdown("""
        **3. Random Forest**
        - Predicts AWPRI from 15 input variables
        - Top predictor: Civil Liberties Risk (49.2%)
        - Demonstrates governance dominance over agricultural factors

        **4. ARIMA Forecasting**
        - Per-country time-series forecasting to 2030
        - Stationarity-tested (ADF test)
        - 95% confidence intervals reported
        """)

    st.divider()

    st.markdown("## Data Sources")
    st.markdown("""
    | Source | Variables | Coverage |
    |---|---|---|
    | [V-Dem v15](https://www.v-dem.net) | Animal rights, rule of law, civic space, civil liberties | 1900–2023 |
    | [FAOSTAT](https://www.fao.org/faostat) | Farmed animals, aquaculture, meat consumption, plant protein | 2004–2022 |
    | [Google Trends](https://trends.google.com) | Public concern for animal welfare | 2004–2022 |
    | [OpenAlex](https://openalex.org) | AI welfare and sentience research output | 2004–2022 |
    | PATSTAT / LENS | Livestock AI patent intensity | 2004–2022 |
    | AWPRI Database | AI governance framework adoption | 2004–2022 |
    """)

    st.divider()

    st.markdown("## Key Findings")
    st.markdown("""
    1. **Governance dominates risk** — Civil liberties, rule of law, and civic space explain 51.6% of cross-national AWPRI variance (PC1)
    2. **Three country archetypes** — Critical (China), High Risk (Global South), Moderate (OECD)
    3. **The welfare-productivity paradox** — Countries with strong animal rights laws sometimes show higher AWPRI due to industrial scale overwhelming oversight capacity
    4. **AI amplification is the largest layer** — Mean L3=0.55 vs L1=0.42, L2=0.41 globally
    5. **Global bifurcation by 2030** — OECD improving, non-OECD worsening; gap structural not cyclical
    """)

    st.divider()
    st.markdown("## Citation")
    st.code("AWPRI Project (2025). Animal Welfare and Policy Risk Index: An AI/ML-driven prototype. Futurekind AI Fellowship.", language=None)
