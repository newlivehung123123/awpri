"""
AWPRI Dashboard — Streamlit App (Enhanced Interactivity)
================================
Animal Welfare and Policy Risk Index
AI/ML-driven prototype for the Futurekind AI Fellowship

Pages:
  1. Global Overview    — world map + rankings + animation
  2. Country Deep-Dive  — variable breakdown + radar chart + trajectory + compare mode
  3. Policy Simulator   — intervention sliders + impact projection + advanced mode
  4. Forecasts 2030     — trajectory charts per country + multi-country overlay
  5. Methodology        — variable definitions + data sources
"""

import os, sys, time
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

# ISO2 to ISO3 mapping for choropleth
ISO2_TO_ISO3 = {
    "AR":"ARG","AU":"AUS","BR":"BRA","CA":"CAN","CN":"CHN",
    "DE":"DEU","DK":"DNK","ES":"ESP","FR":"FRA","GB":"GBR",
    "IN":"IND","IT":"ITA","JP":"JPN","KE":"KEN","KR":"KOR",
    "MX":"MEX","NG":"NGA","NL":"NLD","NZ":"NZL","PL":"POL",
    "SE":"SWE","TH":"THA","US":"USA","VN":"VNM","ZA":"ZAF",
}

# ── Colour palette ────────────────────────────────────────────────────────────
RISK_COLORS = {
    "Critical Risk": "#d32f2f",
    "High Risk":     "#f57c00",
    "Moderate Risk": "#fbc02d",
    "Lower Risk":    "#388e3c",
    "Minimal Risk":  "#1b5e20",
}

# ── Initialize session state ────────────────────────────────────────────────────
if "selected_country" not in st.session_state:
    st.session_state.selected_country = "VN"
if "selected_year" not in st.session_state:
    st.session_state.selected_year = 2022
if "compare_country" not in st.session_state:
    st.session_state.compare_country = "CN"
if "policy_intensities" not in st.session_state:
    st.session_state.policy_intensities = {pid: 100 for pid in POLICIES.keys()}

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

    col_year, col_play = st.columns([3, 1])
    
    with col_year:
        selected_year = st.select_slider(
            "Select year", options=YEARS_HIST, value=st.session_state.selected_year, key="year_slider"
        )
        st.session_state.selected_year = selected_year
    
    with col_play:
        if st.button("▶ Play Animation", key="play_btn"):
            with st.spinner("Animating 2004→2022..."):
                anim_placeholder = st.empty()
                for yr in YEARS_HIST:
                    anim_placeholder.write(f"**Year: {yr}**")
                    time.sleep(0.3)
                st.session_state.selected_year = selected_year

    year_df = norm[norm["year"] == selected_year].merge(
        clusters[["country_iso2","risk_archetype"]], on="country_iso2", how="left"
    ).sort_values("AWPRI_score", ascending=False).reset_index(drop=True)
    year_df.index += 1
    year_df["country_name"] = year_df["country_iso2"].map(COUNTRY_NAMES)
    year_df["iso3"] = year_df["country_iso2"].map(ISO2_TO_ISO3)

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
        
        fig_map = px.choropleth(
            year_df,
            locations="iso3",
            locationmode="ISO-3",
            color="AWPRI_score",
            hover_name="country_name",
            hover_data={
                "iso3": False,
                "AWPRI_score":  ":.3f",
                "L1_score":     ":.3f",
                "L2_score":     ":.3f",
                "L3_score":     ":.3f",
                "country_iso2": True,
                "risk_archetype": True,
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
            clickmode="event+select",
        )
        st.plotly_chart(fig_map, use_container_width=True, key="global_map")

        # ── Download rankings ──
        csv_rankings = year_df[["country_iso2", "country_name", "AWPRI_score", "L1_score", "L2_score", "L3_score", "risk_archetype"]].to_csv(index=False)
        st.download_button(
            label="📥 Download Rankings CSV",
            data=csv_rankings,
            file_name=f"awpri_rankings_{selected_year}.csv",
            mime="text/csv",
            key="download_rankings"
        )

    with col_rank:
        st.subheader(f"Risk Rankings — {selected_year}")
        for i, row in year_df.iterrows():
            archetype = row.get("risk_archetype", "Moderate Risk")
            color = RISK_COLORS.get(archetype, "#9e9e9e")
            emoji = RISK_TIER_LABELS.get(archetype, ("⚪",))[0]
            
            # Clickable country card
            if st.button(
                f"{emoji} **{i}. {row['country_iso2']}** {row['country_name']}\n"
                f"AWPRI: {row['AWPRI_score']:.3f}",
                key=f"rank_btn_{row['country_iso2']}"
            ):
                st.session_state.selected_country = row['country_iso2']
                st.success(f"Selected {row['country_name']} — go to Country Deep-Dive page")

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
    st.plotly_chart(fig_layers, use_container_width=True, key="layer_breakdown")

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
    st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COUNTRY DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Country Deep-Dive":
    st.title("🔍 Country Deep-Dive")

    col_sel1, col_sel2, col_compare = st.columns([2, 1, 1])
    
    with col_sel1:
        selected_country = st.selectbox(
            "Select country",
            ALL_COUNTRIES,
            index=ALL_COUNTRIES.index(st.session_state.selected_country) if st.session_state.selected_country in ALL_COUNTRIES else 0,
            format_func=lambda x: f"{x} — {COUNTRY_NAMES.get(x, x)}",
            key="country_select"
        )
        st.session_state.selected_country = selected_country
    
    with col_sel2:
        selected_year = st.selectbox("Year", YEARS_HIST, index=len(YEARS_HIST)-1, key="country_year")
        st.session_state.selected_year = selected_year
    
    with col_compare:
        compare_mode = st.toggle("Compare Mode", key="compare_toggle")

    # ── Get baseline data ──
    result = nc.nowcast(selected_country, selected_year)
    if "error" in result:
        st.error(result["error"])
        st.stop()

    tier_emoji, tier_label, tier_color = result["risk_tier"]

    # ── Single country or compare ──
    if not compare_mode:
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
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_single")

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

        # ── Download button ──
        csv_vars = var_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Variable Data CSV",
            data=csv_vars,
            file_name=f"awpri_variables_{selected_country}_{selected_year}.csv",
            mime="text/csv",
            key="download_vars"
        )

        st.divider()

        # ── Historical trajectory ──
        st.subheader(f"Historical Trajectory — {result['country_name']}")
        traj = result["trajectory"]
        fc   = result["forecast_2030"]

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=list(traj.keys()), y=list(traj.values()),
            mode="lines+markers", name="Historical AWPRI",
            line=dict(color=tier_color, width=2),
            marker=dict(size=5),
        ))
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
        st.plotly_chart(fig_traj, use_container_width=True, key="traj_single")

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

    else:
        # ── COMPARE MODE ──
        st.subheader(f"Comparing {selected_country} with another country")
        
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            st.write(f"**Primary: {selected_country}**")
        with col_comp2:
            compare_country = st.selectbox(
                "Select country to compare",
                [c for c in ALL_COUNTRIES if c != selected_country],
                index=0,
                format_func=lambda x: f"{x} — {COUNTRY_NAMES.get(x, x)}",
                key="compare_country_select"
            )

        result2 = nc.nowcast(compare_country, selected_year)

        tier_emoji2, tier_label2, tier_color2 = result2["risk_tier"]

        # ── Metrics side by side ──
        col1_a, col2_a, col1_b, col2_b = st.columns(4)
        col1_a.metric(f"{selected_country} AWPRI", f"{result['awpri_score']:.3f}", 
                      delta=f"Rank {result['rank']}")
        col2_a.metric(f"{compare_country} AWPRI", f"{result2['awpri_score']:.3f}",
                      delta=f"Rank {result2['rank']}")
        col1_b.metric(f"{selected_country} Archetype", result["risk_archetype"])
        col2_b.metric(f"{compare_country} Archetype", result2["risk_archetype"])

        st.divider()

        # ── Radar charts side by side ──
        col_rad1, col_rad2 = st.columns(2)

        with col_rad1:
            st.subheader(f"{selected_country} Risk Profile")
            var_names  = [VAR_LABELS[v] for v in list(result["var_scores"].keys())]
            var_values = [result["var_scores"][v]["score"] for v in result["var_scores"]]
            global_avg = [result["var_scores"][v]["global_mean"] for v in result["var_scores"]]

            fig_radar1 = go.Figure()
            fig_radar1.add_trace(go.Scatterpolar(
                r=var_values + [var_values[0]],
                theta=var_names + [var_names[0]],
                fill="toself", name=result["country_name"],
                line_color=tier_color, fillcolor=tier_color, opacity=0.6,
            ))
            fig_radar1.add_trace(go.Scatterpolar(
                r=global_avg + [global_avg[0]],
                theta=var_names + [var_names[0]],
                fill="toself", name="Global Avg",
                line_color="#9e9e9e", fillcolor="#9e9e9e", opacity=0.3,
            ))
            fig_radar1.update_layout(height=400, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(fig_radar1, use_container_width=True, key="radar_compare1")

        with col_rad2:
            st.subheader(f"{compare_country} Risk Profile")
            var_names2  = [VAR_LABELS[v] for v in list(result2["var_scores"].keys())]
            var_values2 = [result2["var_scores"][v]["score"] for v in result2["var_scores"]]
            global_avg2 = [result2["var_scores"][v]["global_mean"] for v in result2["var_scores"]]

            fig_radar2 = go.Figure()
            fig_radar2.add_trace(go.Scatterpolar(
                r=var_values2 + [var_values2[0]],
                theta=var_names2 + [var_names2[0]],
                fill="toself", name=result2["country_name"],
                line_color=tier_color2, fillcolor=tier_color2, opacity=0.6,
            ))
            fig_radar2.add_trace(go.Scatterpolar(
                r=global_avg2 + [global_avg2[0]],
                theta=var_names2 + [var_names2[0]],
                fill="toself", name="Global Avg",
                line_color="#9e9e9e", fillcolor="#9e9e9e", opacity=0.3,
            ))
            fig_radar2.update_layout(height=400, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(fig_radar2, use_container_width=True, key="radar_compare2")

        # ── Trajectory comparison ──
        st.divider()
        st.subheader(f"Trajectory Comparison: {selected_country} vs {compare_country}")

        traj1 = result["trajectory"]
        traj2 = result2["trajectory"]

        fig_traj_comp = go.Figure()
        fig_traj_comp.add_trace(go.Scatter(
            x=list(traj1.keys()), y=list(traj1.values()),
            mode="lines+markers", name=f"{selected_country}",
            line=dict(color=tier_color, width=2),
        ))
        fig_traj_comp.add_trace(go.Scatter(
            x=list(traj2.keys()), y=list(traj2.values()),
            mode="lines+markers", name=f"{compare_country}",
            line=dict(color=tier_color2, width=2),
        ))
        fig_traj_comp.add_vline(x=2022, line_dash="dot", line_color="gray")
        fig_traj_comp.update_layout(
            height=350, yaxis=dict(range=[0,1], title="AWPRI Score"),
            xaxis=dict(title="Year"),
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_traj_comp, use_container_width=True, key="traj_compare")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — POLICY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Policy Simulator":
    st.title("⚙️ Policy Impact Simulator")
    st.caption("Simulate the effect of policy interventions on animal welfare risk")

    col_sel1, col_adv = st.columns([2, 1])
    with col_sel1:
        sim_country = st.selectbox(
            "Select country to simulate",
            ALL_COUNTRIES,
            index=ALL_COUNTRIES.index(st.session_state.selected_country) if st.session_state.selected_country in ALL_COUNTRIES else 0,
            format_func=lambda x: f"{x} — {COUNTRY_NAMES.get(x, x)}",
            key="sim_country_select"
        )
        st.session_state.selected_country = sim_country
    
    with col_adv:
        advanced_mode = st.toggle("Advanced Mode", key="advanced_toggle")

    if not advanced_mode:
        selected_policies = st.multiselect(
            "Select policy interventions",
            options=list(POLICIES.keys()),
            default=["ai_governance_framework"],
            format_func=lambda x: POLICIES[x]["label"],
            key="policy_multiselect"
        )
        policy_ids = selected_policies
    else:
        st.subheader("Policy Intensity Controls (0–100%)")
        policy_ids = []
        for pid in POLICIES.keys():
            intensity = st.slider(
                f"{POLICIES[pid]['label']}",
                min_value=0, max_value=100, value=100,
                key=f"policy_slider_{pid}"
            )
            if intensity > 0:
                policy_ids.append(pid)
            st.session_state.policy_intensities[pid] = intensity / 100.0

    if not policy_ids:
        st.info("Select at least one policy intervention above.")
        st.stop()

    # Run simulation
    sim_result = ps.simulate(sim_country, policy_ids)
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

        hist = norm[norm["country_iso2"]==sim_country].sort_values("year")
        hist_years  = list(hist["year"])
        hist_scores = list(hist["AWPRI_score"])

        fc_base = forecasts[forecasts["country_iso2"]==sim_country].sort_values("year")
        fc_years  = list(fc_base["year"])
        fc_scores = list(fc_base["AWPRI_forecast"])

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=hist_years, y=hist_scores,
            mode="lines+markers", name="Historical",
            line=dict(color="#555", width=2),
            marker=dict(size=4),
        ))
        fig_sim.add_trace(go.Scatter(
            x=[2022] + fc_years, y=[hist_scores[-1]] + fc_scores,
            mode="lines", name="Baseline (no policy)",
            line=dict(color="#f57c00", width=2, dash="dash"),
        ))
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
        st.plotly_chart(fig_sim, use_container_width=True, key="policy_trajectory")

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

    # ── Download trajectory ──
    csv_traj = pd.DataFrame({"Year": list(traj.keys()), "AWPRI": list(traj.values())}).to_csv(index=False)
    st.download_button(
        label="📥 Download Trajectory CSV",
        data=csv_traj,
        file_name=f"awpri_policy_trajectory_{sim_country}.csv",
        mime="text/csv",
        key="download_trajectory"
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

    col_multi, col_ci = st.columns([2, 1])
    
    with col_multi:
        multi_countries = st.multiselect(
            "Select countries to overlay (up to 6)",
            ALL_COUNTRIES,
            default=["CN", "VN", "BR", "US", "GB", "AU"],
            format_func=lambda x: f"{x} — {COUNTRY_NAMES.get(x, x)}",
            key="multi_select_forecast",
            max_selections=6
        )
    
    with col_ci:
        show_ci = st.toggle("Show 95% CI bands", value=True, key="show_ci_toggle")

    if multi_countries:
        st.subheader("Multi-Country Forecast Overlay")
        
        fig_multi = go.Figure()
        
        for iso2 in multi_countries:
            hist = norm[norm["country_iso2"]==iso2].sort_values("year")
            fc = forecasts[forecasts["country_iso2"]==iso2].sort_values("year")
            
            result_temp = nc.nowcast(iso2, 2022)
            _, _, color = result_temp["risk_tier"]
            
            # Historical
            fig_multi.add_trace(go.Scatter(
                x=list(hist["year"]), y=list(hist["AWPRI_score"]),
                mode="lines", name=f"{iso2} (historical)",
                line=dict(color=color, width=2),
            ))
            
            # Forecast
            if not fc.empty:
                fig_multi.add_trace(go.Scatter(
                    x=[2022]+list(fc["year"]),
                    y=[float(hist[hist["year"]==2022]["AWPRI_score"])]+list(fc["AWPRI_forecast"]),
                    mode="lines", name=f"{iso2} (forecast)",
                    line=dict(color=color, width=2, dash="dash"),
                ))
                
                # CI bands
                if show_ci:
                    fig_multi.add_trace(go.Scatter(
                        x=list(fc["year"])+list(fc["year"])[::-1],
                        y=list(fc["upper_95"])+list(fc["lower_95"])[::-1],
                        fill="toself", fillcolor=color,
                        opacity=0.1, line=dict(color="rgba(0,0,0,0)"),
                        name=f"{iso2} (95% CI)",
                        showlegend=False,
                    ))
        
        fig_multi.add_vline(x=2022, line_dash="dot", line_color="gray")
        fig_multi.update_layout(
            height=450, yaxis=dict(range=[0,1], title="AWPRI Score"),
            xaxis=dict(title="Year"),
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="v", yanchor="top", y=0.99),
        )
        st.plotly_chart(fig_multi, use_container_width=True, key="multi_forecast")

        # ── Download button ──
        fc_data = forecasts[forecasts["country_iso2"].isin(multi_countries)]
        csv_fc = fc_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data CSV",
            data=csv_fc,
            file_name="awpri_forecasts_2030.csv",
            mime="text/csv",
            key="download_forecast"
        )
    else:
        st.info("Select at least one country to display forecast overlay.")

    st.divider()

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
    st.plotly_chart(fig_f2030, use_container_width=True, key="rank_2030")


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
