"""
AWPRI Panel — ML Model
======================
1. PCA — dimensionality reduction + variance explanation
2. K-Means clustering — country risk archetypes
3. Random Forest — predict AWPRI from variables (feature importance)
4. Panel Fixed Effects regression — which variables drive risk change
5. ARIMA forecasting — project AWPRI to 2030 per country

Outputs saved to data/ml/
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime

warnings.filterwarnings('ignore')

# ── ML libraries ──────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Disable loky for sandbox
import joblib

# statsmodels for panel regression and ARIMA
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_OK = True
except (ImportError, PermissionError) as e:
    STATSMODELS_OK = False
    print(f"WARNING: statsmodels unavailable ({type(e).__name__}). Panel regression and ARIMA will be skipped.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_config import NAMES, ISO2_LIST, YEARS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR   = os.path.join(BASE_DIR, "data", "ml")
FIG_DIR  = os.path.join(ML_DIR, "figures")
os.makedirs(ML_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

NORM_PATH = os.path.join(BASE_DIR, "data", "final", "panel_awpri_normalized.csv")
WIDE_PATH = os.path.join(BASE_DIR, "data", "final", "panel_awpri_wide.csv")

# ── Variable groups ───────────────────────────────────────────────────────────
TIME_VARYING = [
    "farmed_animals_per_capita",
    "aquaculture_pct",
    "animal_rights_risk",
    "rule_of_law_risk",
    "animal_rights_delta_risk",
    "plant_protein_risk",
    "civic_space_risk",
    "civil_liberties_risk",
    "ai_governance_risk",
]

TIME_INVARIANT = [
    "meat_consumption_kg",
    "public_concern_risk",
    "ai_aw_research_risk",
    "ai_sentience_risk",
    "speciesist_bias_ratio",
    "patent_intensity",
]

ALL_VARS = TIME_VARYING + TIME_INVARIANT

# Country region mapping for visualization
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

REGION_COLORS = {
    "Europe":    "#2196F3",
    "N.America": "#4CAF50",
    "L.America": "#FF9800",
    "Asia":      "#F44336",
    "Oceania":   "#9C27B0",
    "Africa":    "#795548",
}

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    norm = pd.read_csv(NORM_PATH)
    wide = pd.read_csv(WIDE_PATH)
    print(f"Loaded norm: {norm.shape} | wide: {wide.shape}")
    return norm, wide

# ── 1. PCA ────────────────────────────────────────────────────────────────────

def run_pca(norm):
    print("\n" + "="*60)
    print("1. PRINCIPAL COMPONENT ANALYSIS")
    print("="*60)

    # Use 2022 cross-section for PCA (most recent, all vars available)
    df_2022 = norm[norm["year"] == 2022][["country_iso2"] + ALL_VARS].copy()
    df_2022 = df_2022.dropna()

    X = df_2022[ALL_VARS].values
    countries = df_2022["country_iso2"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Variance explained
    var_exp = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_exp)

    print(f"\nVariance explained by component:")
    n_components = None
    for i, (v, cv) in enumerate(zip(var_exp, cum_var)):
        print(f"  PC{i+1}: {v*100:.1f}% (cumulative: {cv*100:.1f}%)")
        if cv > 0.85 and n_components is None:
            print(f"  → {i+1} components explain >85% variance")
            n_components = i + 1

    # Component loadings
    loadings = pd.DataFrame(
        pca.components_[:4].T,
        index=ALL_VARS,
        columns=[f"PC{i+1}" for i in range(4)]
    )
    print(f"\nTop variable loadings on PC1 (dominant risk dimension):")
    pc1_sorted = loadings["PC1"].abs().sort_values(ascending=False)
    for var, loading in pc1_sorted.head(6).items():
        direction = "↑ risk" if loadings.loc[var, "PC1"] > 0 else "↓ risk"
        print(f"  {var}: {loadings.loc[var, 'PC1']:.3f} ({direction})")

    # Save loadings
    loadings.to_csv(os.path.join(ML_DIR, "pca_loadings.csv"))

    # Plot: scree + biplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scree plot
    axes[0].bar(range(1, len(var_exp)+1), var_exp*100, alpha=0.7, color="#2196F3")
    axes[0].plot(range(1, len(cum_var)+1), cum_var*100, "r-o", markersize=4)
    axes[0].axhline(85, color="gray", linestyle="--", alpha=0.5, label="85% threshold")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_title("PCA Scree Plot")
    axes[0].legend()

    # PC1 vs PC2 scatter
    region_colors = [REGION_COLORS[REGIONS[c]] for c in countries]
    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=region_colors, s=120, zorder=5)
    for i, iso2 in enumerate(countries):
        axes[1].annotate(iso2, (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=8, ha="center", va="bottom", xytext=(0, 5),
                        textcoords="offset points")
    # Legend
    for region, color in REGION_COLORS.items():
        axes[1].scatter([], [], c=color, label=region, s=60)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)")
    axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)")
    axes[1].set_title("Countries in PCA Space (2022)")
    axes[1].axhline(0, color="gray", alpha=0.3)
    axes[1].axvline(0, color="gray", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPCA figure saved → data/ml/figures/pca.png")

    # Save PCA scores
    pca_scores = pd.DataFrame(X_pca[:, :4],
                               columns=[f"PC{i+1}" for i in range(4)])
    pca_scores.insert(0, "country_iso2", countries)
    pca_scores.to_csv(os.path.join(ML_DIR, "pca_scores_2022.csv"), index=False)

    return pca, scaler, X_pca, countries, n_components

# ── 2. K-Means Clustering ─────────────────────────────────────────────────────

def run_clustering(norm, pca, scaler):
    print("\n" + "="*60)
    print("2. K-MEANS CLUSTERING — RISK ARCHETYPES")
    print("="*60)

    df_2022 = norm[norm["year"] == 2022][["country_iso2"] + ALL_VARS].dropna()
    X = df_2022[ALL_VARS].values
    countries = df_2022["country_iso2"].values
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)[:, :4]

    # Find optimal k using silhouette score
    sil_scores = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_pca)
        sil_scores[k] = silhouette_score(X_pca, labels)
        print(f"  k={k}: silhouette={sil_scores[k]:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\nOptimal k={best_k} (silhouette={sil_scores[best_k]:.4f})")

    # Fit final model
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = km_final.fit_predict(X_pca)

    # Cluster profiles
    df_2022 = df_2022.copy()
    df_2022["cluster"] = labels
    df_2022["AWPRI_score"] = norm[norm["year"]==2022]["AWPRI_score"].values

    print(f"\nCluster profiles (mean AWPRI score):")
    cluster_means = df_2022.groupby("cluster")["AWPRI_score"].agg(["mean","count"])
    cluster_means = cluster_means.sort_values("mean", ascending=False)

    # Label clusters by risk level
    cluster_labels = {}
    risk_levels = ["Critical Risk", "High Risk", "Moderate Risk", "Lower Risk", "Minimal Risk"]
    for i, (cluster_id, row) in enumerate(cluster_means.iterrows()):
        label = risk_levels[min(i, len(risk_levels)-1)]
        cluster_labels[cluster_id] = label
        countries_in = df_2022[df_2022["cluster"]==cluster_id]["country_iso2"].tolist()
        print(f"  Cluster {cluster_id} — {label}: AWPRI={row['mean']:.3f} | {countries_in}")

    df_2022["risk_archetype"] = df_2022["cluster"].map(cluster_labels)

    # Save clustering results
    cluster_out = df_2022[["country_iso2", "cluster", "risk_archetype", "AWPRI_score"]].copy()
    cluster_out["country_name"] = cluster_out["country_iso2"].map(NAMES)
    cluster_out.to_csv(os.path.join(ML_DIR, "clusters_2022.csv"), index=False)

    # Plot clusters
    X_pca_full = pca.transform(scaler.transform(X))[:, :2]
    colors = cm.Set1(np.linspace(0, 0.8, best_k))

    fig, ax = plt.subplots(figsize=(12, 8))
    for cluster_id in range(best_k):
        mask = labels == cluster_id
        ax.scatter(X_pca_full[mask, 0], X_pca_full[mask, 1],
                  c=[colors[cluster_id]], s=150, label=cluster_labels[cluster_id],
                  zorder=5, edgecolors="white", linewidth=1)
        for i in np.where(mask)[0]:
            ax.annotate(countries[i], (X_pca_full[i, 0], X_pca_full[i, 1]),
                       fontsize=9, ha="center", va="bottom",
                       xytext=(0, 6), textcoords="offset points")

    ax.set_xlabel("PC1 (Governance & Welfare Capacity)")
    ax.set_ylabel("PC2 (Agricultural Intensity)")
    ax.set_title("AWPRI Country Risk Archetypes — K-Means Clustering (2022)")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nClustering figure saved → data/ml/figures/clusters.png")

    # Save model
    joblib.dump(km_final, os.path.join(ML_DIR, "kmeans_model.pkl"))
    joblib.dump({"pca": pca, "scaler": scaler, "cluster_labels": cluster_labels,
                 "best_k": best_k, "all_vars": ALL_VARS},
                os.path.join(ML_DIR, "model_metadata.pkl"))

    return km_final, labels, cluster_labels, df_2022

# ── 3. Random Forest — Feature Importance ────────────────────────────────────

def run_random_forest(norm):
    print("\n" + "="*60)
    print("3. RANDOM FOREST — FEATURE IMPORTANCE")
    print("="*60)

    # Use full panel (all years, all countries)
    df = norm[["country_iso2", "year"] + ALL_VARS + ["AWPRI_score"]].dropna()

    X = df[ALL_VARS].values
    y = df["AWPRI_score"].values

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1  # Disable parallel processing for sandbox compatibility
    )

    cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring="r2")
    print(f"  Cross-validated R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full data for feature importance
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=ALL_VARS)
    importances = importances.sort_values(ascending=False)

    print(f"\nFeature importance ranking:")
    for var, imp in importances.items():
        bar = "█" * int(imp * 100)
        tv = "TV" if var in TIME_VARYING else "TI"
        print(f"  [{tv}] {var}: {imp:.4f} {bar}")

    # Save
    importances.to_csv(os.path.join(ML_DIR, "feature_importance.csv"), header=["importance"])
    joblib.dump(rf, os.path.join(ML_DIR, "rf_model.pkl"))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2196F3" if v in TIME_VARYING else "#FF9800" for v in importances.index]
    bars = ax.barh(range(len(importances)), importances.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index, fontsize=9)
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title("AWPRI Variable Importance\n(Blue=Time-varying, Orange=Time-invariant)")
    ax.invert_yaxis()
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
               f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFeature importance figure saved → data/ml/figures/feature_importance.png")

    return rf, importances

# ── 4. Panel Fixed Effects Regression ────────────────────────────────────────

def run_panel_regression(norm):
    print("\n" + "="*60)
    print("4. PANEL FIXED EFFECTS REGRESSION")
    print("="*60)

    if not STATSMODELS_OK:
        print("  SKIPPED — statsmodels not available")
        return None

    df = norm[["country_iso2", "year"] + TIME_VARYING + ["AWPRI_score"]].dropna().copy()

    # Demean within country (within-estimator = fixed effects)
    df_fe = df.copy()
    for col in TIME_VARYING + ["AWPRI_score"]:
        country_means = df_fe.groupby("country_iso2")[col].transform("mean")
        df_fe[f"{col}_dm"] = df_fe[col] - country_means

    y = df_fe["AWPRI_score_dm"].values
    X_cols = [f"{v}_dm" for v in TIME_VARYING]
    X = sm.add_constant(df_fe[X_cols].values)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    print(f"\n  Fixed Effects OLS Results:")
    print(f"  R² (within): {model.rsquared:.4f}")
    print(f"  N observations: {len(y)}")
    print(f"\n  Coefficients (demeaned):")
    for i, var in enumerate(TIME_VARYING):
        coef = model.params[i+1]
        pval = model.pvalues[i+1]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        direction = "↑ risk" if coef > 0 else "↓ risk"
        print(f"  {var}: β={coef:.4f} p={pval:.3f} {sig} ({direction})")

    # Save results
    results_df = pd.DataFrame({
        "variable": TIME_VARYING,
        "coefficient": model.params[1:],
        "p_value": model.pvalues[1:],
        "significant": model.pvalues[1:] < 0.05
    })
    results_df.to_csv(os.path.join(ML_DIR, "panel_regression.csv"), index=False)

    return model

# ── 5. ARIMA Forecasting to 2030 ─────────────────────────────────────────────

def run_forecasting(norm):
    print("\n" + "="*60)
    print("5. ARIMA FORECASTING TO 2030")
    print("="*60)

    if not STATSMODELS_OK:
        print("  SKIPPED — statsmodels not available")
        return None

    forecast_years = list(range(2023, 2031))
    all_forecasts = []

    fig, axes = plt.subplots(5, 5, figsize=(22, 18))
    axes_flat = axes.flatten()

    for idx, iso2 in enumerate(ISO2_LIST):
        country_data = norm[norm["country_iso2"] == iso2].sort_values("year")
        ts = country_data["AWPRI_score"].values
        years = country_data["year"].values

        ax = axes_flat[idx]

        try:
            # Test stationarity
            adf_result = adfuller(ts, autolag="AIC")
            is_stationary = adf_result[1] < 0.05
            d = 0 if is_stationary else 1

            # Fit ARIMA
            model = ARIMA(ts, order=(1, d, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=len(forecast_years))
            conf_int = fitted.get_forecast(steps=len(forecast_years)).conf_int()

            # Clamp forecasts to [0,1]
            forecast = np.clip(forecast, 0, 1)
            lower = np.clip(conf_int.iloc[:, 0].values, 0, 1)
            upper = np.clip(conf_int.iloc[:, 1].values, 0, 1)

            # Direction
            trend = "↑" if forecast[-1] > ts[-1] else "↓"
            change = forecast[-1] - ts[-1]

            print(f"  {iso2}: 2022={ts[-1]:.3f} → 2030={forecast[-1]:.3f} "
                  f"({trend}{abs(change):.3f}) ARIMA(1,{d},1)")

            for yr, fc in zip(forecast_years, forecast):
                all_forecasts.append({
                    "country_iso2": iso2,
                    "country_name": NAMES[iso2],
                    "year": yr,
                    "AWPRI_forecast": round(fc, 4),
                    "lower_95": round(lower[forecast_years.index(yr)], 4),
                    "upper_95": round(upper[forecast_years.index(yr)], 4),
                    "trend": trend,
                })

            # Plot
            ax.plot(years, ts, "b-o", markersize=3, linewidth=1.5, label="Historical")
            ax.plot(forecast_years, forecast, "r--o", markersize=3,
                   linewidth=1.5, label="Forecast")
            ax.fill_between(forecast_years, lower, upper, alpha=0.2, color="red")
            ax.axvline(2022, color="gray", linestyle=":", alpha=0.5)

        except Exception as e:
            print(f"  {iso2}: ARIMA failed ({e}) — using linear trend")
            # Fallback: linear trend
            z = np.polyfit(range(len(ts)), ts, 1)
            p = np.poly1d(z)
            forecast = np.clip([p(len(ts) + i) for i in range(len(forecast_years))], 0, 1)
            ax.plot(years, ts, "b-o", markersize=3, linewidth=1.5)
            ax.plot(forecast_years, forecast, "r--", linewidth=1.5)
            for yr, fc in zip(forecast_years, forecast):
                all_forecasts.append({
                    "country_iso2": iso2,
                    "country_name": NAMES[iso2],
                    "year": yr,
                    "AWPRI_forecast": round(fc, 4),
                    "lower_95": round(max(0, fc - 0.05), 4),
                    "upper_95": round(min(1, fc + 0.05), 4),
                    "trend": "↑" if forecast[-1] > ts[-1] else "↓",
                })

        ax.set_title(f"{iso2} — {NAMES[iso2]}", fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)

    plt.suptitle("AWPRI Forecasts to 2030 — All 25 Countries", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "forecasts_2030.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save forecast data
    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df.to_csv(os.path.join(ML_DIR, "forecasts_2030.csv"), index=False)
    print(f"\nForecasting figures saved → data/ml/figures/forecasts_2030.png")
    print(f"Forecast data saved → data/ml/forecasts_2030.csv")

    # Print 2030 leaderboard
    f2030 = forecast_df[forecast_df["year"] == 2030].sort_values(
        "AWPRI_forecast", ascending=False).reset_index(drop=True)
    f2030.index += 1
    print(f"\n2030 Projected Rankings:")
    for i, r in f2030.iterrows():
        print(f"  {i:2d}. {r['country_iso2']} {r['AWPRI_forecast']:.4f} {r['trend']}")

    return forecast_df

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print(f"AWPRI ML MODEL — {datetime.now().isoformat()}")
    print("="*60)

    norm, wide = load_data()

    # 1. PCA
    pca, scaler, X_pca, countries, n_components = run_pca(norm)

    # 2. Clustering
    km_model, labels, cluster_labels, cluster_df = run_clustering(norm, pca, scaler)

    # 3. Random Forest
    rf_model, importances = run_random_forest(norm)

    # 4. Panel regression
    panel_model = run_panel_regression(norm)

    # 5. Forecasting
    forecast_df = run_forecasting(norm)

    print("\n" + "="*60)
    print("ML MODEL COMPLETE")
    print(f"Outputs saved to: data/ml/")
    print("  pca_loadings.csv")
    print("  pca_scores_2022.csv")
    print("  clusters_2022.csv")
    print("  feature_importance.csv")
    print("  panel_regression.csv")
    print("  forecasts_2030.csv")
    print("  kmeans_model.pkl")
    print("  rf_model.pkl")
    print("  model_metadata.pkl")
    print("  figures/pca.png")
    print("  figures/clusters.png")
    print("  figures/feature_importance.png")
    print("  figures/forecasts_2030.png")
    print("="*60)

if __name__ == "__main__":
    main()
