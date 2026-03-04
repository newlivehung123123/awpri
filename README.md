# 🐾 AWPRI — Animal Welfare and Policy Risk Index

An AI/ML-driven prototype for assessing animal welfare policy risk across 25 countries (2004–2030). Built for the **Futurekind AI Fellowship**.

## 🚀 Live Dashboard
**[Launch AWPRI Dashboard →](https://awpri-dashboard.streamlit.app)**

## What It Does
- **Nowcasting** — real-time risk scores for any country, any year (2004–2022)
- **Policy Simulation** — models impact of 7 policy interventions on risk trajectory to 2030
- **Forecasting** — ARIMA time-series projections to 2030 with 95% confidence intervals
- **ML Analysis** — PCA, K-Means clustering, Random Forest feature importance

## Key Findings
1. **Governance dominates risk** — Civil liberties and rule of law explain 51.6% of cross-national AWPRI variance (PC1)
2. **Three country archetypes** — Critical Risk (China), High Risk (Global South), Moderate Risk (OECD)
3. **Welfare-productivity paradox** — Countries with strong animal rights laws paradoxically show higher AWPRI due to industrial scale overwhelming oversight capacity
4. **AI amplification is the largest risk layer** — Mean L3=0.55 vs L1=0.42, L2=0.41 globally
5. **Global bifurcation by 2030** — OECD nations improving, non-OECD worsening; gap is structural

## Dataset
| | |
|---|---|
| Countries | 25 |
| Years | 2004–2022 (19 years) |
| Observations | 475 |
| Variables | 15 across 3 layers |
| Missing values | 0 |

**Data sources:** V-Dem v15, FAOSTAT, Google Trends, OpenAlex

## Index Structure
| Layer | Variables | Description |
|---|---|---|
| L1 — Current State | 5 | Agricultural intensity + baseline governance |
| L2 — Policy Trajectory | 5 | Direction of policy change + civil society capacity |
| L3 — AI Amplification | 5 | How AI adoption and governance gaps amplify risk |

## ML Pipeline
- **PCA** — PC1 (51.6% variance) captures governance risk dimension
- **K-Means** — k=3 archetypes optimised by silhouette score
- **Random Forest** — civil liberties risk is top predictor (49.2% importance)
- **ARIMA** — per-country forecasting to 2030 with 95% CI

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
awpri-pipeline/
├── app.py                               # Streamlit dashboard (5 pages)
├── src/
│   ├── nowcast.py                       # Nowcasting engine
│   ├── policy_sim.py                    # Policy simulation engine
│   ├── panel_ml.py                      # ML pipeline
│   └── panel_config.py                  # Configuration
├── data/
│   ├── final/panel_awpri_normalized.csv # Main panel dataset (475 × 22)
│   └── ml/                              # ML outputs + model files
└── requirements.txt
```

## Citation
```
AWPRI Project (2025). Animal Welfare and Policy Risk Index:
An AI/ML-driven prototype. Futurekind AI Fellowship.
GitHub: https://awpri-dashboard.streamlit.app
```
