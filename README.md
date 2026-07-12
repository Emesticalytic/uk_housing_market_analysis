# UK Regional Housing Market Analysis

End-to-end machine learning pipeline analysing UK regional house prices from 2010–2024. Covers data ingestion, EDA, feature engineering, model training, SHAP explainability, and a live Streamlit dashboard with a FastAPI prediction endpoint.

## Key Findings

- **London** price-to-income ratio rose from 12.5x (2010) to 19x (2024) — an entire generation priced out
- **Scotland** and **North East** remain the most affordable regions (ratio < 5x)
- Mortgage rate spikes post-2022 suppressed growth in all regions except London
- Lagged price features dominate predictive power (SHAP analysis confirms)


## Tech Stack

| Layer | Tool |
|---|---|
| Data | ONS House Price Index API |
| EDA | pandas, matplotlib, seaborn, plotly |
| Modelling | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI + uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
