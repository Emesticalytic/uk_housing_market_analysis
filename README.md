# UK Regional Housing Market Analysis

End-to-end machine learning pipeline analysing UK regional house prices from 2010–2024. Covers data ingestion, EDA, feature engineering, model training, SHAP explainability, and a live Streamlit dashboard with a FastAPI prediction endpoint.

---

## Project Structure

```
UK Housing Analysis/
├── notebooks/
│   ├── analysis_code.ipynb          # Full end-to-end analysis (46 cells)
│   ├── 01_eda/01_eda.ipynb
│   ├── 02_feature_engineering/
│   ├── 03_modelling/
│   ├── 04_evaluation/
│   ├── data/
│   │   └── uk_housing_data.csv      # ONS HPI — 9 regions, 2010–2024
│   ├── models/
│   │   ├── housing_model.pkl        # Ridge Regression (R² = 0.994)
│   │   ├── housing_scaler.pkl
│   │   └── housing_features.pkl
│   ├── reports/
│   │   ├── affordability_heatmap.png
│   │   ├── affordability_league.png
│   │   ├── regional_price_trajectories.png
│   │   ├── shap_analysis.png
│   │   ├── rate_vs_growth_animated.html
│   │   └── regional_forecast.html
│   ├── src/
│   │   ├── dashboard.py             # Streamlit dashboard
│   │   └── api.py                   # FastAPI prediction endpoint
│   ├── docker/
│   │   └── Dockerfile.api
│   ├── mlruns/                      # MLflow experiment tracking
│   └── requirements.txt
├── infrastructure/
│   └── docker/
│       ├── Dockerfile.api
│       ├── Dockerfile.dashboard
│       └── Dockerfile.pipeline
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Results

| Model | CV R² | CV MAE |
|---|---|---|
| Ridge Regression | **0.994** | **£9,094** |
| Gradient Boosting | 0.935 | £25,770 |
| Random Forest | 0.930 | — |

Ridge Regression won — lagged price features are highly linear in this dataset.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
cd notebooks/src
streamlit run dashboard.py
# → http://localhost:8501
```

### 3. Run the FastAPI prediction service

```bash
cd notebooks/src
uvicorn api:app --reload --port 8000
# → http://localhost:8000/docs
```

### 4. Run the full notebook

Open `notebooks/analysis_code.ipynb` in Jupyter and run all cells.

---

## Docker

### Run all services

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Prediction API | http://localhost:8000 |
| MLflow Tracking Server | http://localhost:5000 |

### Run training pipeline only

```bash
docker compose --profile train up pipeline
```

### Individual builds

```bash
# API only
docker build -f infrastructure/docker/Dockerfile.api -t housing-api .

# Dashboard only
docker build -f infrastructure/docker/Dockerfile.dashboard -t housing-dashboard .
```

---

## API Usage

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "price_lag_1q": 850000,
    "price_lag_2q": 840000,
    "price_lag_4q": 820000,
    "price_ma4q": 845000,
    "price_vol4q": 12000,
    "price_to_income": 16.5,
    "affordability_stress": 0.45,
    "base_rate": 5.25,
    "mortgage_rate": 5.5,
    "unemployment_rate": 4.2,
    "gdp_growth_qoq": 0.3,
    "construction_index": 110,
    "median_income": 52000,
    "region_code": 0,
    "trend": 60,
    "quarter_sin": 1.0,
    "quarter_cos": 0.0
  }'
```

Response:
```json
{
  "predicted_price": 862500,
  "confidence_lower": 793500,
  "confidence_upper": 931500
}
```

---

## Key Findings

- **London** price-to-income ratio rose from 12.5x (2010) to 19x (2024) — an entire generation priced out
- **Scotland** and **North East** remain the most affordable regions (ratio < 5x)
- Mortgage rate spikes post-2022 suppressed growth in all regions except London
- Lagged price features dominate predictive power (SHAP analysis confirms)

---

## Environment Variables

Copy `.env.example` to `.env` before running Docker:

```bash
cp .env.example .env
```

---

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
