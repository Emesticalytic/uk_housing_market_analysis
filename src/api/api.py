
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app    = FastAPI(title="UK Housing Price API",
                 description="Predict average house price for a UK region",
                 version="1.0")
model  = joblib.load(Path("models/housing_model.pkl"))
scaler = joblib.load(Path("models/housing_scaler.pkl"))
feats  = joblib.load(Path("models/housing_features.pkl"))

class HousingInput(BaseModel):
    price_lag_1q:          float
    price_lag_2q:          float
    price_lag_4q:          float
    price_ma4q:            float
    price_vol4q:           float
    price_to_income:       float
    affordability_stress:  float
    base_rate:             float
    mortgage_rate:         float
    unemployment_rate:     float
    gdp_growth_qoq:        float
    construction_index:    float
    median_income:         float
    region_code:           int
    trend:                 float
    quarter_sin:           float
    quarter_cos:           float

class HousingOutput(BaseModel):
    predicted_price:  float
    confidence_lower: float
    confidence_upper: float

@app.get("/health")
def health():
    return {"status": "healthy", "model": "GradientBoosting v1.0"}

@app.post("/predict", response_model=HousingOutput)
def predict(data: HousingInput):
    try:
        X = np.array([[getattr(data, f) for f in feats]])
        X_sc = scaler.transform(X)
        pred = model.predict(X_sc)[0]
        return HousingOutput(
            predicted_price  = round(pred, 0),
            confidence_lower = round(pred * 0.92, 0),
            confidence_upper = round(pred * 1.08, 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
