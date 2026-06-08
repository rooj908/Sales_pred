"""
FastAPI Backend — BI Sales Forecasting Platform
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pickle, json, os
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(
    title="AI-Powered BI & Sales Forecasting API",
    description="Enterprise API for sales prediction, customer churn, and business intelligence.",
    version="1.0.0"
)

# ── Load models ──────────────────────────────────────────────────────────────
def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

sales_model   = load_model("models/saved/sales_model.pkl")
churn_model   = load_model("models/saved/churn_model.pkl")
forecast_model= load_model("models/saved/forecast_model.pkl")
scaler        = load_model("models/saved/scaler.pkl")

# ── Schemas ───────────────────────────────────────────────────────────────────
class SalesPredictionRequest(BaseModel):
    quantity: float
    unit_price: float
    discount: float = 0.0
    customer_age: int = 35
    customer_loyalty_years: float = 2.0
    marketing_spend: float = 1000.0
    competitor_price_index: float = 1.0
    month: int = 6
    quarter: int = 2
    product_encoded: int = 0
    region_encoded: int = 0
    segment_encoded: int = 0

class ChurnPredictionRequest(BaseModel):
    age: int
    total_orders: int
    total_spend: float
    loyalty_years: float
    satisfaction_score: float
    income_bracket_encoded: int = 1
    region_encoded: int = 0
    segment_encoded: int = 0

class ForecastRequest(BaseModel):
    months_ahead: int = 6

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "BI Sales Platform API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "sales_model": sales_model is not None,
            "churn_model": churn_model is not None,
            "forecast_model": forecast_model is not None,
        }
    }

@app.post("/predict/sales")
def predict_sales(req: SalesPredictionRequest):
    if sales_model is None:
        raise HTTPException(503, "Sales model not loaded. Run train_models.py first.")
    
    features = [[
        req.quantity, req.unit_price, req.discount, req.customer_age,
        req.customer_loyalty_years, req.marketing_spend, req.competitor_price_index,
        req.month, req.quarter, req.product_encoded, req.region_encoded, req.segment_encoded
    ]]
    
    if scaler:
        features = scaler.transform(features)
    
    prediction = sales_model.predict(features)[0]
    return {
        "predicted_revenue": round(float(prediction), 2),
        "currency": "USD",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/churn")
def predict_churn(req: ChurnPredictionRequest):
    if churn_model is None:
        raise HTTPException(503, "Churn model not loaded. Run train_models.py first.")
    
    features = [[
        req.age, req.total_orders, req.total_spend, req.loyalty_years,
        req.satisfaction_score, req.income_bracket_encoded,
        req.region_encoded, req.segment_encoded
    ]]
    
    prob = churn_model.predict_proba(features)[0][1]
    label = "High Risk" if prob > 0.6 else "Medium Risk" if prob > 0.3 else "Low Risk"
    
    return {
        "churn_probability": round(float(prob), 4),
        "risk_level": label,
        "recommendation": (
            "Immediate retention action needed" if prob > 0.6
            else "Monitor closely" if prob > 0.3
            else "Customer is stable"
        )
    }

@app.post("/forecast/revenue")
def forecast_revenue(req: ForecastRequest):
    if forecast_model is None:
        raise HTTPException(503, "Forecast model not loaded.")
    
    # Current time index assumed at 36 (end of 3 years of data)
    base_idx = 36
    future_indices = list(range(base_idx + 1, base_idx + req.months_ahead + 1))
    preds = forecast_model.predict([[i] for i in future_indices])
    
    results = []
    for i, rev in zip(future_indices, preds):
        results.append({
            "month_ahead": i - base_idx,
            "forecasted_revenue": round(float(max(rev, 0)), 2)
        })
    
    return {"forecast": results, "months_ahead": req.months_ahead}

@app.get("/metrics")
def get_metrics():
    metrics = {}
    for name, path in [("sales", "models/saved/sales_metrics.json"),
                       ("churn", "models/saved/churn_metrics.json")]:
        if os.path.exists(path):
            with open(path) as f:
                metrics[name] = json.load(f)
    return metrics

@app.get("/kpis")
def get_kpis():
    try:
        df = pd.read_csv("data/processed/sales_clean.csv")
        return {
            "total_revenue": round(df["revenue"].sum(), 2),
            "total_orders": len(df),
            "avg_order_value": round(df["revenue"].mean(), 2),
            "total_profit": round(df["profit"].sum(), 2),
            "top_product": df.groupby("product")["revenue"].sum().idxmax(),
            "top_region": df.groupby("region")["revenue"].sum().idxmax(),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
