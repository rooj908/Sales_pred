"""
Train ML models: Sales Forecasting + Customer Churn Prediction
"""
import pandas as pd
import numpy as np
import pickle, os, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, classification_report, roc_auc_score
)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessing import load_and_clean_sales, load_and_clean_customers, create_feature_matrix

os.makedirs("models/saved", exist_ok=True)

def train_sales_model():
    print("Training Sales Revenue Prediction Model...")
    df = load_and_clean_sales()
    X, y, scaler = create_feature_matrix(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "r2": round(r2_score(y_test, y_pred), 4),
    }
    print(f"  Sales Model — R²: {metrics['r2']}, MAE: {metrics['mae']}, RMSE: {metrics['rmse']}")
    
    with open("models/saved/sales_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("models/saved/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/saved/sales_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Feature importance
    fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv("models/saved/feature_importance.csv", index=False)
    return metrics

def train_churn_model():
    print("Training Customer Churn Model...")
    df = load_and_clean_customers()
    
    features = ["age", "total_orders", "total_spend", "loyalty_years",
                "satisfaction_score", "income_bracket_encoded",
                "region_encoded", "segment_encoded"]
    
    df["churn_label"] = (df["churn_risk"] > 0.6).astype(int)
    X = df[features]
    y = df["churn_label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "report": classification_report(y_test, y_pred),
    }
    print(f"  Churn Model — Accuracy: {metrics['accuracy']}, ROC-AUC: {metrics['roc_auc']}")
    
    with open("models/saved/churn_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("models/saved/churn_metrics.json", "w") as f:
        json.dump({"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"]}, f, indent=2)
    
    return metrics

def train_forecasting_model():
    print("Training Time-Series Forecasting (Linear Trend)...")
    df = load_and_clean_sales()
    monthly = df.groupby(["year", "month"]).agg(
        total_revenue=("revenue", "sum")
    ).reset_index()
    monthly["time_index"] = range(len(monthly))
    
    X = monthly[["time_index"]]
    y = monthly["total_revenue"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "r2": round(r2_score(y_test, y_pred), 4),
    }
    print(f"  Forecast Model — R²: {metrics['r2']}, MAE: {metrics['mae']}")
    
    with open("models/saved/forecast_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    
    return metrics

if __name__ == "__main__":
    s = train_sales_model()
    c = train_churn_model()
    f = train_forecasting_model()
    print("\nAll models trained successfully!")
