"""
Data Cleaning & Preprocessing Module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_and_clean_sales(filepath="data/raw/sales_data.csv"):
    df = pd.read_csv(filepath, parse_dates=["date"])
    
    # Drop duplicates
    df.drop_duplicates(subset=["order_id"], inplace=True)
    
    # Fill missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Remove obvious outliers (revenue > 0)
    df = df[df["revenue"] > 0]
    
    # Encode categoricals
    le = LabelEncoder()
    for col in ["product", "region", "segment"]:
        df[f"{col}_encoded"] = le.fit_transform(df[col])
    
    return df

def load_and_clean_customers(filepath="data/raw/customer_data.csv"):
    df = pd.read_csv(filepath)
    df.drop_duplicates(subset=["customer_id"], inplace=True)
    
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    le = LabelEncoder()
    for col in ["income_bracket", "region", "segment"]:
        df[f"{col}_encoded"] = le.fit_transform(df[col])
    
    return df

def create_feature_matrix(sales_df):
    features = [
        "quantity", "unit_price", "discount", "customer_age",
        "customer_loyalty_years", "marketing_spend", "competitor_price_index",
        "month", "quarter", "product_encoded", "region_encoded", "segment_encoded"
    ]
    X = sales_df[features].copy()
    y = sales_df["revenue"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=features), y, scaler

def get_monthly_aggregates(sales_df):
    monthly = sales_df.groupby(["year", "month"]).agg(
        total_revenue=("revenue", "sum"),
        total_profit=("profit", "sum"),
        total_orders=("order_id", "count"),
        avg_order_value=("revenue", "mean"),
    ).reset_index()
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    return monthly.sort_values("date")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    sales = load_and_clean_sales()
    customers = load_and_clean_customers()
    sales.to_csv("data/processed/sales_clean.csv", index=False)
    customers.to_csv("data/processed/customers_clean.csv", index=False)
    print("Preprocessing complete.")
    print(f"Sales: {sales.shape}, Customers: {customers.shape}")
