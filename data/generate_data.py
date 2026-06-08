"""
Generate synthetic business sales data for the BI platform.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_sales_data(n_records=5000):
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 1095)) for _ in range(n_records)]
    
    products = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse", "Headphones", "Webcam"]
    regions = ["North", "South", "East", "West", "Central"]
    segments = ["Enterprise", "SMB", "Retail", "Online"]
    
    data = {
        "order_id": [f"ORD-{i:05d}" for i in range(n_records)],
        "date": sorted(dates),
        "product": np.random.choice(products, n_records),
        "region": np.random.choice(regions, n_records),
        "segment": np.random.choice(segments, n_records),
        "quantity": np.random.randint(1, 20, n_records),
        "unit_price": np.round(np.random.uniform(50, 2000, n_records), 2),
        "discount": np.round(np.random.uniform(0, 0.3, n_records), 2),
        "customer_age": np.random.randint(18, 70, n_records),
        "customer_loyalty_years": np.round(np.random.uniform(0, 10, n_records), 1),
        "marketing_spend": np.round(np.random.uniform(100, 5000, n_records), 2),
        "competitor_price_index": np.round(np.random.uniform(0.8, 1.2, n_records), 2),
    }
    df = pd.DataFrame(data)
    df["revenue"] = np.round(df["quantity"] * df["unit_price"] * (1 - df["discount"]), 2)
    df["profit_margin"] = np.round(np.random.uniform(0.1, 0.5, n_records), 3)
    df["profit"] = np.round(df["revenue"] * df["profit_margin"], 2)
    df["month"] = df["date"].apply(lambda x: x.month)
    df["year"] = df["date"].apply(lambda x: x.year)
    df["quarter"] = df["date"].apply(lambda x: (x.month - 1) // 3 + 1)
    return df

def generate_customer_data(n_customers=1000):
    data = {
        "customer_id": [f"CUST-{i:04d}" for i in range(n_customers)],
        "age": np.random.randint(18, 70, n_customers),
        "income_bracket": np.random.choice(["Low", "Medium", "High", "Very High"], n_customers),
        "region": np.random.choice(["North", "South", "East", "West", "Central"], n_customers),
        "segment": np.random.choice(["Enterprise", "SMB", "Retail", "Online"], n_customers),
        "total_orders": np.random.randint(1, 50, n_customers),
        "total_spend": np.round(np.random.uniform(500, 50000, n_customers), 2),
        "loyalty_years": np.round(np.random.uniform(0, 10, n_customers), 1),
        "churn_risk": np.round(np.random.uniform(0, 1, n_customers), 3),
        "satisfaction_score": np.round(np.random.uniform(1, 5, n_customers), 1),
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    
    print("Generating sales data...")
    sales_df = generate_sales_data(5000)
    sales_df.to_csv("data/raw/sales_data.csv", index=False)
    print(f"  Saved {len(sales_df)} sales records")

    print("Generating customer data...")
    customer_df = generate_customer_data(1000)
    customer_df.to_csv("data/raw/customer_data.csv", index=False)
    print(f"  Saved {len(customer_df)} customer records")

    print("Data generation complete!")
