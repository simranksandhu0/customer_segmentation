"""
generate_data.py
Generates synthetic customer churn data and saves to data/churn_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 5000

def generate_churn_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"C{str(i).zfill(5)}" for i in range(1, n + 1)]

    # Tenure in months (1–72)
    tenure = rng.integers(1, 73, size=n)

    # Login frequency per month (1–30); lower → higher churn risk
    login_frequency = rng.integers(1, 31, size=n)

    # Support tickets raised in last 90 days (0–15); higher → higher churn risk
    support_tickets = rng.integers(0, 16, size=n)

    # Number of distinct features used (1–10)
    features_used = rng.integers(1, 11, size=n)

    # Average order value in CAD (10–500)
    avg_order_value = rng.uniform(10, 500, size=n).round(2)

    # Days since last login (0–120); higher → higher churn risk
    days_since_login = rng.integers(0, 121, size=n)

    # Monthly spend in CAD (0–1000)
    monthly_spend = rng.uniform(0, 1000, size=n).round(2)

    # Contract type
    contract_type = rng.choice(["Month-to-Month", "One Year", "Two Year"], size=n, p=[0.5, 0.3, 0.2])

    # Churn label — constructed to be realistic, not purely random
    churn_score = (
        (login_frequency < 8).astype(int) * 2
        + (support_tickets > 8).astype(int) * 2
        + (days_since_login > 60).astype(int) * 2
        + (features_used < 3).astype(int)
        + (contract_type == "Month-to-Month").astype(int)
        + rng.integers(0, 3, size=n)
    )
    churned = (churn_score >= 5).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure,
        "login_frequency_monthly": login_frequency,
        "support_tickets_90d": support_tickets,
        "features_used": features_used,
        "avg_order_value": avg_order_value,
        "days_since_last_login": days_since_login,
        "monthly_spend": monthly_spend,
        "contract_type": contract_type,
        "churned": churned,
    })

    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate_churn_data()
    df.to_csv("data/churn_data.csv", index=False)
    print(f"Saved {len(df):,} rows to data/churn_data.csv")
    print(f"Churn rate: {df['churned'].mean():.1%}")
