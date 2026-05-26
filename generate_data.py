"""
generate_data.py
Generates synthetic transactional data for customer segmentation.
Saves to data/transactions.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_CUSTOMERS = 2000
START_DATE = pd.Timestamp("2022-01-01")
END_DATE   = pd.Timestamp("2024-12-31")

# Define 5 latent customer personas — mirrors the 5 segments we'll discover
PERSONAS = [
    # (weight, recency_days_range, freq_range, aov_range, categories_range, label)
    (0.15, (1,  15),  (20, 50), (150, 500), (5, 10), "Champions"),
    (0.20, (1,  45),  (10, 25), (80,  200), (3,  8), "Loyal Regulars"),
    (0.25, (45, 120), (3,  10), (50,  150), (2,  5), "At-Risk"),
    (0.25, (30, 90),  (1,   5), (40,  120), (1,  4), "Occasional Buyers"),
    (0.15, (60, 180), (1,   2), (20,   60), (1,  2), "One-Time Buyers"),
]


def generate_segmentation_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    customer_id = 1

    for weight, recency_range, freq_range, aov_range, cat_range, label in PERSONAS:
        n = int(N_CUSTOMERS * weight)
        for _ in range(n):
            cid = f"C{str(customer_id).zfill(5)}"
            recency     = int(rng.integers(*recency_range))
            frequency   = int(rng.integers(*freq_range))
            aov         = round(float(rng.uniform(*aov_range)), 2)
            cat_breadth = int(rng.integers(*cat_range))
            total_spend = round(aov * frequency * rng.uniform(0.8, 1.2), 2)

            records.append({
                "customer_id":       cid,
                "recency_days":      recency,
                "frequency":         frequency,
                "avg_order_value":   aov,
                "total_spend":       total_spend,
                "category_breadth":  cat_breadth,
                "true_segment":      label,   # for validation only — not used in clustering
            })
            customer_id += 1

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate_segmentation_data()
    # Drop true_segment before saving — clustering is unsupervised
    df.drop(columns=["true_segment"]).to_csv("data/transactions.csv", index=False)
    print(f"Saved {len(df):,} customer records to data/transactions.csv")
    print(df.describe().round(1))
