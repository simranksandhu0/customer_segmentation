"""
clustering.py
K-Means customer segmentation using RFM + derived features.
Selects optimal k via Elbow Method + Silhouette Score.
Exports segment profiles and targeting recommendations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


FEATURE_COLS = [
    "recency_days",
    "frequency",
    "avg_order_value",
    "total_spend",
    "category_breadth",
]

SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Regulars",
    2: "At-Risk",
    3: "Occasional Buyers",
    4: "One-Time Buyers",
}

TARGETING_RECOMMENDATIONS = {
    "Champions":       "VIP early access, referral programme, premium loyalty tier",
    "Loyal Regulars":  "Bundle offers, subscription upgrade nudge, personalised restock reminders",
    "At-Risk":         "Win-back email sequence, time-limited discount, personal check-in",
    "Occasional Buyers": "Seasonal campaign, category-specific promotions, browse abandonment triggers",
    "One-Time Buyers": "Post-purchase nurture sequence, second-purchase incentive (10–15% off)",
}


def load_and_validate(path: str = "data/transactions.csv") -> pd.DataFrame:
    """Load and validate customer data."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run generate_data.py first.")

    df = pd.read_csv(path)

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if df[FEATURE_COLS].isnull().any().any():
        raise ValueError("Null values found in feature columns.")

    if (df[FEATURE_COLS] < 0).any().any():
        raise ValueError("Negative values found — all RFM features must be non-negative.")

    print(f"Loaded {len(df):,} customer records.")
    return df


def scale_features(df: pd.DataFrame) -> tuple:
    """
    StandardScaler normalisation — mandatory for K-Means
    (distance-based; unscaled features bias clusters toward high-magnitude dimensions).

    Returns scaled array and fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    return X_scaled, scaler


def select_k(X_scaled: np.ndarray, k_range: range = range(2, 11), seed: int = 42) -> int:
    """
    Evaluate k values using Silhouette Score (primary) and Inertia (elbow, secondary).
    Returns the k with the highest silhouette score.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"k={k}: Inertia={km.inertia_:,.0f} | Silhouette={sil:.4f}")

    results_df = pd.DataFrame(results)
    best_k = int(results_df.loc[results_df["silhouette"].idxmax(), "k"])
    print(f"\nOptimal k = {best_k} (highest Silhouette Score: "
          f"{results_df.loc[results_df['silhouette'].idxmax(), 'silhouette']:.4f})")
    return best_k


def fit_and_profile(
    df: pd.DataFrame, X_scaled: np.ndarray, k: int, seed: int = 42
) -> pd.DataFrame:
    """
    Fit final K-Means model with optimal k.
    Assigns business-readable segment labels.
    Exports segment profiles and targeting recommendations.
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    # Assign readable labels by matching cluster centroid profiles
    # Sort clusters by mean total_spend descending to assign labels consistently
    cluster_spend = (
        df.groupby("cluster")["total_spend"].mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    label_map = {
        row["cluster"]: SEGMENT_LABELS.get(i, f"Segment {i}")
        for i, row in cluster_spend.iterrows()
    }
    df["segment"] = df["cluster"].map(label_map)

    # Profile each segment
    profile = (
        df.groupby("segment")[FEATURE_COLS]
        .mean()
        .round(1)
        .reset_index()
    )
    profile["size"] = df.groupby("segment").size().values
    profile["targeting_recommendation"] = profile["segment"].map(TARGETING_RECOMMENDATIONS)

    print("\nSegment Profiles:")
    print(profile.to_string(index=False))

    Path("outputs").mkdir(exist_ok=True)
    profile.to_csv("outputs/segment_profiles.csv", index=False)
    df[["customer_id", "segment"] + FEATURE_COLS].to_csv(
        "outputs/customer_segments.csv", index=False
    )
    print("\nSaved to outputs/segment_profiles.csv and outputs/customer_segments.csv")

    return df


if __name__ == "__main__":
    df = load_and_validate()
    X_scaled, scaler = scale_features(df)
    best_k = select_k(X_scaled)
    df = fit_and_profile(df, X_scaled, k=best_k)
