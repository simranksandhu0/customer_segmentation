# Customer Segmentation Analysis

> K-Means clustering on transactional data to identify 5 distinct customer groups — with targeting recommendations for each segment.

---

## Overview

Not all customers are the same. This project segments a customer base into 5 meaningful groups based on purchasing behaviour and lifetime value, then produces concrete targeting recommendations for each segment — benchmarked against standard industry segmentation approaches (RFM analysis).

The goal: give the marketing and merchandising teams a data-backed answer to "who are our customers, and how should we speak to each of them?"

---

## Problem Statement

The business was treating all customers the same — same promotions, same messaging, same cadence. Without segmentation, high-value customers receive the same communication as one-time buyers, and campaigns can't be optimised for relevance or conversion.

---

## Approach

### 1. Feature Engineering — RFM Framework
Built features grounded in the industry-standard RFM model:

| Feature | Description |
|---|---|
| **Recency** | Days since last purchase |
| **Frequency** | Number of transactions in the period |
| **Monetary** | Total spend in the period |
| **AOV** | Average order value |
| **Category Breadth** | Number of distinct categories purchased |

### 2. Data Preparation
- Normalised all features using StandardScaler (K-Means is distance-based — scale matters)
- Removed outliers that would distort cluster centroids

### 3. Optimal Cluster Count
- Tested k = 2 through 10
- Used the **Elbow Method** (inertia) and **Silhouette Score** to select k = 5
- Validated cluster stability with multiple random seeds

### 4. Cluster Profiling
Each cluster was profiled on all RFM dimensions and given a business-readable label:

| Segment | Label | Characteristics |
|---|---|---|
| 1 | Champions | High frequency, high spend, recent — most valuable customers |
| 2 | Loyal Regulars | Consistent frequency, mid-to-high spend |
| 3 | At-Risk | Previously active, declining recency |
| 4 | Occasional Buyers | Low frequency, average spend, sporadic |
| 5 | One-Time Buyers | Single purchase, low spend |

### 5. Targeting Recommendations
Each segment received specific recommendations for campaign type, messaging tone, and offer structure — designed to improve conversion and reduce churn within each group.

---

## Results

- **5 distinct customer segments** identified with clear behavioural profiles
- Segment sizes and revenue contribution quantified
- Targeting recommendations produced for each group
- Benchmarked against standard RFM segmentation to validate segment validity

---

## Stack

- **Python** — full analysis pipeline
- **Pandas / NumPy** — data wrangling and feature engineering
- **Scikit-learn** — K-Means clustering, StandardScaler, Silhouette analysis
- **Matplotlib / Seaborn** — cluster visualisation (PCA plots, heatmaps)

---

## File Structure

```
customer_segmentation/
├── generate_data.py    # Generates synthetic customer transaction data
├── clustering.py       # RFM feature engineering, K-Means clustering, segment profiling
└── README.md
```

---

## How to Run

```
# Clone the repo
git clone https://github.com/simranksandhu0/customer_segmentation.git
cd customer_segmentation

# Generate synthetic data
python generate_data.py

# Run clustering analysis
python clustering.py
```

---

## Key Takeaways

- Feature scaling is non-negotiable for K-Means — unscaled RFM features produced clusters dominated by spend alone
- Elbow method alone was insufficient; Silhouette Scores were the deciding factor for k = 5
- Business-readable segment labels were as important as the technical output — a cluster profile no one understands doesn't get acted on
