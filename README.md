# Credit Card Fraud Detection - Databricks ML Pipeline

An end-to-end fraud detection pipeline built in Databricks using PySpark and Delta Live Tables, covering data ingestion, cleaning, and binary classification on the classic Kaggle credit card fraud dataset.

## Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 anonymized European card transactions from September 2013, with 492 confirmed fraud cases (~0.17% of transactions). Features V1–V28 are PCA-transformed for confidentiality, alongside `Time`, `Amount`, and the `Class` label (0 = legitimate, 1 = fraud).

## Pipeline

**01_fraud_ingest** — Reads the raw CSV into a Spark DataFrame and writes it to a `raw_transactions` Delta table.

**02_fraud_transform** — Runs data quality checks (invalid amounts, nulls, duplicates) and cleans the dataset:
- Removed 1,825 rows with invalid (≤0) amounts
- Removed 1,081 duplicate rows
- 0 rows with missing critical fields
- Result: 284,807 → 281,918 clean rows, written to `clean_transactions`

**03_fraud_model** — Assembles features with `VectorAssembler`, splits into an 80/20 train/test set (225,414 / 56,504 rows), and trains a **Logistic Regression** classifier.

A **Delta Live Tables pipeline** (`fraud_pipeline`) also orchestrates the ingestion and cleaning stages as materialized views (`raw_transactions_dlt` → `clean_transactions_dlt`), following the Medallion Architecture.

## Results

| Metric | Overall | Fraud class only |
|---|---|---|
| Accuracy | 99.93% | — |
| Precision | 99.94% | 90.9% |
| Recall | 99.99% | 60.2% |

The model achieves strong overall performance, with especially high precision on fraud predictions (90.9%) — meaning when it flags a transaction as fraud, it's right the vast majority of the time. This kind of precision is valuable in production, where false fraud alerts create friction for legitimate customers.

## Tech stack
`python` `pyspark` `databricks` `delta-live-tables` `machine-learning` `medallion-architecture` `logistic-regression`

## Next steps
- Experiment with class balancing techniques (SMOTE, class weighting) to further improve fraud detection coverage
- Compare against tree-based models (Random Forest, XGBoost)
- Add precision-recall curve visualization for deeper threshold analysis
