# Credit Card Fraud Detection - Databricks ML Pipeline

An end-to-end fraud detection pipeline built in Databricks using PySpark and Delta Live Tables, covering data ingestion, cleaning, and binary classification on the classic Kaggle credit card fraud dataset.

## Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 anonymized European card transactions from September 2013, with 492 confirmed fraud cases (~0.17% of transactions). Features V1–V28 are PCA-transformed for confidentiality, alongside `Time`, `Amount`, and the `Class` label (0 = legitimate, 1 = fraud).

![Class Distribution](outputs/class_distribution.png)

## Pipeline

**01_fraud_ingest** — Reads the raw CSV into a Spark DataFrame and writes it to a `raw_transactions` Delta table.

**02_fraud_transform** — Runs data quality checks (invalid amounts, nulls, duplicates) and cleans the dataset:
- Removed 1,825 rows with invalid (≤0) amounts
- Removed 1,081 duplicate rows
- 0 rows with missing critical fields
- Result: 284,807 → 281,918 clean rows, written to `clean_transactions`

**03_fraud_model** — Assembles features with `VectorAssembler`, splits into an 80/20 train/test set (225,414 / 56,504 rows), and trains a **Logistic Regression** baseline.

**04_fraud_model_comparison** — Trains a **Random Forest** classifier (100 trees) on the same split, compares it against the baseline, and applies threshold tuning to optimize fraud detection.

A **Delta Live Tables pipeline** (`fraud_pipeline`) also orchestrates the ingestion and cleaning stages as materialized views (`raw_transactions_dlt` → `clean_transactions_dlt`), following the Medallion Architecture.

## Why Precision & Recall Over Accuracy

With only 0.17% of transactions being fraudulent, a model that predicts "not fraud" for everything would score 99.83% accuracy while catching zero fraud. Precision and recall on the fraud class are the metrics that actually matter here — precision tells us how many fraud alerts are real, recall tells us how much fraud we're catching.

## Model Comparison

| Model | Accuracy | Fraud Precision | Fraud Recall |
|---|---|---|---|
| Logistic Regression | 99.93% | 90.9% | 60.2% |
| Random Forest | 99.94% | 88.1% | 71.1% |
| **Random Forest (threshold=0.3)** | **99.95%** | 85.3% | **77.1%** |

![Confusion Matrix](outputs/confusion_matrix.png)

## Threshold Tuning

By default, classifiers use a 0.5 probability threshold to decide fraud vs. legitimate. Lowering Random Forest's threshold to 0.3 improved fraud recall from 71.1% to 77.1% — meaning the model now catches over 3 in 4 fraud cases, up from roughly 7 in 10. The tradeoff is a modest drop in precision (88.1% → 85.3%), which is an acceptable cost in a fraud detection context, where missing a fraudulent transaction is typically far more expensive than an occasional false alarm requiring manual review.

**Selected model: Random Forest with threshold 0.3.**

## Tech stack
`python` `pyspark` `databricks` `delta-live-tables` `machine-learning` `medallion-architecture` `logistic-regression` `random-forest` `threshold-tuning`

## Next steps
- Experiment with class balancing techniques (SMOTE, class weighting) to push recall further
- Try XGBoost for comparison
- Add ROC curve and precision-recall curve visualization across thresholds
