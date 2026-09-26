# Training Report — Amazon ML Challenge 2026

## 1. Dataset & Split Configuration

- **Total S1 Sample Size**: 15,000
- **Training S1 Count (80%)**: 11,998
- **Validation S1 Count (20%)**: 3,002
- **Random Seed**: 42

## 2. Candidate Generation & Sampling

- **Blocking Passes**: 8 passes (passes 1-8)
- **Raw Candidate Pairs**: 3,322,395
- **Top-K Candidates Retained**: 293,598 (K=25)
- **Candidate Stats**: Avg=19.7, Med=25, p95=25, Max=25
- **Train Positive Pairs**: 35,092
- **Train Negative Pairs (Raw)**: 200,055
- **Train Hard Negatives (Sampled)**: 105,276 (3.00:1 ratio)

## 3. Model Comparison

| Model | Optimal Threshold | Validation Macro F0.5 | Precision | Recall | F1 | ROC-AUC | PR-AUC | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.60 | **0.8911** | 0.9751 | 0.7948 | 0.9729 | 0.9993 | 0.9962 | 1.01s |
| XGBoost | 0.90 | **0.8711** | 0.9550 | 0.7758 | 0.9513 | 0.9977 | 0.9906 | 4.38s |

## 4. Final Decision

- **Best Model**: **LogisticRegression**
- **Optimal Decision Threshold (tau*)**: **0.60**
- **Validation Macro F0.5**: **0.8911**
- **Validation Precision**: 0.9751
- **Validation Recall**: 0.7948
- **Total Runtime**: 86.1s (1.44 min)
- **Peak Memory**: 309.1 MB
