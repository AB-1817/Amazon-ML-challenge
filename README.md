# Amazon ML Challenge 2026 — Business Entity Resolution

End-to-end pipeline for matching S1 business entities against S2/S3 target pools.

## Pipeline Overview

```
S1 Sample (150k) → 8-Pass Blocking → Hard-Negative Sampling → Feature Engineering → XGBoost / CatBoost → Threshold Search → Macro F0.5
```

## Baseline Result

| Model | Blocking | Macro F0.5 | Precision | Recall | Threshold |
|-------|----------|-----------|-----------|--------|-----------|
| XGBoost | 8-pass K=25 | 0.8841 | 0.9816 | 0.7777 | 0.80 |

## Experiment v2

Controlled experiment to improve candidate recall and matcher F0.5:

- **Blocking strategies A–F**: 8-pass K=25/50/100, permutation passes, char n-gram TF-IDF (block-wise sparse), address-driven passes
- **Models**: XGBoost v2, CatBoost, XGB/CatBoost ensemble (α = 0.25 / 0.50 / 0.75)
- **Features**: 30 features (original 24 + 6 candidate-context features)
- **Metric**: Macro F0.5 evaluated on 30,001 validation S1 entities

## Key Files

| File | Description |
|------|-------------|
| `notebooks/train_matching_model_colab.ipynb` | Main training notebook (Google Colab) |
| `scratch/` | Builder scripts used to construct the notebook |
| `results/` | Baseline CSV outputs |
| `artifacts/feature_config.json` | Feature name list |
| `artifacts/selected_model.json` | Best baseline model config |

## Architecture Decisions

- **Memory safety**: All blocking and feature extraction is chunked (50k rows). No full Cartesian joins. DuckDB disk-backed at `/content/duckdb_tmp`.
- **Anti-leakage**: S1 entities are split 80/20 by country+singleton. Three explicit leakage checks raise `AssertionError` on failure.
- **Char n-gram retrieval**: Block-wise sparse top-K (never materialises full similarity matrix). Feasibility gate skips countries with >300k targets.
- **Hard negatives**: 6-category deliberate sampling at ratio 3.0, 3-pass chunked (no full pool materialisation).

## Reproducibility

```
SEED = 42
TRAIN_S1_LIMIT = 150000
VALIDATION_FRACTION = 0.20
TOP_K = 25
HARD_NEG_RATIO = 3.0
```

## Environment

Google Colab (Python 3.10+). Dependencies installed in Section 3 of the notebook:
```
duckdb>=1.1.0  rapidfuzz>=3.0.0  xgboost>=2.0.0  scikit-learn>=1.3.0  catboost>=1.2
```

Dataset ZIP is uploaded directly to `/content/` — no Google Drive required.
