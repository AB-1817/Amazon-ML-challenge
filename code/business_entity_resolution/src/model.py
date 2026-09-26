import os
import joblib
import numpy as np
import xgboost as xgb
from typing import Tuple, Optional
from .config import MODELS_DIR, NUM_WORKERS
from .features import FEATURE_NAMES

MODEL_PATH = os.path.join(MODELS_DIR, "entity_matcher_xgb.json")


def train_matching_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> xgb.XGBClassifier:
    """Train XGBoost binary classifier with histogram-based binning on CPU threads."""
    print(f"Training XGBoost classifier on {len(X_train):,} samples with {X_train.shape[1]} features...")
    
    # Calculate scale_pos_weight if needed, or balance samples
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    print(f"Dataset balance: Positives={pos_count:,}, Negatives={neg_count:,}")
    
    clf = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        n_jobs=NUM_WORKERS,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        random_state=42
    )
    
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
        
    clf.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=50
    )
    
    # Save trained model
    os.makedirs(MODELS_DIR, exist_ok=True)
    clf.save_model(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature Importances:")
    for idx in sorted_idx:
        print(f"  {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")
        
    return clf


def load_matching_model() -> xgb.XGBClassifier:
    """Load pre-trained XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_PATH)
    return clf
