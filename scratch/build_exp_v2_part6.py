"""Append Section 23: XGBoost v2 + CatBoost training."""
import json

NB = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

nb['cells'].append(md(
"### Section 23: Model Training v2 — XGBoost + CatBoost\n\n"
"Train on the improved feature set (30 features) from the best blocking table.\n"
"Same train/val split as baseline. Conservative memory settings throughout.\n"
))

src_23 = (
"import xgboost as xgb\n"
"import numpy as _np2\n"
"\n"
"# ── XGBoost v2 ────────────────────────────────────────────────────────────\n"
"print('Training XGBoost v2 (30 features, best blocking)...')\n"
"t_xgb2 = time.time()\n"
"xgb_model_v2 = xgb.XGBClassifier(\n"
"    n_estimators=400, max_depth=7, learning_rate=0.07,\n"
"    subsample=0.8, colsample_bytree=0.8,\n"
"    tree_method='hist', random_state=SEED, n_jobs=4\n"
")\n"
"xgb_model_v2.fit(X_train_v2, y_train_v2)\n"
"xgb_time_v2 = time.time() - t_xgb2\n"
"xgb_probs_v2 = xgb_model_v2.predict_proba(X_val_v2)[:,1]\n"
"print(f'  XGBoost v2 trained in {xgb_time_v2:.2f}s')\n"
"\n"
"imp_v2 = xgb_model_v2.feature_importances_\n"
"top_idx_v2 = _np2.argsort(imp_v2)[::-1][:10]\n"
"print('Top 10 XGBoost v2 Feature Importances:')\n"
"for r, idx in enumerate(top_idx_v2, 1):\n"
"    print(f'  {r:>2}. {FEATURE_NAMES_V2[idx]:<30}: {imp_v2[idx]*100:5.2f}%')\n"
"\n"
"# ── CatBoost ──────────────────────────────────────────────────────────────\n"
"print('\\nTraining CatBoost...')\n"
"t_cb = time.time()\n"
"cb_model = CatBoostClassifier(\n"
"    iterations=400, depth=7, learning_rate=0.07,\n"
"    loss_function='Logloss', eval_metric='AUC',\n"
"    verbose=False, random_seed=SEED, thread_count=4,\n"
"    task_type='CPU'\n"
")\n"
"cb_model.fit(X_train_v2, y_train_v2)\n"
"cb_time = time.time() - t_cb\n"
"cb_probs = cb_model.predict_proba(X_val_v2)[:,1]\n"
"print(f'  CatBoost trained in {cb_time:.2f}s')\n"
"\n"
"cb_imp = cb_model.get_feature_importance()\n"
"top_cb = _np2.argsort(cb_imp)[::-1][:10]\n"
"print('Top 10 CatBoost Feature Importances:')\n"
"for r, idx in enumerate(top_cb, 1):\n"
"    print(f'  {r:>2}. {FEATURE_NAMES_V2[idx]:<30}: {cb_imp[idx]:5.2f}')\n"
)

nb['cells'].append(code(src_23))

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f'Cells: {len(nb["cells"])}  Part6 written.')
