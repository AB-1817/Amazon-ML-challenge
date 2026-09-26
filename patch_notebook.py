import json

NB_PATH = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

new_cells = []

# ── Section 19: Label baseline + save baseline artifacts ──────────────────
new_cells.append(md(
    "### Section 19: Label Baseline & Save Baseline Artifacts\n\n"
    "**BASELINE_8PASS_XGBOOST** — 8-pass blocking, K=25, XGBoost.\n"
    "Known result: Macro F0.5 = 0.8841, Precision = 0.9816, Recall = 0.7777, threshold = 0.80.\n"
    "Baseline artifacts are written to `/content/artifacts/baseline/` and must not be overwritten.\n"
))

new_cells.append(code(
    "import joblib, os, json as _json\n"
    "\n"
    "BASELINE_TAG = 'BASELINE_8PASS_XGBOOST'\n"
    "baseline_dir = '/content/artifacts/baseline'\n"
    "os.makedirs(baseline_dir, exist_ok=True)\n"
    "\n"
    "# Copy existing XGBoost model as baseline\n"
    "xgb_model.save_model(os.path.join(baseline_dir, 'xgboost_baseline.json'))\n"
    "joblib.dump(lr_model, os.path.join(baseline_dir, 'lr_baseline.joblib'))\n"
    "joblib.dump(scaler,   os.path.join(baseline_dir, 'lr_scaler_baseline.joblib'))\n"
    "\n"
    "baseline_meta = {\n"
    "    'tag': BASELINE_TAG,\n"
    "    'blocking': '8-pass',\n"
    "    'K': 25,\n"
    "    'model': 'XGBoost',\n"
    "    'macro_f05': xgb_best['macro_f05'],\n"
    "    'precision': xgb_best['precision'],\n"
    "    'recall':    xgb_best['recall'],\n"
    "    'threshold': xgb_best['threshold'],\n"
    "    'n_val_s1':  xgb_best['n_val_s1_evaluated'],\n"
    "    'known_result': {'macro_f05': 0.8841, 'precision': 0.9816, 'recall': 0.7777, 'threshold': 0.80},\n"
    "}\n"
    "with open(os.path.join(baseline_dir, 'baseline_meta.json'), 'w') as f:\n"
    "    _json.dump(baseline_meta, f, indent=2)\n"
    "\n"
    "# Also copy existing results CSVs\n"
    "import shutil\n"
    "for fn in ['model_comparison.csv', 'threshold_results.csv', 'validation_predictions.csv']:\n"
    "    src = os.path.join('results', fn)\n"
    "    if os.path.exists(src):\n"
    "        shutil.copy(src, os.path.join(baseline_dir, fn))\n"
    "\n"
    "print(f'Baseline [{BASELINE_TAG}] artifacts saved to {baseline_dir}/')\n"
    "print(f'  XGBoost Macro F0.5 (measured): {xgb_best[\"macro_f05\"]}')\n"
    "print(f'  Known result:                   0.8841')\n"
    "print('Baseline is LOCKED. Subsequent sections write to /content/experiment_v2/ only.')\n"
))

# ── Section 20: Install CatBoost ───────────────────────────────────────────
new_cells.append(md("### Section 20: Install CatBoost\n"))

new_cells.append(code(
    "try:\n"
    "    import catboost as _cb\n"
    "    print(f'CatBoost already installed: {_cb.__version__}')\n"
    "except ImportError:\n"
    "    print('Installing CatBoost...')\n"
    "    import subprocess, sys\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'catboost>=1.2'])\n"
    "    import catboost as _cb\n"
    "    print(f'CatBoost installed: {_cb.__version__}')\n"
    "\n"
    "from catboost import CatBoostClassifier\n"
    "print('CatBoost ready.')\n"
))

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].extend(new_cells)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f'Appended {len(new_cells)} cells (Sections 19-20). Total cells: {len(nb["cells"])}')
