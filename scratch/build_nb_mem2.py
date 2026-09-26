import json, os

nb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb'))
with open(nb_path, 'r', encoding='utf-8') as f:
    NB = json.load(f)

assert NB['cells'][19]['cell_type'] == 'markdown'
assert 'Section 10' in NB['cells'][19]['source']

NB['cells'][19]['source'] = (
    "### Section 10: Feature Engineering (24 Features) + Chunked Extraction + Feature Validation\n\n"
    "**Memory Architecture**: Training and validation pairs are fetched from DuckDB in chunks\n"
    "of `FEATURE_CHUNK_SIZE` rows. Only one chunk is in pandas RAM at a time.\n"
    "The final `X_train` / `X_val` are pre-allocated float32 NumPy arrays.\n\n"
    "Before allocation, expected memory is printed:\n"
    "`rows x 24 features x 4 bytes`\n\n"
    "After extraction:\n"
    "- intermediate DataFrames are deleted and `gc.collect()` is called\n"
    "- only `X_train`, `y_train`, `X_val`, `y_val` remain in memory\n"
    "- `val_meta` (s1_id, target_id, label) is kept as a minimal array for threshold search\n\n"
    "Feature validation checks (on X_train):\n"
    "- No NaN / inf values\n"
    "- Feature min / max / mean\n"
    "- Constant features (zero variance)\n"
    "- Highly correlated features (|r| > 0.98)\n"
    "- Feature name alignment\n"
)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=2, ensure_ascii=False)
print('Fix 2 applied: Section 10 markdown updated.')
