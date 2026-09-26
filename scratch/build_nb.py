import json, os

NB = {"cells": [], "metadata": {"colab": {"name": "train_matching_model_colab.ipynb", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

# ── TITLE ──────────────────────────────────────────────────────────────────────
NB["cells"].append(md(
    "# Amazon ML Challenge 2026: Business Entity Resolution\n"
    "## Candidate-Pair Matching Model Training & Validation (Google Colab Ready)\n\n"
    "**NOTE: DEVELOPMENT/TRAINING EXPERIMENT** — 150k S1 entities + random distractors. Not final full-scale training.\n\n"
    "### Pipeline Architecture:\n"
    "1. Environment Setup\n2. S1 Sampling & Train/Val Split (80/20, zero leakage)\n"
    "3. 8-Pass Blocking (K=25)\n4. Deliberate Hard-Negative Sampling (6 categories)\n"
    "5. Feature Engineering (24 features)\n6. Model Training: LR vs XGBoost\n"
    "7. Threshold Search & Macro F0.5\n8. Error Analysis\n9. Artifact Export\n"
))

# ── SEC 1: ENV ─────────────────────────────────────────────────────────────────
NB["cells"].append(md("### Section 1: Environment Setup & Reproducibility\n"))
NB["cells"].append(code(
    "import os, sys, time, json, random, platform\n"
    "import numpy as np\nimport pandas as pd\n\n"
    "SEED = 42\nrandom.seed(SEED)\nnp.random.seed(SEED)\n\n"
    "print('=' * 60)\n"
    "print(f'Python:  {platform.python_version()}')\n"
    "print(f'OS:      {platform.system()} {platform.release()}')\n"
    "print(f'CPUs:    {os.cpu_count()}')\n"
    "print(f'NumPy:   {np.__version__}')\n"
    "print(f'Pandas:  {pd.__version__}')\n"
    "print(f'SEED:    {SEED}')\n"
    "print('=' * 60)\n"
))

# ── SEC 2: DATA ROOT ───────────────────────────────────────────────────────────
NB["cells"].append(md("### Section 2: Dataset Location / Google Drive Support\n"))
NB["cells"].append(code(
    "try:\n    import google.colab\n    IN_COLAB = True\nexcept ImportError:\n    IN_COLAB = False\n\n"
    "if IN_COLAB:\n"
    "    from google.colab import drive\n    drive.mount('/content/drive')\n"
    "    DATA_ROOT = '/content/drive/MyDrive/amazon_ml_challenge'\n"
    "    if not os.path.exists(DATA_ROOT):\n        DATA_ROOT = '/content/dataset/train'\n"
    "else:\n"
    "    DATA_ROOT = os.path.abspath('../6ab10eb3b23ba_student_resource/student_resource/dataset/train')\n\n"
    "print(f'DATA_ROOT: {DATA_ROOT}')\n"
    "REQUIRED_FILES = ['train_source1.tsv','train_source2.tsv','train_source3.tsv','train_ground_truth.tsv']\n"
    "all_found = True\n"
    "for fname in REQUIRED_FILES:\n"
    "    fpath = os.path.join(DATA_ROOT, fname)\n"
    "    if os.path.exists(fpath):\n"
    "        print(f'  [OK] {fname:<25} ({os.path.getsize(fpath)/1024/1024:6.1f} MB)')\n"
    "    else:\n"
    "        print(f'  [MISSING] {fname}')\n        all_found = False\n\n"
    "if not all_found:\n    raise FileNotFoundError('One or more required files missing. Check DATA_ROOT.')\n"
    "print('All required training files verified.')\n"
))

# ── SEC 3: DEPS ────────────────────────────────────────────────────────────────
NB["cells"].append(md("### Section 3: Install Dependencies\n"))
NB["cells"].append(code(
    "!pip install -q duckdb>=1.1.0 rapidfuzz>=3.0.0 xgboost>=2.0.0 scikit-learn>=1.3.0 joblib>=1.3.0\n\n"
    "import duckdb, rapidfuzz, xgboost as xgb, sklearn\n"
    "print(f'DuckDB:{duckdb.__version__}  RapidFuzz:{rapidfuzz.__version__}  XGBoost:{xgb.__version__}  sklearn:{sklearn.__version__}')\n"
))

# ── SEC 4: LOAD DATA ───────────────────────────────────────────────────────────
NB["cells"].append(md("### Section 4: Load and Validate Data\n"))
NB["cells"].append(code(
    "s1_path = os.path.join(DATA_ROOT,'train_source1.tsv').replace('\\\\','/')\n"
    "s2_path = os.path.join(DATA_ROOT,'train_source2.tsv').replace('\\\\','/')\n"
    "s3_path = os.path.join(DATA_ROOT,'train_source3.tsv').replace('\\\\','/')\n"
    "gt_path = os.path.join(DATA_ROOT,'train_ground_truth.tsv').replace('\\\\','/')\n\n"
    "con = duckdb.connect(':memory:')\n"
    "con.execute(\"SET memory_limit='6GB';\")\n"
    "con.execute('SET threads=8;')\n\n"
    "for name, p in [('Source 1',s1_path),('Source 2',s2_path),('Source 3',s3_path),('Ground Truth',gt_path)]:\n"
    "    df = con.execute(f\"SELECT * FROM read_csv_auto('{p}', delim='\\t', header=true, quote='') LIMIT 3\").df()\n"
    "    cnt = con.execute(f\"SELECT count(*) FROM read_csv_auto('{p}', delim='\\t', header=true, quote='')\").fetchone()[0]\n"
    "    print(f'\\n--- {name} ({cnt:,} rows) ---')\n"
    "    print(f'Columns: {list(df.columns)}')\n"
    "    display(df)\n"
))

# ── SEC 5: MACROS ──────────────────────────────────────────────────────────────
NB["cells"].append(md("### Section 5: Preprocessing & SQL Normalization Macros\n"))
NB["cells"].append(code(
    "con.execute(\"\"\"\n"
    "CREATE OR REPLACE MACRO norm_text(s) AS\n"
    "    lower(trim(regexp_replace(regexp_replace(\n"
    "        regexp_replace(coalesce(s,''), 'https?://(?:www\\.)?|www\\.', '', 'g'),\n"
    "        '\\.(?:com|org|net|in|co|gov|edu|fr|io)\\\\b', ' ', 'g'),\n"
    "        '[^\\\\w\\\\s]', ' ', 'g')));\n\n"
    "CREATE OR REPLACE MACRO clean_legal(s) AS\n"
    "    trim(regexp_replace(norm_text(s),\n"
    "    '\\\\b(?:private\\\\s+limited|pvt\\\\s+ltd|pvt\\\\s+limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|holdings|holding|services|service|company|limited|llc|inc|corp|ltd|llp|co|sarl|sas|sa|sci|eurl|sasu)\\\\b',\n"
    "    ' ', 'g'));\n\n"
    "CREATE OR REPLACE MACRO compact_name(s) AS\n"
    "    regexp_replace(norm_text(s), '\\\\s+', '', 'g');\n\n"
    "CREATE OR REPLACE MACRO get_token(s, n) AS\n"
    "    CASE WHEN array_length(string_split(trim(coalesce(s,'')), ' ')) >= n\n"
    "         THEN string_split(trim(coalesce(s,'')), ' ')[n]\n"
    "         ELSE '' END;\n\n"
    "CREATE OR REPLACE MACRO clean_addr(s) AS\n"
    "    lower(trim(regexp_replace(\n"
    "        regexp_replace(coalesce(s,''), '[^\\\\w\\\\s]', ' ', 'g'),\n"
    "        '\\\\s+', ' ', 'g')));\n"
    "\"\"\")\n"
    "print('DuckDB preprocessing macros registered.')\n"
))

out_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb')
with open(os.path.normpath(out_path), 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=2, ensure_ascii=False)
print(f'Written sections 1-5 to {os.path.normpath(out_path)}')
