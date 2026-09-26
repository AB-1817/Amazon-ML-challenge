import json, os

nb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb'))
with open(nb_path, 'r', encoding='utf-8') as f:
    NB = json.load(f)

# Cell 3 = Section 2 markdown, Cell 4 = Section 2 code
assert NB['cells'][3]['cell_type'] == 'markdown', 'Cell 3 is not markdown'
assert NB['cells'][4]['cell_type'] == 'code',     'Cell 4 is not code'
assert 'drive' in NB['cells'][4]['source'],       'Cell 4 does not contain drive code'

# ── UPDATE MARKDOWN ────────────────────────────────────────────────────────────
NB['cells'][3]['source'] = (
    "### Section 2: Dataset Location — ZIP Upload (/content)\n\n"
    "The dataset ZIP is uploaded directly to `/content` in the Colab runtime.\n"
    "This cell automatically detects the ZIP, extracts it, locates the four required\n"
    "TSV files, and sets `DATA_ROOT`. No Google Drive is used.\n"
)

# ── REPLACE CODE CELL ──────────────────────────────────────────────────────────
NB['cells'][4]['source'] = (
    "import glob, zipfile\n"
    "\n"
    "EXTRACT_DIR   = '/content/amazon_ml_challenge'\n"
    "REQUIRED_FILES = [\n"
    "    'train_source1.tsv',\n"
    "    'train_source2.tsv',\n"
    "    'train_source3.tsv',\n"
    "    'train_ground_truth.tsv',\n"
    "]\n"
    "\n"
    "# ── 1. Locate the uploaded ZIP under /content ──────────────────────────────\n"
    "zip_candidates = glob.glob('/content/*.zip')\n"
    "if not zip_candidates:\n"
    "    raise FileNotFoundError(\n"
    "        'No ZIP file found in /content. '\n"
    "        'Please upload the dataset ZIP directly to the Colab runtime.'\n"
    "    )\n"
    "if len(zip_candidates) > 1:\n"
    "    raise FileNotFoundError(\n"
    "        f'Multiple ZIP files found in /content: {zip_candidates}. '\n"
    "        'Please ensure only the dataset ZIP is present.'\n"
    "    )\n"
    "zip_path = zip_candidates[0]\n"
    "print(f'ZIP found:            {zip_path}')\n"
    "print(f'Extraction directory: {EXTRACT_DIR}')\n"
    "\n"
    "# ── 2. Extract ZIP to /content/amazon_ml_challenge ────────────────────────\n"
    "os.makedirs(EXTRACT_DIR, exist_ok=True)\n"
    "with zipfile.ZipFile(zip_path, 'r') as zf:\n"
    "    zf.extractall(EXTRACT_DIR)\n"
    "\n"
    "# ── 3. Recursively locate all four required TSV files ─────────────────────\n"
    "found = {}\n"
    "for root, dirs, files in os.walk(EXTRACT_DIR):\n"
    "    for fname in files:\n"
    "        if fname in REQUIRED_FILES and fname not in found:\n"
    "            found[fname] = os.path.join(root, fname)\n"
    "\n"
    "missing = [f for f in REQUIRED_FILES if f not in found]\n"
    "if missing:\n"
    "    raise FileNotFoundError(\n"
    "        f'Could not locate the following files after extraction: {missing}\\n'\n"
    "        f'Files found so far: {list(found.keys())}\\n'\n"
    "        f'Check the ZIP structure under {EXTRACT_DIR}'\n"
    "    )\n"
    "\n"
    "# ── 4. Set DATA_ROOT to the directory containing the four files ───────────\n"
    "# All four files must be in the same directory\n"
    "dirs_found = {os.path.dirname(p) for p in found.values()}\n"
    "if len(dirs_found) != 1:\n"
    "    raise FileNotFoundError(\n"
    "        f'Required TSV files are spread across multiple directories: {dirs_found}. '\n"
    "        'Expected all four files in the same folder.'\n"
    "    )\n"
    "DATA_ROOT = dirs_found.pop()\n"
    "print(f'Resolved DATA_ROOT:   {DATA_ROOT}')\n"
    "\n"
    "# ── 5. Print each file with full path and size ────────────────────────────\n"
    "print()\n"
    "print(f'  {\"File\":<28} {\"Size\":>10}  Path')\n"
    "print(f'  {\"-\"*28} {\"-\"*10}  {\"-\"*40}')\n"
    "for fname in REQUIRED_FILES:\n"
    "    fpath = found[fname]\n"
    "    size_mb = os.path.getsize(fpath) / (1024 * 1024)\n"
    "    print(f'  {fname:<28} {size_mb:>8.1f}MB  {fpath}')\n"
    "\n"
    "# ── 6. Final verification ─────────────────────────────────────────────────\n"
    "for fname in REQUIRED_FILES:\n"
    "    if not os.path.isfile(found[fname]):\n"
    "        raise FileNotFoundError(f'Verification failed: {found[fname]} is not a file.')\n"
    "print()\n"
    "print('All 4 required training files verified. DATA_ROOT is set.')\n"
    "print('Ready to proceed — datasets will be loaded in Section 4.')\n"
)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=2, ensure_ascii=False)
print('Section 2 cells updated.')
