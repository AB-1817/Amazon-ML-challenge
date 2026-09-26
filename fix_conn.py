import json, sys

path = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

old = "con = duckdb.connect(':memory:')"
new = (
    "DB_PATH = '/content/duckdb_training.db'\n"
    "if os.path.exists(DB_PATH):\n"
    "    os.remove(DB_PATH)\n"
    "con = duckdb.connect(DB_PATH)"
)

changed = 0
for cell in nb['cells']:
    if cell.get('cell_type') == 'code' and old in cell.get('source', ''):
        cell['source'] = cell['source'].replace(old, new, 1)
        changed += 1

print(f"Cells changed: {changed}")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("File written.")

# Verify
with open(path, encoding='utf-8') as f:
    nb2 = json.load(f)
for cell in nb2['cells']:
    if cell.get('cell_type') == 'code' and 'DB_PATH' in cell.get('source', ''):
        snippet = cell['source']
        start = snippet.find('DB_PATH')
        print("VERIFIED snippet:")
        print(snippet[start:start+120])
        break
