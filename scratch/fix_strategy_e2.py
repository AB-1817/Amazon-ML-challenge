"""Replace Strategy E cell 38 by reading source from strategy_e_src.py."""
import json, ast, os

NB  = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'
SRC = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\scratch\strategy_e_src.py'

with open(SRC, 'r', encoding='utf-8') as f:
    new_source = f.read()

# Syntax-check the new source before injecting
# Strip magic lines (none here) then parse
try:
    ast.parse(new_source)
    print('Syntax check: PASS')
except SyntaxError as e:
    raise SystemExit(f'Syntax error in new source: {e}')

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Verify target cell
assert '# -- Strategy E: char 3/4-gram TF-IDF sparse retrieval' in nb['cells'][38]['source'] or \
       '# ── Strategy E: char 3/4-gram TF-IDF sparse retrieval' in nb['cells'][38]['source'], \
    f"Cell 38 does not look like Strategy E. First 80 chars: {nb['cells'][38]['source'][:80]}"

nb['cells'][38]['source'] = new_source

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f'Cell 38 replaced. Total cells: {len(nb["cells"])}')
