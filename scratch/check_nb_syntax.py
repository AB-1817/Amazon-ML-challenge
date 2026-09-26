import json, ast, sys, os

nb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb'))
nb = json.load(open(nb_path, encoding='utf-8'))

errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    # strip IPython shell magics
    lines = [l for l in src.splitlines() if not l.strip().startswith('!')]
    clean = '\n'.join(lines)
    try:
        ast.parse(clean)
    except SyntaxError as e:
        errors.append((i, str(e)))

code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
if errors:
    print(f'SYNTAX ERRORS in {len(errors)} cell(s):')
    for cell_idx, msg in errors:
        print(f'  Cell {cell_idx}: {msg}')
    sys.exit(1)
else:
    print(f'All {code_count} code cells pass Python syntax check.')
    print(f'Total cells: {len(nb["cells"])}  (code={code_count}, markdown={len(nb["cells"])-code_count})')
