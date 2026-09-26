"""Step 1: Remove duplicate cells 34-37 from notebook."""
import json

NB = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Before: {len(nb["cells"])} cells')
# Remove cells 34-37 (duplicates of 30-33)
nb['cells'] = nb['cells'][:34]
print(f'After:  {len(nb["cells"])} cells')

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print('Duplicates removed.')
