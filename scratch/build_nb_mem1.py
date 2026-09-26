import json, os

nb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb'))
with open(nb_path, 'r', encoding='utf-8') as f:
    NB = json.load(f)

src = NB['cells'][14]['source']

OLD = (
    "    SELECT *,\n"
    "        row_number() over (\n"
    "            partition by entity_id\n"
    "            order by entity_id  -- deterministic tie-break\n"
    "        ) as rn\n"
    "    FROM target_pool_raw\n"
)

NEW = (
    "    SELECT *,\n"
    "        row_number() over (\n"
    "            partition by entity_id\n"
    "            -- true-target rows first (entity_id IN true_target_ids = 1),\n"
    "            -- then deterministic tie-break by entity_id\n"
    "            order by\n"
    "                CASE WHEN entity_id IN (SELECT target_id FROM true_target_ids)\n"
    "                     THEN 0 ELSE 1 END ASC,\n"
    "                entity_id ASC\n"
    "        ) as rn\n"
    "    FROM target_pool_raw\n"
)

assert OLD in src, 'Could not find dedup ORDER BY block in cell 14'
NB['cells'][14]['source'] = src.replace(OLD, NEW, 1)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=2, ensure_ascii=False)
print('Fix 1 applied: dedup ORDER BY now prefers true-target rows.')
