import json, os

nb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb'))
with open(nb_path, 'r', encoding='utf-8') as f:
    NB = json.load(f)

# ── LOCATE CELLS ───────────────────────────────────────────────────────────────
sec4_code_idx  = None   # DuckDB connect + memory_limit
sec7_md_idx    = None   # Section 7 markdown
sec7_code_idx  = None   # Section 7 code (blocking + ranking)

for i, cell in enumerate(NB['cells']):
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if cell['cell_type'] == 'code' and "memory_limit='6GB'" in src:
        sec4_code_idx = i
    if cell['cell_type'] == 'markdown' and 'Section 7' in src and '8-Pass Blocking' in src:
        sec7_md_idx = i
    if cell['cell_type'] == 'code' and 'raw_candidate_pairs' in src and 'ranked_candidates' in src:
        sec7_code_idx = i

print(f'sec4_code_idx={sec4_code_idx}  sec7_md_idx={sec7_md_idx}  sec7_code_idx={sec7_code_idx}')
assert sec4_code_idx  is not None
assert sec7_md_idx    is not None
assert sec7_code_idx  is not None

# ── FIX 1: lower memory limit to 4GB, add temp dir ────────────────────────────
src4 = NB['cells'][sec4_code_idx]['source']
src4 = src4.replace("SET memory_limit='6GB';", "SET memory_limit='4GB';")
# Insert temp_directory line after threads line
src4 = src4.replace(
    "con.execute('SET threads=8;')",
    "con.execute('SET threads=4;')\n"
    "con.execute(\"SET temp_directory='/content/duckdb_tmp';\")\n"
    "os.makedirs('/content/duckdb_tmp', exist_ok=True)"
)
NB['cells'][sec4_code_idx]['source'] = src4

# ── FIX 2: update Section 7 markdown ──────────────────────────────────────────
NB['cells'][sec7_md_idx]['source'] = (
    "### Section 7: Final 8-Pass Blocking & Chunked Candidate Ranking\n\n"
    "**Development Experiment**: true targets + random S2/S3 distractors (150k each).\n"
    "Target pool is deduplicated to one unique record per entity_id before blocking.\n\n"
    "**Memory Architecture**: Raw candidate pairs (~3.25M) are generated once and stored\n"
    "on disk. Jaccard scoring + top-K ranking is then done in **S1 chunks of 10,000**,\n"
    "writing only the top-25 rows per S1 to the persistent `topk_candidates` table.\n"
    "The full scored intermediate is never materialized in memory.\n\n"
    "> **Validation Scope**: This model validation measures matcher performance "
    "**conditional on the true target being present in the candidate-generation pool**. "
    "End-to-end blocking recall is evaluated separately by `blocking_experiment.py`. "
    "The F0.5 reported here is **not** the final end-to-end competition score.\n"
)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=2, ensure_ascii=False)
print('Fix 1 and Fix 2 applied.')
