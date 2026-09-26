"""Append Section 21 (blocking comparison) cells to notebook."""
import json

NB = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

# ── Section 21 markdown ────────────────────────────────────────────────────
nb['cells'].append(md(
"### Section 21: Blocking Strategy Comparison\n\n"
"Evaluate multiple blocking strategies on the **same validation S1 entities** used by the baseline.\n"
"For each strategy measure: candidate pairs, avg/p95 candidates per S1, recall@K, runtime, peak memory.\n\n"
"Strategies tested:\n"
"- **A** — Baseline 8-pass K=25 (already built as `topk_candidates`)\n"
"- **B** — 8-pass K=50\n"
"- **C** — 8-pass K=100\n"
"- **D** — 8-pass + 4 permutation passes, K=50\n"
"- **E** — 8-pass + char 3/4-gram TF-IDF sparse retrieval, K=50\n"
"- **F** — 8-pass + address-driven passes (postal+door, street tokens), K=50\n\n"
"**Memory safety**: all blocking is done in DuckDB (disk-backed). "
"Char n-gram retrieval uses sparse scipy matrices processed country-by-country. "
"No full Cartesian product is ever materialized.\n"
))

# ── Section 21 code ────────────────────────────────────────────────────────
nb['cells'].append(code(
"import gc, time\n"
"import numpy as np\n"
"import pandas as pd\n\n"
"# ── helpers ───────────────────────────────────────────────────────────────\n"
"def _recall_at_k(con, topk_table, k, val_gt):\n"
"    \"\"\"Candidate recall@k for val S1s that have at least one true match.\"\"\"\n"
"    rows = con.execute(f\"\"\"\n"
"        SELECT c.s1_id, c.target_id\n"
"        FROM {topk_table} c\n"
"        WHERE c.split='val' AND c.candidate_rank <= {k}\n"
"    \"\"\").fetchall()\n"
"    cands = {}\n"
"    for s1, tgt in rows:\n"
"        cands.setdefault(s1, set()).add(tgt)\n"
"    hits = 0; total = 0\n"
"    for sid, true_set in val_gt.items():\n"
"        if not true_set: continue\n"
"        total += len(true_set)\n"
"        hits  += len(true_set & cands.get(sid, set()))\n"
"    return hits / total if total > 0 else 0.0\n\n"
"def _cand_stats(con, topk_table, split='val'):\n"
"    rows = con.execute(f\"\"\"\n"
"        SELECT count(*) as n, count(distinct s1_id) as s1s\n"
"        FROM {topk_table} WHERE split='{split}'\n"
"    \"\"\").fetchone()\n"
"    total, s1s = rows\n"
"    per = con.execute(f\"\"\"\n"
"        SELECT count(*) as c FROM {topk_table}\n"
"        WHERE split='{split}' GROUP BY s1_id ORDER BY c\n"
"    \"\"\").fetchall()\n"
"    per = [r[0] for r in per]\n"
"    avg = sum(per)/len(per) if per else 0\n"
"    p95 = per[int(len(per)*0.95)] if per else 0\n"
"    return total, s1s, avg, p95\n\n"
"def _mem_mb():\n"
"    try:\n"
"        import psutil\n"
"        return psutil.Process().memory_info().rss / 1024 / 1024\n"
"    except ImportError:\n"
"        return 0.0\n\n"
"blocking_results = []\n\n"
"# ── Strategy A: baseline 8-pass K=25 (already built) ─────────────────────\n"
"print('Strategy A: baseline 8-pass K=25 (pre-built)...')\n"
"t0 = time.time()\n"
"tot, s1s, avg, p95 = _cand_stats(con, 'topk_candidates')\n"
"r25 = _recall_at_k(con, 'topk_candidates', 25, val_gt)\n"
"r50 = _recall_at_k(con, 'topk_candidates', 50, val_gt)  # same table, K capped at 25\n"
"blocking_results.append({\n"
"    'strategy':'A_8pass_K25','K':25,\n"
"    'candidate_pairs':tot,'avg_cands':round(avg,1),'p95_cands':p95,\n"
"    'recall_at_25':round(r25,4),'recall_at_50':round(r25,4),'recall_at_100':round(r25,4),\n"
"    'runtime_s':round(time.time()-t0,1),'peak_mem_mb':round(_mem_mb(),1)\n"
"})\n"
"print(f'  pairs={tot:,}  avg={avg:.1f}  p95={p95}  recall@25={r25:.4f}')\n"
))

print(f'Cells after part2: {len(nb["cells"])}')
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print('Part 2 written.')
