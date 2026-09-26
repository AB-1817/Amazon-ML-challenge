"""Replace Strategy E (cell 38) with memory-safe block-wise sparse top-K retrieval.

Key design:
  - NEVER calls .toarray() on Q @ T.T
  - Processes target matrix in column-blocks (TARGET_BLOCK rows at a time)
  - For each S1 query chunk x target block: multiplies two sparse matrices,
    converts ONLY that small block result to dense (bounded size), updates
    a running top-N heap per query
  - Peak dense allocation = NGRAM_CHUNK * TARGET_BLOCK floats (e.g. 500*2000 = 1M floats = 4MB)
  - If a country's target pool exceeds FEASIBILITY_LIMIT, skips with a clear report
"""
import json

NB = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Verify we are touching the right cell
assert '# ── Strategy E: char 3/4-gram TF-IDF sparse retrieval' in nb['cells'][38]['source'], \
    "Cell 38 is not Strategy E — aborting"

NEW_SRC = """\
# ── Strategy E: char 3/4-gram TF-IDF sparse retrieval, K=50 ─────────────
# Memory-safe block-wise top-K retrieval.
# NEVER calls .toarray() on the full Q @ T.T product.
# Processes target matrix in TARGET_BLOCK-row blocks.
# Peak dense allocation per iteration = NGRAM_CHUNK * TARGET_BLOCK floats.
print('Strategy E: char n-gram TF-IDF sparse retrieval (block-wise), K=50...')
t0 = time.time(); m0 = _mem_mb()
K_E = 50
NGRAM_TOP_N   = 30    # top-N per S1 from TF-IDF before merging with SQL passes
NGRAM_CHUNK   = 500   # S1 query rows per inner iteration
TARGET_BLOCK  = 2000  # target rows per block  →  500*2000 = 1M floats ≈ 4MB dense
MIN_SCORE     = 0.05  # discard near-zero cosine matches
FEASIBILITY_LIMIT = 300000  # skip country if target pool exceeds this (OOM risk)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import heapq, gc

def _sparse_topk_block(Q_norm, T_norm, top_n, min_score, q_chunk, t_block):
    """
    Compute top-n cosine matches for each row of Q_norm against T_norm
    using block-wise multiplication.  Never materialises the full matrix.

    Q_norm : csr_matrix  (n_queries, n_features)
    T_norm : csr_matrix  (n_targets, n_features)
    Returns list of length n_queries, each element is a list of (score, tgt_idx).
    """
    n_q   = Q_norm.shape[0]
    n_t   = T_norm.shape[0]
    # Running top-n heap per query: min-heap of (score, tgt_idx)
    heaps = [[] for _ in range(n_q)]

    t_start = 0
    while t_start < n_t:
        t_end   = min(t_start + t_block, n_t)
        T_block = T_norm[t_start:t_end]          # sparse slice, no copy
        # Sparse x sparse → sparse result, then .toarray() on the SMALL block only
        # Shape: (n_q, t_end-t_start)  ← bounded by NGRAM_CHUNK * TARGET_BLOCK
        block_scores = (Q_norm @ T_block.T).toarray()  # safe: small dense block
        for qi in range(n_q):
            row = block_scores[qi]
            for local_ti, sc in enumerate(row):
                if sc <= min_score:
                    continue
                global_ti = t_start + local_ti
                if len(heaps[qi]) < top_n:
                    heapq.heappush(heaps[qi], (sc, global_ti))
                elif sc > heaps[qi][0][0]:
                    heapq.heapreplace(heaps[qi], (sc, global_ti))
        del block_scores
        t_start = t_end

    return heaps

# Create staging table for n-gram candidates
con.execute(\"\"\"
CREATE OR REPLACE TABLE ngram_cands_E (
    s1_id VARCHAR, target_id VARCHAR, ngram_score FLOAT
);
\"\"\")

countries = [r[0] for r in con.execute(
    "SELECT DISTINCT country FROM s1_split ORDER BY country"
).fetchall()]
print(f'  Processing {len(countries)} countries...')

total_ngram_pairs = 0
skipped_countries = []

for ctry in countries:
    t_ctry = time.time()

    # Load target pool for this country
    tgt_df = con.execute(f\"\"\"
        SELECT entity_id, coalesce(norm_name,'') as name
        FROM target_pool WHERE country='{ctry}'
    \"\"\").df()
    n_tgt = len(tgt_df)

    if n_tgt < 2:
        del tgt_df; gc.collect(); continue

    # Feasibility gate: skip if target pool is too large
    if n_tgt > FEASIBILITY_LIMIT:
        print(f'  [{ctry}] SKIP — target pool {n_tgt:,} > limit {FEASIBILITY_LIMIT:,}')
        skipped_countries.append({'country': ctry, 'n_tgt': n_tgt, 'reason': 'exceeds_feasibility_limit'})
        del tgt_df; gc.collect(); continue

    # Load S1 entities for this country
    s1_df = con.execute(f\"\"\"
        SELECT entity_id, coalesce(norm_name,'') as name
        FROM s1_split WHERE country='{ctry}'
    \"\"\").df()
    n_s1 = len(s1_df)

    if n_s1 == 0:
        del tgt_df, s1_df; gc.collect(); continue

    # Fit TF-IDF on target names (char 3-4 gram)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4),
                          min_df=1, max_features=50000, sublinear_tf=True)
    try:
        T = vec.fit_transform(tgt_df['name'].tolist())
    except Exception as e:
        print(f'  [{ctry}] TF-IDF fit failed: {e}')
        del tgt_df, s1_df; gc.collect(); continue

    T = normalize(T, norm='l2', copy=False)
    tgt_ids = tgt_df['entity_id'].values
    del tgt_df; gc.collect()

    # Estimate peak dense block size and warn if large
    peak_block_mb = NGRAM_CHUNK * TARGET_BLOCK * 4 / 1024 / 1024
    if peak_block_mb > 200:
        print(f'  [{ctry}] WARNING: peak block ~{peak_block_mb:.0f}MB — reducing TARGET_BLOCK')
        t_block_ctry = max(500, int(200 * 1024 * 1024 / (NGRAM_CHUNK * 4)))
    else:
        t_block_ctry = TARGET_BLOCK

    # Process S1 in NGRAM_CHUNK-row batches
    n_chunks_s1 = (n_s1 + NGRAM_CHUNK - 1) // NGRAM_CHUNK
    ctry_pairs  = 0
    rows_out    = []

    for ci in range(n_chunks_s1):
        s1_chunk = s1_df.iloc[ci * NGRAM_CHUNK : (ci + 1) * NGRAM_CHUNK]
        Q = vec.transform(s1_chunk['name'].tolist())
        Q = normalize(Q, norm='l2', copy=False)

        # Block-wise top-K — never materialises full Q x T^T
        heaps = _sparse_topk_block(Q, T, NGRAM_TOP_N, MIN_SCORE, NGRAM_CHUNK, t_block_ctry)

        s1_chunk_ids = s1_chunk['entity_id'].values
        for qi, heap in enumerate(heaps):
            for sc, ti in heap:
                rows_out.append((s1_chunk_ids[qi], tgt_ids[ti], float(sc)))

        del Q, heaps
        gc.collect()

        # Flush to DuckDB every 50k rows to avoid accumulating in RAM
        if len(rows_out) >= 50000:
            batch = pd.DataFrame(rows_out, columns=['s1_id', 'target_id', 'ngram_score'])
            con.register('_ng_batch', batch)
            con.execute('INSERT INTO ngram_cands_E SELECT * FROM _ng_batch')
            ctry_pairs += len(rows_out)
            del batch, rows_out; rows_out = []
            gc.collect()

    # Flush remainder
    if rows_out:
        batch = pd.DataFrame(rows_out, columns=['s1_id', 'target_id', 'ngram_score'])
        con.register('_ng_batch', batch)
        con.execute('INSERT INTO ngram_cands_E SELECT * FROM _ng_batch')
        ctry_pairs += len(rows_out)
        del batch, rows_out
        gc.collect()

    total_ngram_pairs += ctry_pairs
    ctry_mem = _mem_mb() - m0
    ctry_rt  = time.time() - t_ctry
    print(f'  [{ctry}] tgt={n_tgt:,}  s1={n_s1:,}  chunk={NGRAM_CHUNK}  '
          f'pairs={ctry_pairs:,}  mem_delta={ctry_mem:+.0f}MB  rt={ctry_rt:.1f}s')

    del s1_df, T, tgt_ids, vec
    gc.collect()

if skipped_countries:
    print(f'\\n  Skipped {len(skipped_countries)} countries (feasibility limit):')
    for sc in skipped_countries:
        print(f'    {sc[\"country\"]}  n_tgt={sc[\"n_tgt\"]:,}')

print(f'\\n  Total n-gram candidate pairs inserted: {total_ngram_pairs:,}')
print(f'  Peak process memory delta: {_mem_mb()-m0:+.0f}MB')

# Merge n-gram candidates with strategy D raw pairs
con.execute(\"\"\"
CREATE OR REPLACE TABLE raw_cands_E AS
SELECT s1_id, target_id, max(pass_count) as pass_count
FROM (
    SELECT s1_id, target_id, pass_count FROM raw_cands_D
    UNION ALL
    SELECT s1_id, target_id, 1 as pass_count FROM ngram_cands_E
) sub
GROUP BY s1_id, target_id;
\"\"\")
raw_e = con.execute('SELECT count(*) FROM raw_cands_E').fetchone()[0]
print(f'  Raw pairs E (D + n-gram): {raw_e:,}')

# Rank to K_E
con.execute('CREATE OR REPLACE TABLE topk_E (s1_id VARCHAR, target_id VARCHAR, pass_count BIGINT,'
            ' split VARCHAR, s1_country VARCHAR,'
            ' s1_raw_name VARCHAR, s1_raw_addr VARCHAR,'
            ' s1_norm_name VARCHAR, s1_norm_addr VARCHAR,'
            ' s1_postal VARCHAR, s1_door VARCHAR,'
            ' t_country VARCHAR, t_raw_name VARCHAR, t_raw_addr VARCHAR,'
            ' t_norm_name VARCHAR, t_norm_addr VARCHAR,'
            ' t_postal VARCHAR, t_door VARCHAR,'
            ' jaccard_score DOUBLE, candidate_rank BIGINT);')
s1_ids_e = [r[0] for r in con.execute('SELECT DISTINCT s1_id FROM raw_cands_E ORDER BY s1_id').fetchall()]
n_chunks_e2 = (len(s1_ids_e) + RANK_CHUNK_SIZE - 1) // RANK_CHUNK_SIZE
for ci in range(n_chunks_e2):
    chunk = s1_ids_e[ci*RANK_CHUNK_SIZE:(ci+1)*RANK_CHUNK_SIZE]
    con.register('_cids', pd.DataFrame({'s1_id': chunk}))
    con.execute(f\"\"\"
    INSERT INTO topk_E
    SELECT s1_id,target_id,pass_count,split,s1_country,s1_raw_name,s1_raw_addr,
           s1_norm_name,s1_norm_addr,s1_postal,s1_door,t_country,t_raw_name,t_raw_addr,
           t_norm_name,t_norm_addr,t_postal,t_door,jaccard_score,candidate_rank
    FROM (
        SELECT c.s1_id,c.target_id,c.pass_count,s.split,s.country AS s1_country,
               s.raw_name AS s1_raw_name,s.raw_addr AS s1_raw_addr,
               s.norm_name AS s1_norm_name,s.norm_addr AS s1_norm_addr,
               s.postal AS s1_postal,s.door AS s1_door,
               t.country AS t_country,t.raw_name AS t_raw_name,t.raw_addr AS t_raw_addr,
               t.norm_name AS t_norm_name,t.norm_addr AS t_norm_addr,
               t.postal AS t_postal,t.door AS t_door,
               (0.6*(len(list_intersect(string_split(s.norm_name,' '),string_split(t.norm_name,' ')))::float/
                     nullif(len(list_distinct(list_concat(string_split(s.norm_name,' '),string_split(t.norm_name,' ')))),0))+
                0.4*(len(list_intersect(string_split(s.norm_addr,' '),string_split(t.norm_addr,' ')))::float/
                     nullif(len(list_distinct(list_concat(string_split(s.norm_addr,' '),string_split(t.norm_addr,' ')))),0)))
               AS jaccard_score,
               row_number() OVER (PARTITION BY c.s1_id ORDER BY
                   (0.6*(len(list_intersect(string_split(s.norm_name,' '),string_split(t.norm_name,' ')))::float/
                         nullif(len(list_distinct(list_concat(string_split(s.norm_name,' '),string_split(t.norm_name,' ')))),0))+
                    0.4*(len(list_intersect(string_split(s.norm_addr,' '),string_split(t.norm_addr,' ')))::float/
                         nullif(len(list_distinct(list_concat(string_split(s.norm_addr,' '),string_split(t.norm_addr,' ')))),0))) DESC,
                   c.pass_count DESC) AS candidate_rank
        FROM raw_cands_E c
        JOIN _cids x ON c.s1_id=x.s1_id
        JOIN s1_split s ON c.s1_id=s.entity_id
        JOIN target_pool t ON c.target_id=t.entity_id
    ) ranked WHERE candidate_rank <= {K_E};
    \"\"\")
tot,s1s,avg,p95 = _cand_stats(con,'topk_E')
r25 = _recall_at_k(con,'topk_E',25,val_gt)
r50 = _recall_at_k(con,'topk_E',50,val_gt)
blocking_results.append({
    'strategy':'E_8pass_perm_ngram_K50','K':50,
    'candidate_pairs':tot,'avg_cands':round(avg,1),'p95_cands':p95,
    'recall_at_25':round(r25,4),'recall_at_50':round(r50,4),'recall_at_100':round(r50,4),
    'runtime_s':round(time.time()-t0,1),'peak_mem_mb':round(_mem_mb()-m0,1)
})
print(f'  pairs={tot:,}  avg={avg:.1f}  p95={p95}  recall@25={r25:.4f}  recall@50={r50:.4f}')
gc.collect()
"""

nb['cells'][38]['source'] = NEW_SRC

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print('Cell 38 replaced.')
