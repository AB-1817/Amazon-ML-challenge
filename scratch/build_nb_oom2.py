import json, os

nb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb'))
with open(nb_path, 'r', encoding='utf-8') as f:
    NB = json.load(f)

sec7_code_idx = 14
src = NB['cells'][sec7_code_idx]['source']

# The blocking section ends after the raw_pairs print line.
# We keep everything up to and including that line, then replace the rest.
KEEP_UNTIL = "print(f'Generated {raw_pairs:,} distinct raw candidate pairs in {time.time()-t_block:.2f}s.')"
cut = src.find(KEEP_UNTIL)
assert cut != -1, 'Could not find cut point'
kept = src[:cut + len(KEEP_UNTIL)]

# Build the new chunked ranking section
CHUNKED = (
    "\n\n"
    "# ── CHUNKED TOP-K RANKING ─────────────────────────────────────────────────\n"
    "# Score and rank candidates in S1 chunks of RANK_CHUNK_SIZE.\n"
    "# Only top-K rows per S1 are written to topk_candidates.\n"
    "# The full scored intermediate is never materialized in memory.\n"
    "RANK_CHUNK_SIZE = 10000\n\n"
    "print(f'Chunked ranking: K={TOP_K}, chunk_size={RANK_CHUNK_SIZE}...')\n"
    "t_rank = time.time()\n\n"
    "# Fetch the ordered list of distinct S1 IDs from raw_candidate_pairs\n"
    "s1_ids_all = [r[0] for r in con.execute(\n"
    "    'SELECT DISTINCT s1_id FROM raw_candidate_pairs ORDER BY s1_id'\n"
    ").fetchall()]\n"
    "n_s1 = len(s1_ids_all)\n"
    "print(f'  S1 entities with raw candidates: {n_s1:,}')\n\n"
    "# Create the persistent topk_candidates table (empty, correct schema)\n"
    "con.execute(\"\"\"\n"
    "CREATE OR REPLACE TABLE topk_candidates (\n"
    "    s1_id VARCHAR, target_id VARCHAR, pass_count BIGINT,\n"
    "    split VARCHAR, s1_country VARCHAR,\n"
    "    s1_raw_name VARCHAR, s1_raw_addr VARCHAR,\n"
    "    s1_norm_name VARCHAR, s1_norm_addr VARCHAR,\n"
    "    s1_postal VARCHAR, s1_door VARCHAR,\n"
    "    t_country VARCHAR,\n"
    "    t_raw_name VARCHAR, t_raw_addr VARCHAR,\n"
    "    t_norm_name VARCHAR, t_norm_addr VARCHAR,\n"
    "    t_postal VARCHAR, t_door VARCHAR,\n"
    "    jaccard_score DOUBLE, candidate_rank BIGINT\n"
    ");\n"
    "\"\"\")\n\n"
    "n_chunks = (n_s1 + RANK_CHUNK_SIZE - 1) // RANK_CHUNK_SIZE\n"
    "total_written = 0\n\n"
    "for chunk_idx in range(n_chunks):\n"
    "    chunk_s1 = s1_ids_all[chunk_idx * RANK_CHUNK_SIZE : (chunk_idx + 1) * RANK_CHUNK_SIZE]\n"
    "    con.register('chunk_s1_ids', __import__('pandas').DataFrame({'s1_id': chunk_s1}))\n\n"
    "    con.execute(f\"\"\"\n"
    "    INSERT INTO topk_candidates\n"
    "    SELECT s1_id, target_id, pass_count,\n"
    "           split, s1_country,\n"
    "           s1_raw_name, s1_raw_addr,\n"
    "           s1_norm_name, s1_norm_addr,\n"
    "           s1_postal, s1_door,\n"
    "           t_country,\n"
    "           t_raw_name, t_raw_addr,\n"
    "           t_norm_name, t_norm_addr,\n"
    "           t_postal, t_door,\n"
    "           jaccard_score, candidate_rank\n"
    "    FROM (\n"
    "        SELECT\n"
    "            c.s1_id, c.target_id, c.pass_count,\n"
    "            s.split, s.country AS s1_country,\n"
    "            s.raw_name AS s1_raw_name, s.raw_addr AS s1_raw_addr,\n"
    "            s.norm_name AS s1_norm_name, s.norm_addr AS s1_norm_addr,\n"
    "            s.postal AS s1_postal, s.door AS s1_door,\n"
    "            t.country AS t_country,\n"
    "            t.raw_name AS t_raw_name, t.raw_addr AS t_raw_addr,\n"
    "            t.norm_name AS t_norm_name, t.norm_addr AS t_norm_addr,\n"
    "            t.postal AS t_postal, t.door AS t_door,\n"
    "            (0.6 * (len(list_intersect(string_split(s.norm_name,' '),string_split(t.norm_name,' ')))::float /\n"
    "                    nullif(len(list_distinct(list_concat(string_split(s.norm_name,' '),string_split(t.norm_name,' ')))),0)) +\n"
    "             0.4 * (len(list_intersect(string_split(s.norm_addr,' '),string_split(t.norm_addr,' ')))::float /\n"
    "                    nullif(len(list_distinct(list_concat(string_split(s.norm_addr,' '),string_split(t.norm_addr,' ')))),0)))\n"
    "                AS jaccard_score,\n"
    "            row_number() OVER (\n"
    "                PARTITION BY c.s1_id\n"
    "                ORDER BY (0.6*(len(list_intersect(string_split(s.norm_name,' '),string_split(t.norm_name,' ')))::float /\n"
    "                               nullif(len(list_distinct(list_concat(string_split(s.norm_name,' '),string_split(t.norm_name,' ')))),0)) +\n"
    "                          0.4*(len(list_intersect(string_split(s.norm_addr,' '),string_split(t.norm_addr,' ')))::float /\n"
    "                               nullif(len(list_distinct(list_concat(string_split(s.norm_addr,' '),string_split(t.norm_addr,' ')))),0))) DESC,\n"
    "                c.pass_count DESC\n"
    "            ) AS candidate_rank\n"
    "        FROM raw_candidate_pairs c\n"
    "        JOIN chunk_s1_ids        x ON c.s1_id = x.s1_id\n"
    "        JOIN s1_split            s ON c.s1_id = s.entity_id\n"
    "        JOIN target_pool         t ON c.target_id = t.entity_id\n"
    "    ) ranked\n"
    "    WHERE candidate_rank <= {TOP_K};\n"
    "    \"\"\")\n\n"
    "    chunk_written = con.execute('SELECT count(*) FROM topk_candidates').fetchone()[0] - total_written\n"
    "    total_written += chunk_written\n"
    "    if (chunk_idx + 1) % 5 == 0 or chunk_idx == n_chunks - 1:\n"
    "        print(f'  Chunk {chunk_idx+1:>3}/{n_chunks}  '\n"
    "              f'S1s processed: {min((chunk_idx+1)*RANK_CHUNK_SIZE, n_s1):>7,}  '\n"
    "              f'Rows written so far: {total_written:>9,}')\n\n"
    "topk_cnt = con.execute('SELECT count(*) FROM topk_candidates').fetchone()[0]\n"
    "per_s1 = [r[0] for r in con.execute('SELECT count(*) FROM topk_candidates GROUP BY s1_id').fetchall()]\n"
    "per_s1 = [x for x in per_s1 if x > 0]\n"
    "rank_elapsed = time.time() - t_rank\n\n"
    "print(f'\\nChunked Ranking Complete in {rank_elapsed:.1f}s:')\n"
    "print(f'  Total Retained Pairs:    {topk_cnt:,} across {len(per_s1):,} S1 entities')\n"
    "print(f'  Max possible (K*S1):     {TOP_K * n_s1:,}')\n"
    "print(f'  Avg Candidates/S1:       {sum(per_s1)/len(per_s1):.1f}')\n"
    "print(f'  Median:                  {sorted(per_s1)[len(per_s1)//2]}')\n"
    "print(f'  Maximum:                 {max(per_s1)}')\n"
    "if topk_cnt > TOP_K * n_s1:\n"
    "    raise AssertionError(f'RANKING ERROR: {topk_cnt} rows > theoretical max {TOP_K * n_s1}')\n"
    "print('  [PASS] Row count within theoretical maximum.')\n"
)

NB['cells'][sec7_code_idx]['source'] = kept + CHUNKED

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=2, ensure_ascii=False)
print('Chunked ranking replacement applied to cell 14.')
