# -*- coding: utf-8 -*-
"""
chunked_13pass_experiment.py
============================
Tests whether 13-pass blocking under a chunked, on-disk-DuckDB architecture
is practical for full-scale inference.

Architecture:
  For each S1 chunk of CHUNK_SIZE entities:
    1. Run all 13 blocking passes against the full target pool
    2. Deduplicate within the chunk (DuckDB DISTINCT)
    3. Score with RapidFuzz: 0.6 * name_token_sort + 0.4 * addr_token_sort
    4. Keep top K=25 per S1 via min-heap (never holds all pairs in RAM)
    5. Append K=25 results to on-disk DuckDB table

Experiments:
  PART A — 10k eval set (same as blocking_experiment.py)
    Directly comparable to prior results:
      8-pass  Recall@25 = 80.64%  avg=143.7
      13-pass Recall@25 = 84.40%  avg=630.7 (in-memory full materialise)

  PART B — Full-scale stress test
    50k real US-partition S1 entities against full US target pool
    Measures per-chunk peak memory & runtime to extrapolate full-dataset cost
"""

import sys
import os
import time
import tracemalloc
import random
import json
import heapq
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
TRAIN_DIR = os.path.join(ROOT_DIR, "6ab10eb3b23ba_student_resource",
                         "student_resource", "dataset", "train")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

SCRATCH_DIR = os.path.join(ROOT_DIR, "scratch")
os.makedirs(SCRATCH_DIR, exist_ok=True)

ONDI_DB_PATH_A = os.path.join(SCRATCH_DIR, "cands_13pass_eval.duckdb")
ONDI_DB_PATH_B = os.path.join(SCRATCH_DIR, "cands_13pass_stress.duckdb")
OUTPUT_JSON    = os.path.join(SRC_DIR, "analysis", "chunked_13pass_results.json")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_EVAL_S1    = 10_000    # must match blocking_experiment.py for comparability
N_BACKGROUND = 200_000   # same background pool
RANDOM_SEED  = 42        # same seed

EVAL_CHUNK   = 2_000     # S1 chunk size for eval (5 chunks of 2k)
STRESS_CHUNK = 10_000    # S1 chunk size for stress test

K            = 25        # keep top K candidates per S1
SCORE_BATCH  = 10_000    # pairs per RapidFuzz scoring batch (bounds RAM)

W_NAME = 0.60
W_ADDR = 0.40

# ---------------------------------------------------------------------------
# Prior 8-pass result (empirically measured, for direct comparison)
# ---------------------------------------------------------------------------
PRIOR_8PASS = {
    "raw_recall": 0.8199,
    "recall@25":  0.8064,
    "avg_cands":  143.7,
    "total_pairs": 1_430_376,
    "time_s": 0.55,
}

# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def make_mem_con(gb=6, threads=8):
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{gb}GB';")
    con.execute(f"SET threads={threads};")
    con.execute("SET preserve_insertion_order=false;")
    return con


def make_disk_con(path, gb=6, threads=8):
    """Open / recreate an on-disk DuckDB database."""
    if os.path.exists(path):
        os.remove(path)
    con = duckdb.connect(path)
    con.execute(f"SET memory_limit='{gb}GB';")
    con.execute(f"SET threads={threads};")
    con.execute("SET preserve_insertion_order=false;")
    return con


def register_macros(con):
    con.execute("""
    CREATE OR REPLACE MACRO norm_text(s) AS
        lower(trim(regexp_replace(regexp_replace(
            regexp_replace(coalesce(s,''), 'https?://(?:www\\.)?|www\\.', '', 'g'),
            '\\.(?:com|org|net|in|co|gov|edu|fr|io)\\b', ' ', 'g'),
            '[^\\w\\s]', ' ', 'g')));
    """)
    con.execute("""
    CREATE OR REPLACE MACRO clean_legal(s) AS
        trim(regexp_replace(norm_text(s),
        '\\b(?:private\\s+limited|pvt\\s+ltd|pvt\\s+limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|holdings|holding|services|service|company|limited|llc|inc|corp|ltd|llp|co|sarl|sas|sa|sci|eurl|sasu)\\b',
        ' ', 'g'));
    """)
    con.execute("""
    CREATE OR REPLACE MACRO compact_name(s) AS
        regexp_replace(norm_text(s), '\\s+', '', 'g');
    """)
    con.execute("""
    CREATE OR REPLACE MACRO get_token(s, n) AS
        CASE WHEN array_length(string_split(trim(coalesce(s,'')), ' ')) >= n
             THEN string_split(trim(coalesce(s,'')), ' ')[n]
             ELSE '' END;
    """)
    con.execute("""
    CREATE OR REPLACE MACRO clean_addr(s) AS
        lower(trim(regexp_replace(
            regexp_replace(coalesce(s,''), '[^\\w\\s]', ' ', 'g'),
            '\\s+', ' ', 'g')));
    """)


PREP_SQL = """
SELECT
    entity_id,
    COALESCE(country,'')                                    AS country,
    norm_text(COALESCE(business_name,''))                   AS norm_name,
    clean_legal(COALESCE(business_name,''))                 AS core_name,
    clean_addr(COALESCE(business_address,''))               AS norm_addr,
    get_token(norm_text(COALESCE(business_name,'')),1)      AS p1,
    get_token(norm_text(COALESCE(business_name,'')),2)      AS p2,
    get_token(norm_text(COALESCE(business_name,'')),3)      AS p3,
    left(compact_name(COALESCE(business_name,'')),4)        AS cmp4,
    left(compact_name(COALESCE(business_name,'')),5)        AS cmp5,
    left(compact_name(COALESCE(business_name,'')),6)        AS cmp6,
    regexp_extract(clean_addr(COALESCE(business_address,'')), '\\b(\\d{3,})\\b',1) AS postal,
    regexp_extract(clean_addr(COALESCE(business_address,'')), '^(\\d+)',1)         AS door,
    get_token(clean_addr(COALESCE(business_address,'')),1)  AS a1,
    get_token(clean_addr(COALESCE(business_address,'')),2)  AS a2
FROM src_raw
"""


def load_and_preprocess(con, tsv_path, out_table, filter_ids_df=None):
    """Load TSV into DuckDB, optionally filter to specific entity_ids."""
    tsv_fwd = tsv_path.replace("\\", "/")
    con.execute(f"""
    CREATE OR REPLACE TABLE src_raw AS
    SELECT * FROM read_csv_auto('{tsv_fwd}', delim='\t', header=true, quote='');
    """)
    if filter_ids_df is not None:
        con.register("_filter_ids", filter_ids_df)
        con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        {PREP_SQL}
        WHERE entity_id IN (SELECT entity_id FROM _filter_ids);
        """)
    else:
        con.execute(f"CREATE OR REPLACE TABLE {out_table} AS {PREP_SQL};")
    return con.execute(f"SELECT count(*) FROM {out_table}").fetchone()[0]


# ---------------------------------------------------------------------------
# 13-pass blocking SQL
# ---------------------------------------------------------------------------

def sql_13pass(s1_tbl, tgt_tbl):
    return f"""
    SELECT s.entity_id AS s1_id, t.entity_id AS target_id
    FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2
    WHERE length(s.p1)>=3 AND length(s.p2)>=3
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.p1=t.p1 AND s.postal=t.postal
    WHERE length(s.p1)>=3 AND s.postal!=''
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.p1=t.p1 AND s.door=t.door
    WHERE length(s.p1)>=3 AND s.door!=''
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.cmp5=t.cmp5
    WHERE length(s.cmp5)>=5
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.p1=t.p1
    WHERE length(s.p1)>=5
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.postal=t.postal AND s.door=t.door
    WHERE s.postal!='' AND length(s.door)>=2
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.door=t.door AND s.a1=t.a1
    WHERE length(s.door)>=2 AND length(s.a1)>=4
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.door=t.door AND s.a2=t.a2
    WHERE length(s.door)>=2 AND length(s.a2)>=4
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.p2=t.p1
    WHERE length(s.p2)>=4
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.p1=t.p2
    WHERE length(s.p1)>=4
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.p3=t.p1
    WHERE length(s.p3)>=4
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.postal=t.postal AND s.cmp4=t.cmp4
    WHERE s.postal!='' AND length(s.cmp4)>=4
    UNION
    SELECT s.entity_id, t.entity_id FROM {s1_tbl} s JOIN {tgt_tbl} t
      ON s.country=t.country AND s.door=t.door AND s.cmp4=t.cmp4
    WHERE length(s.door)>=2 AND length(s.cmp4)>=4
    """


# ---------------------------------------------------------------------------
# Core: chunked 13-pass blocking with on-disk persistence
# ---------------------------------------------------------------------------

def run_chunked_13pass(
    mem_con,          # in-memory DuckDB (for blocking computations)
    disk_con,         # on-disk DuckDB (for persisting top-K results)
    s1_ids,           # list of S1 entity IDs to block
    s1_info,          # dict: s1_id -> (norm_name, norm_addr)
    target_info,      # dict: target_id -> (norm_name, norm_addr)
    gt_pairs,         # dict: s1_id -> set of true target_ids (for recall measurement)
    chunk_size,
    k=K,
    score_batch=SCORE_BATCH,
    label="",
):
    """
    Run 13-pass blocking in S1 chunks, score with RapidFuzz token_sort_ratio,
    persist top-K to on-disk DuckDB. Returns recall metrics and stats.
    """
    # Create results table on disk
    disk_con.execute("""
    CREATE OR REPLACE TABLE top_k_candidates (
        s1_id VARCHAR,
        target_id VARCHAR,
        score FLOAT
    );
    """)

    n_chunks   = (len(s1_ids) + chunk_size - 1) // chunk_size
    t_total    = time.time()
    peak_mem_mb = 0.0

    chunk_raw_pair_counts = []   # raw (pre-dedup) pairs per chunk
    chunk_dedup_counts    = []   # after DISTINCT
    chunk_times           = []

    for ci in range(n_chunks):
        t_chunk = time.time()
        tracemalloc.start()

        chunk_ids = s1_ids[ci * chunk_size: (ci + 1) * chunk_size]
        chunk_df  = pd.DataFrame({"entity_id": chunk_ids})
        mem_con.register("_chunk_ids", chunk_df)

        # Build chunk_s1 table
        mem_con.execute("""
        CREATE OR REPLACE TABLE chunk_s1 AS
        SELECT * FROM eval_s1_prep
        WHERE entity_id IN (SELECT entity_id FROM _chunk_ids);
        """)

        # Run 13-pass blocking — DISTINCT to dedup within chunk
        pairs_df = mem_con.execute(f"""
        SELECT DISTINCT s1_id, target_id
        FROM ({sql_13pass('chunk_s1', 'target_prep')}) sub
        """).df()

        n_raw = len(pairs_df)
        chunk_raw_pair_counts.append(n_raw)
        chunk_dedup_counts.append(n_raw)  # already distinct

        _, cur_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mb = max(peak_mem_mb, cur_peak / 1e6)

        if n_raw == 0:
            chunk_times.append(time.time() - t_chunk)
            print(f"  Chunk {ci+1}/{n_chunks}: 0 pairs, skipping", flush=True)
            continue

        # ------------------------------------------------------------------
        # Score with RapidFuzz token_sort_ratio in batches
        # Use per-S1 min-heap to maintain top-K without holding all scores
        # ------------------------------------------------------------------
        # heap: {s1_id -> [(score, tgt_id), ...]} — min-heap of size K
        heaps = defaultdict(list)

        s1_ids_arr  = pairs_df["s1_id"].values
        tgt_ids_arr = pairs_df["target_id"].values
        n_pairs     = len(s1_ids_arr)

        for bi in range(0, n_pairs, score_batch):
            batch_s1  = s1_ids_arr[bi: bi + score_batch]
            batch_tgt = tgt_ids_arr[bi: bi + score_batch]

            for s1_id, tgt_id in zip(batch_s1, batch_tgt):
                s1_name, s1_addr = s1_info.get(s1_id,  ("", ""))
                t_name,  t_addr  = target_info.get(tgt_id, ("", ""))

                name_sc = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
                addr_sc = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
                score   = W_NAME * name_sc + W_ADDR * addr_sc

                h = heaps[s1_id]
                if len(h) < k:
                    heapq.heappush(h, (score, tgt_id))
                elif score > h[0][0]:
                    heapq.heapreplace(h, (score, tgt_id))

        # Flush top-K to disk
        rows_to_insert = []
        for s1_id, h in heaps.items():
            for score, tgt_id in h:
                rows_to_insert.append({"s1_id": s1_id, "target_id": tgt_id, "score": score})

        if rows_to_insert:
            insert_df = pd.DataFrame(rows_to_insert)
            disk_con.register("_insert_batch", insert_df)
            disk_con.execute("""
            INSERT INTO top_k_candidates
            SELECT s1_id, target_id, score FROM _insert_batch;
            """)

        t_elapsed = time.time() - t_chunk
        chunk_times.append(t_elapsed)

        print(f"  Chunk {ci+1}/{n_chunks}: {n_raw:,} pairs -> scored -> "
              f"{len(rows_to_insert):,} top-K inserted  [{t_elapsed:.2f}s  "
              f"peak {peak_mem_mb:.0f}MB]", flush=True)

    mem_con.execute("DROP TABLE IF EXISTS chunk_s1;")
    total_time = time.time() - t_total

    # ------------------------------------------------------------------
    # Aggregate: de-dup top-K across chunks (same S1 may appear in chunk
    # boundaries if chunks overlap — here they don't, so just read back)
    # For each S1, keep final top-K by score
    # ------------------------------------------------------------------
    print("  Consolidating on-disk top-K results...", flush=True)
    final_df = disk_con.execute("""
    SELECT s1_id, target_id, score,
           ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY score DESC) AS rn
    FROM top_k_candidates
    """).df()
    final_df = final_df[final_df["rn"] <= k]

    # Build final candidate map: s1_id -> [target_id, ...]
    final_cands = defaultdict(list)
    for _, row in final_df.iterrows():
        final_cands[row["s1_id"]].append(row["target_id"])

    # ------------------------------------------------------------------
    # Recall measurement
    # ------------------------------------------------------------------
    total_true = 0
    matched_raw = 0
    matched_k   = {10: 0, 25: 0, 50: 0}
    total_for_k = 0

    for s1_id in s1_ids:
        true_set = gt_pairs.get(s1_id, set())
        if not true_set:
            continue
        total_true += len(true_set)
        total_for_k += len(true_set)
        cands = set(final_cands.get(s1_id, []))
        matched_raw += len(true_set & cands)
        # For recall@K: take the top K by score from final_df for this S1
        topk_by_score = (
            final_df[final_df["s1_id"] == s1_id]
            .sort_values("score", ascending=False)
        )
        for kv in [10, 25, 50]:
            top_ids = set(topk_by_score.head(kv)["target_id"].values)
            matched_k[kv] += len(true_set & top_ids)

    raw_recall   = matched_raw / total_true if total_true else 0
    recall_at_k  = {kv: matched_k[kv] / total_for_k if total_for_k else 0
                    for kv in [10, 25, 50]}

    # Candidate stats
    cand_counts = [len(v) for v in final_cands.values()]
    arr = np.array(cand_counts) if cand_counts else np.array([0])
    total_cands = int(arr.sum())

    return {
        "raw_recall":        round(raw_recall, 4),
        "recall_at_k":       {str(k): round(v, 4) for k, v in recall_at_k.items()},
        "total_pairs_top_k": total_cands,
        "avg_raw_per_chunk": round(np.mean(chunk_raw_pair_counts), 1) if chunk_raw_pair_counts else 0,
        "max_raw_per_chunk": int(max(chunk_raw_pair_counts)) if chunk_raw_pair_counts else 0,
        "avg_cands_per_s1":  round(float(np.mean(arr)), 1),
        "max_cands_per_s1":  int(np.max(arr)),
        "p95_cands_per_s1":  round(float(np.percentile(arr, 95)), 0),
        "peak_mem_mb":       round(peak_mem_mb, 1),
        "total_time_s":      round(total_time, 2),
        "chunk_times_s":     [round(t, 2) for t in chunk_times],
        "n_chunks":          n_chunks,
        "chunk_size":        chunk_size,
    }


# ===========================================================================
# PART A: 10k Eval Set (same eval set as blocking_experiment.py)
# ===========================================================================

def run_part_a():
    print("\n" + "=" * 70)
    print("  PART A: 10k EVAL SET — 13-PASS CHUNKED vs 8-PASS BASELINE")
    print("=" * 70)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Setup connections
    mem_con  = make_mem_con(gb=6, threads=8)
    disk_con = make_disk_con(ONDI_DB_PATH_A)
    register_macros(mem_con)

    # Load ground truth
    print("\n[A1] Loading ground truth...", flush=True)
    gt_rows = mem_con.execute(f"""
    SELECT source1_entity_id, matched_entity_ids
    FROM read_csv_auto('{TRAIN_GT.replace(chr(92),"/")}', delim='\t', header=true, quote='')
    """).fetchall()

    gt_pairs = defaultdict(set)
    for s1_id, matched_str in gt_rows:
        if matched_str:
            for tid in str(matched_str).split(","):
                tid = tid.strip()
                if tid:
                    gt_pairs[s1_id].add(tid)

    s1_with_matches = [s for s, t in gt_pairs.items() if t]
    print(f"  S1 with >=1 match: {len(s1_with_matches):,}", flush=True)

    # Preprocess all sources
    print("[A2] Preprocessing all sources...", flush=True)
    t0 = time.time()
    n_s1 = load_and_preprocess(mem_con, TRAIN_S1, "s1_full")
    n_s2 = load_and_preprocess(mem_con, TRAIN_S2, "s2_full")
    n_s3 = load_and_preprocess(mem_con, TRAIN_S3, "s3_full")
    print(f"  S1={n_s1:,}  S2={n_s2:,}  S3={n_s3:,}  [{time.time()-t0:.1f}s]", flush=True)

    # Sample eval set (must match blocking_experiment.py seed)
    random.shuffle(s1_with_matches)
    eval_s1_ids = s1_with_matches[:N_EVAL_S1]
    eval_s1_set = set(eval_s1_ids)

    # True targets for eval set
    true_target_ids = set()
    for s1_id in eval_s1_ids:
        true_target_ids.update(gt_pairs[s1_id])

    # Background distractors
    all_s2 = [r[0] for r in mem_con.execute("SELECT entity_id FROM s2_full").fetchall()]
    all_s3 = [r[0] for r in mem_con.execute("SELECT entity_id FROM s3_full").fetchall()]
    bg_pool = [e for e in (all_s2 + all_s3) if e not in true_target_ids]
    bg_sample = set(random.sample(bg_pool, min(N_BACKGROUND, len(bg_pool))))
    target_pool = true_target_ids | bg_sample

    print(f"  Eval S1: {len(eval_s1_ids):,}  True targets: {len(true_target_ids):,}  "
          f"Pool: {len(target_pool):,}", flush=True)

    # Build filtered tables
    eval_df = pd.DataFrame({"entity_id": eval_s1_ids})
    pool_df = pd.DataFrame({"entity_id": list(target_pool)})
    mem_con.register("_eval_ids", eval_df)
    mem_con.register("_pool_ids", pool_df)

    mem_con.execute("CREATE OR REPLACE TABLE eval_s1_prep AS SELECT * FROM s1_full WHERE entity_id IN (SELECT entity_id FROM _eval_ids);")
    mem_con.execute("""
    CREATE OR REPLACE TABLE target_prep AS
    SELECT * FROM s2_full WHERE entity_id IN (SELECT entity_id FROM _pool_ids)
    UNION ALL
    SELECT * FROM s3_full WHERE entity_id IN (SELECT entity_id FROM _pool_ids);
    """)

    # Pre-fetch info dicts for scoring
    s1_info = {r[0]: (r[1] or "", r[2] or "")
               for r in mem_con.execute("SELECT entity_id, norm_name, norm_addr FROM eval_s1_prep").fetchall()}
    target_info = {r[0]: (r[1] or "", r[2] or "")
                   for r in mem_con.execute("SELECT entity_id, norm_name, norm_addr FROM target_prep").fetchall()}

    actual_pool = mem_con.execute("SELECT count(*) FROM target_prep").fetchone()[0]
    print(f"  Actual target_prep rows: {actual_pool:,}", flush=True)

    # Run chunked 13-pass
    print(f"\n[A3] Running chunked 13-pass (chunk_size={EVAL_CHUNK})...", flush=True)
    results_a = run_chunked_13pass(
        mem_con, disk_con,
        s1_ids=eval_s1_ids,
        s1_info=s1_info,
        target_info=target_info,
        gt_pairs=gt_pairs,
        chunk_size=EVAL_CHUNK,
        k=K,
        score_batch=SCORE_BATCH,
        label="eval",
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("  PART A RESULTS — Direct Comparison")
    print("=" * 70)
    print(f"  {'Metric':<30} {'8-pass (prior)':>18} {'13-pass chunked':>18}")
    print(f"  {'-'*66}")
    print(f"  {'Raw recall':<30} {PRIOR_8PASS['raw_recall']:>18.4f} {results_a['raw_recall']:>18.4f}")
    print(f"  {'Recall@10':<30} {'—':>18} {results_a['recall_at_k']['10']:>18.4f}")
    print(f"  {'Recall@25':<30} {PRIOR_8PASS['recall@25']:>18.4f} {results_a['recall_at_k']['25']:>18.4f}")
    print(f"  {'Recall@50':<30} {'—':>18} {results_a['recall_at_k']['50']:>18.4f}")
    print(f"  {'Avg cands/S1':<30} {PRIOR_8PASS['avg_cands']:>18.1f} {results_a['avg_cands_per_s1']:>18.1f}")
    print(f"  {'Total top-K pairs':<30} {PRIOR_8PASS['total_pairs']:>18,} {results_a['total_pairs_top_k']:>18,}")
    print(f"  {'Peak mem/chunk (MB)':<30} {'—':>18} {results_a['peak_mem_mb']:>18.1f}")
    print(f"  {'Total time (s)':<30} {PRIOR_8PASS['time_s']:>18.2f} {results_a['total_time_s']:>18.2f}")

    return results_a


# ===========================================================================
# PART B: Full-scale stress test (US partition slice)
# ===========================================================================

def run_part_b(n_stress_s1=50_000, stress_chunk=STRESS_CHUNK):
    print("\n" + "=" * 70)
    print(f"  PART B: FULL-SCALE STRESS TEST ({n_stress_s1:,} US S1 x full US targets)")
    print("=" * 70)
    random.seed(RANDOM_SEED + 1)

    mem_con  = make_mem_con(gb=6, threads=8)
    disk_con = make_disk_con(ONDI_DB_PATH_B)
    register_macros(mem_con)

    # Load S1 and filter to US country, sample n_stress_s1
    print("[B1] Loading US-country S1 entities...", flush=True)
    t0 = time.time()
    n_s1 = load_and_preprocess(mem_con, TRAIN_S1, "s1_full")
    us_s1_ids = [r[0] for r in mem_con.execute(
        "SELECT entity_id FROM s1_full WHERE country='US' LIMIT 500000").fetchall()]
    random.shuffle(us_s1_ids)
    stress_s1_ids = us_s1_ids[:n_stress_s1]
    print(f"  US S1 available: {len(us_s1_ids):,}  using: {len(stress_s1_ids):,}", flush=True)

    # Load US targets from S2 + S3
    print("[B2] Loading US-country S2+S3 targets...", flush=True)
    n_s2 = load_and_preprocess(mem_con, TRAIN_S2, "s2_full")
    n_s3 = load_and_preprocess(mem_con, TRAIN_S3, "s3_full")

    mem_con.execute("""
    CREATE OR REPLACE TABLE eval_s1_prep AS SELECT * FROM s1_full WHERE country='US';
    """)
    mem_con.execute("""
    CREATE OR REPLACE TABLE target_prep AS
    SELECT * FROM s2_full WHERE country='US'
    UNION ALL
    SELECT * FROM s3_full WHERE country='US';
    """)
    us_target_n = mem_con.execute("SELECT count(*) FROM target_prep").fetchone()[0]
    us_s1_n     = mem_con.execute("SELECT count(*) FROM eval_s1_prep").fetchone()[0]
    print(f"  US S1 in DB: {us_s1_n:,}  US targets: {us_target_n:,}  [{time.time()-t0:.1f}s]", flush=True)

    # Pre-fetch info dicts
    print("[B3] Fetching info dicts for scoring...", flush=True)
    s1_info = {r[0]: (r[1] or "", r[2] or "")
               for r in mem_con.execute("SELECT entity_id, norm_name, norm_addr FROM eval_s1_prep").fetchall()}
    target_info = {r[0]: (r[1] or "", r[2] or "")
                   for r in mem_con.execute("SELECT entity_id, norm_name, norm_addr FROM target_prep").fetchall()}

    # Filter gt_pairs is not available here (train GT not loaded) — use empty
    # We only measure timing and memory, not recall for stress test
    fake_gt = defaultdict(set)

    print(f"[B4] Running chunked 13-pass stress test "
          f"(chunk_size={stress_chunk}, {(n_stress_s1+stress_chunk-1)//stress_chunk} chunks)...", flush=True)
    results_b = run_chunked_13pass(
        mem_con, disk_con,
        s1_ids=stress_s1_ids,
        s1_info=s1_info,
        target_info=target_info,
        gt_pairs=fake_gt,
        chunk_size=stress_chunk,
        k=K,
        score_batch=SCORE_BATCH,
        label="stress",
    )

    # Extrapolate to full US (663k S1 × 3.82M targets)
    full_us_s1 = 663_000
    full_us_tgt = 3_820_000
    scale_factor = (full_us_s1 / n_stress_s1) * (full_us_tgt / us_target_n)
    est_total_time_min = results_b["total_time_s"] * (full_us_s1 / n_stress_s1) / 60.0
    est_pairs_per_chunk = results_b["avg_raw_per_chunk"] * (full_us_tgt / us_target_n)

    print("\n" + "=" * 70)
    print("  PART B RESULTS — Stress Test + Full-Scale Extrapolation")
    print("=" * 70)
    print(f"  Stress test ({n_stress_s1:,} S1 x {us_target_n:,} US targets):")
    print(f"    Avg raw pairs/chunk  : {results_b['avg_raw_per_chunk']:,.0f}")
    print(f"    Max raw pairs/chunk  : {results_b['max_raw_per_chunk']:,.0f}")
    print(f"    Peak mem/chunk (MB)  : {results_b['peak_mem_mb']:.1f}")
    print(f"    Avg cands/S1 (top-K) : {results_b['avg_cands_per_s1']:.1f}")
    print(f"    Total time           : {results_b['total_time_s']:.1f}s")
    print(f"    Time per chunk       : {results_b['chunk_times_s']}")
    print()
    print(f"  Full US partition extrapolation ({full_us_s1:,} S1 x {full_us_tgt:,} targets):")
    print(f"    Est. raw pairs/chunk : {est_pairs_per_chunk:,.0f}")
    print(f"    Est. peak mem/chunk  : {results_b['peak_mem_mb'] * (full_us_tgt/us_target_n):.0f} MB")
    print(f"    Est. total time      : {est_total_time_min:.1f} min")
    print()
    feasible = results_b["peak_mem_mb"] * (full_us_tgt / us_target_n) < 5_000
    print(f"  FEASIBLE for full US partition: {'YES' if feasible else 'NO (would exceed 5GB/chunk)'}")

    results_b["extrapolation"] = {
        "full_us_s1":              full_us_s1,
        "full_us_target":          full_us_tgt,
        "est_pairs_per_chunk":     round(est_pairs_per_chunk),
        "est_peak_mem_mb_per_chunk": round(results_b["peak_mem_mb"] * (full_us_tgt / us_target_n)),
        "est_total_time_min":      round(est_total_time_min, 1),
        "feasible":                feasible,
    }
    return results_b


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  13-PASS CHUNKED BLOCKING EXPERIMENT")
    print("  Amazon ML Challenge 2026 — Business Entity Resolution")
    print("=" * 70)

    results_a = run_part_a()
    results_b = run_part_b(n_stress_s1=50_000, stress_chunk=STRESS_CHUNK)

    # Save results
    output = {
        "prior_8pass_baseline": PRIOR_8PASS,
        "part_a_eval_10k":      results_a,
        "part_b_stress_50k_us": results_b,
        "comparison": {
            "recall_gain_at_25": round(results_a["recall_at_k"]["25"] - PRIOR_8PASS["recall@25"], 4),
            "recall_gain_raw":   round(results_a["raw_recall"] - PRIOR_8PASS["raw_recall"], 4),
            "avg_cands_ratio":   round(results_a["avg_cands_per_s1"] / PRIOR_8PASS["avg_cands"], 2),
        }
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_JSON}", flush=True)

    # Final recommendation
    print("\n" + "=" * 70)
    print("  FINAL RECOMMENDATION")
    print("=" * 70)
    recall_gain = output["comparison"]["recall_gain_at_25"]
    feasible    = results_b.get("extrapolation", {}).get("feasible", False)
    est_mem     = results_b.get("extrapolation", {}).get("est_peak_mem_mb_per_chunk", 9999)
    est_time    = results_b.get("extrapolation", {}).get("est_total_time_min", 9999)

    print(f"  Recall@25 gain over 8-pass : +{recall_gain:.4f} ({recall_gain*100:.2f}%)")
    print(f"  Full-US est. mem/chunk     : {est_mem:.0f} MB")
    print(f"  Full-US est. total time    : {est_time:.1f} min")
    print(f"  Feasible within 6GB budget : {'YES' if feasible else 'NO'}")

    if feasible:
        print("\n  DECISION: USE 13-PASS CHUNKED BLOCKING")
        print("  Rationale: provides +{:.2f}% higher recall@25 with feasible memory.".format(recall_gain*100))
    else:
        print("\n  DECISION: RETAIN 8-PASS CHUNKED BLOCKING")
        print("  Rationale: 13-pass chunked exceeds 6GB RAM per chunk for full US partition.")

    print("\n  DONE.", flush=True)
    return output


if __name__ == "__main__":
    main()
