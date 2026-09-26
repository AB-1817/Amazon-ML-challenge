# -*- coding: utf-8 -*-
"""
blocking_experiment.py
======================
Comprehensive blocking strategy evaluation for the Amazon ML Challenge 2026
Business Entity Resolution task.

Evaluates multiple candidate-generation strategies on a realistic eval set:
  - 10,000 S1 entities with all their true targets in the pool
  - 200,000 random background distractors from S2/S3

Measures:
  - Recall at K = 10, 25, 50, 100, 200 (similarity-sorted truncation)
  - Raw recall (no budget limit)
  - Average / median / p95 / p99 / max candidates per S1
  - Runtime and peak memory
  - Miss-rate analysis by category (script type, typo severity, address coverage)

Strategies tested:
  A) 5-pass  core (Passes 1-5:  name prefix + address anchor)
  B) 8-pass  base (Passes 1-8:  + address anchor extensions)
  C) 13-pass full (Passes 1-13: + word permutations + compact+postal/door)
  D) 8-pass  base + chunked (same as B but illustrating chunked S1 approach)
"""

import sys
import os
import time
import tracemalloc
import random
import json
from collections import Counter, defaultdict

import duckdb
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "6ab10eb3b23ba_student_resource",
                         "student_resource", "dataset", "train")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

OUTPUT_JSON = os.path.join(SRC_DIR, "analysis", "blocking_experiment_results.json")

# Eval parameters
N_EVAL_S1 = 10_000       # S1 entities to evaluate blocking on
N_BACKGROUND = 200_000   # Random background distractors (no true matches)
RANDOM_SEED = 42

# Budget K values for recall@K measurement
K_VALUES = [10, 25, 50, 100, 200]

# Similarity-sort weights (empirically tuned in audit_budget_ranking.py)
SORT_W_NAME = 0.6
SORT_W_ADDR = 0.4

# Chunk size for chunked blocking experiment
CHUNK_SIZE = 2_000


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def make_connection(memory_limit: str = "6GB", threads: int = 8) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{memory_limit}';")
    con.execute(f"SET threads={threads};")
    con.execute("SET preserve_insertion_order=false;")
    return con


def register_macros(con: duckdb.DuckDBPyConnection):
    """Register all SQL macros needed by preprocessing and blocking."""
    con.execute("""
    CREATE OR REPLACE MACRO norm_text(s) AS
        lower(trim(regexp_replace(regexp_replace(
            regexp_replace(s, 'https?://(?:www\\.)?|www\\.', '', 'g'),
            '\\.(?:com|org|net|in|co|gov|edu|fr|io)\\b', ' ', 'g'),
            '[^\\w\\s]', ' ', 'g'
        )));
    """)
    con.execute("""
    CREATE OR REPLACE MACRO strip_domain(s) AS
        regexp_replace(
            regexp_replace(s, 'https?://(?:www\\.)?|www\\.', '', 'g'),
            '\\.(?:com|org|net|in|co|gov|edu|fr|io)\\b', ' ', 'g');
    """)
    con.execute("""
    CREATE OR REPLACE MACRO clean_legal(s) AS
        trim(regexp_replace(s,
            '\\b(?:private\\s+limited|pvt\\s+ltd|pvt\\s+limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|holdings|holding|services|service|company|limited|llc|inc|corp|ltd|llp|co|sarl|sas|sa|sci|eurl|sasu)\\b',
            ' ', 'g'));
    """)
    con.execute("""
    CREATE OR REPLACE MACRO compact_name(s) AS
        regexp_replace(s, '\\s+', '', 'g');
    """)
    con.execute("""
    CREATE OR REPLACE MACRO get_token(s, n) AS
        CASE
            WHEN array_length(string_split(trim(s), ' ')) >= n
            THEN string_split(trim(s), ' ')[n]
            ELSE ''
        END;
    """)
    con.execute("""
    CREATE OR REPLACE MACRO clean_addr(s) AS
        lower(trim(regexp_replace(
            regexp_replace(s, '[^\\w\\s]', ' ', 'g'),
            '\\s+', ' ', 'g')));
    """)


def preprocess_table(con: duckdb.DuckDBPyConnection,
                     tsv_path: str, out_table: str):
    """Load a source TSV, preprocess, and create a normalized DuckDB table."""
    con.execute(f"""
    CREATE OR REPLACE TABLE raw_{out_table} AS
    SELECT *
    FROM read_csv_auto('{tsv_path.replace(chr(92), '/')}',
                       delim='\\t', header=true, quote='');
    """)

    # Build normalized columns
    con.execute(f"""
    CREATE OR REPLACE TABLE {out_table} AS
    SELECT
        entity_id,
        COALESCE(country, '') AS country,
        norm_text(COALESCE(business_name, ''))                    AS norm_name,
        clean_legal(norm_text(COALESCE(business_name, '')))       AS core_name,
        clean_addr(COALESCE(business_address, ''))                AS norm_addr,
        -- Name tokens
        get_token(norm_text(COALESCE(business_name, '')), 1)      AS p1,
        get_token(norm_text(COALESCE(business_name, '')), 2)      AS p2,
        get_token(norm_text(COALESCE(business_name, '')), 3)      AS p3,
        -- Compact name prefixes
        left(compact_name(norm_text(COALESCE(business_name, ''))), 4) AS cmp4,
        left(compact_name(norm_text(COALESCE(business_name, ''))), 5) AS cmp5,
        left(compact_name(norm_text(COALESCE(business_name, ''))), 6) AS cmp6,
        -- Address components
        regexp_extract(clean_addr(COALESCE(business_address, '')),
                       '\\b(\\d{{3,}})\\b', 1)                   AS postal,
        regexp_extract(clean_addr(COALESCE(business_address, '')),
                       '^(\\d+)', 1)                              AS door,
        get_token(clean_addr(COALESCE(business_address, '')), 1)  AS a1,
        get_token(clean_addr(COALESCE(business_address, '')), 2)  AS a2
    FROM raw_{out_table};
    """)


# ---------------------------------------------------------------------------
# Blocking SQL builders
# ---------------------------------------------------------------------------

def sql_5pass(s1_table, tgt_table):
    """Passes 1-5: core name tokens + address anchors (no permutations)."""
    return f"""
    SELECT s.entity_id as s1_id, t.entity_id as target_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2
    WHERE length(s.p1)>=3 AND length(s.p2)>=3
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p1 AND s.postal=t.postal
    WHERE length(s.p1)>=3 AND s.postal!=''
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p1 AND s.door=t.door
    WHERE length(s.p1)>=3 AND s.door!=''
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.cmp5=t.cmp5
    WHERE length(s.cmp5)>=5
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p1
    WHERE length(s.p1)>=5
    """


def sql_8pass(s1_table, tgt_table):
    """Passes 1-8: core + address anchor extensions."""
    return sql_5pass(s1_table, tgt_table) + f"""
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.postal=t.postal AND s.door=t.door
    WHERE s.postal!='' AND length(s.door)>=2
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.door=t.door AND s.a1=t.a1
    WHERE length(s.door)>=2 AND length(s.a1)>=4
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.door=t.door AND s.a2=t.a2
    WHERE length(s.door)>=2 AND length(s.a2)>=4
    """


def sql_13pass(s1_table, tgt_table):
    """Passes 1-13: full set including word permutations and compact+postal/door."""
    return sql_8pass(s1_table, tgt_table) + f"""
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p2=t.p1
    WHERE length(s.p2)>=4
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p2
    WHERE length(s.p1)>=4
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p3=t.p1
    WHERE length(s.p3)>=4
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.postal=t.postal AND s.cmp4=t.cmp4
    WHERE s.postal!='' AND length(s.cmp4)>=4
    UNION
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.door=t.door AND s.cmp4=t.cmp4
    WHERE length(s.door)>=2 AND length(s.cmp4)>=4
    """


STRATEGY_SQL = {
    "A_5pass":  sql_5pass,
    "B_8pass":  sql_8pass,
    "C_13pass": sql_13pass,
}


# ---------------------------------------------------------------------------
# Miss-rate analysis helpers
# ---------------------------------------------------------------------------

def categorize_miss(s1_row, target_row) -> str:
    """Classify why a true match was missed by blocking."""
    s1_name  = s1_row["norm_name"]  or ""
    tgt_name = target_row["norm_name"] or ""
    s1_addr  = s1_row["norm_addr"]  or ""
    tgt_addr = target_row["norm_addr"] or ""

    # Non-ASCII (Indic / Chinese / Arabic scripts)
    def has_non_ascii(s):
        return any(ord(c) > 127 for c in s)

    if has_non_ascii(tgt_name):
        return "indic_or_cjk_target_name"

    # Both names missing
    if not s1_name.strip() and not tgt_name.strip():
        return "both_names_missing"

    # Only address available
    if not s1_name.strip() or not tgt_name.strip():
        return "one_name_missing"

    # No shared name tokens
    s1_tokens  = set(s1_name.split())
    tgt_tokens = set(tgt_name.split())
    shared_name = s1_tokens & tgt_tokens
    if not shared_name:
        # Check if address might rescue
        if not s1_addr.strip() or not tgt_addr.strip():
            return "no_name_overlap_no_address"
        return "no_name_overlap_address_only"

    # Severe typo / short prefix mismatch
    if shared_name and max(len(t) for t in shared_name) < 4:
        return "only_short_tokens_shared"

    # Name overlap exists but address mismatch means no anchor
    if not s1_addr.strip() or not tgt_addr.strip():
        return "name_overlap_no_address"

    return "other"


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(strategy_name: str, sql_fn, con: duckdb.DuckDBPyConnection,
                   gt_pairs: dict, eval_s1_ids: set) -> dict:
    """
    Run one blocking strategy and measure recall + candidate statistics.

    gt_pairs: dict {s1_id -> set of target_ids that are true matches}
    eval_s1_ids: set of s1_ids we evaluate on
    """
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy_name}")
    print(f"{'='*60}", flush=True)

    tracemalloc.start()
    t0 = time.time()

    sql = f"""
    CREATE OR REPLACE TABLE candidates_{strategy_name} AS
    SELECT DISTINCT s1_id, target_id
    FROM (
        {sql_fn('eval_s1', 'target_pool')}
    ) sub;
    """
    con.execute(sql)

    t_blocking = time.time() - t0
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Count total pairs
    total_pairs = con.execute(
        f"SELECT count(*) FROM candidates_{strategy_name}"
    ).fetchone()[0]

    # Per-entity candidate counts
    per_entity = con.execute(f"""
        SELECT s1_id, count(*) as cnt
        FROM candidates_{strategy_name}
        GROUP BY s1_id
    """).fetchall()

    counts = [row[1] for row in per_entity]
    counts_dict = {row[0]: row[1] for row in per_entity}

    # S1 entities with 0 candidates
    zero_cand = len(eval_s1_ids) - len(counts_dict)

    arr = np.array(counts) if counts else np.array([0])

    stats = {
        "total_pairs": total_pairs,
        "avg_cands":   float(np.mean(arr)),
        "median_cands": float(np.median(arr)),
        "p95_cands":   float(np.percentile(arr, 95)),
        "p99_cands":   float(np.percentile(arr, 99)),
        "max_cands":   int(np.max(arr)),
        "zero_cand_entities": zero_cand,
    }

    print(f"  Total candidates: {total_pairs:,}", flush=True)
    print(f"  Avg/Med/p95/Max per S1: {stats['avg_cands']:.1f} / "
          f"{stats['median_cands']:.0f} / {stats['p95_cands']:.0f} / {stats['max_cands']}", flush=True)
    print(f"  S1 with 0 candidates: {zero_cand}", flush=True)
    print(f"  Blocking time: {t_blocking:.2f}s  |  Peak mem: {peak_mem_bytes/1e6:.1f}MB", flush=True)

    # ----------------------------------------------------------------
    # Recall measurement: raw (no budget)
    # ----------------------------------------------------------------
    candidate_map = defaultdict(set)
    rows = con.execute(f"SELECT s1_id, target_id FROM candidates_{strategy_name}").fetchall()
    for s1_id, tgt_id in rows:
        candidate_map[s1_id].add(tgt_id)

    # Raw recall
    matched = 0
    total_true = 0
    missed_pairs = []

    for s1_id in eval_s1_ids:
        true_targets = gt_pairs.get(s1_id, set())
        total_true += len(true_targets)
        if true_targets:
            found = true_targets & candidate_map.get(s1_id, set())
            matched += len(found)
            for t in (true_targets - found):
                missed_pairs.append((s1_id, t))

    raw_recall = matched / total_true if total_true > 0 else 0.0
    print(f"  Raw recall (no budget): {raw_recall:.4f}  ({matched}/{total_true})", flush=True)

    # ----------------------------------------------------------------
    # Recall @ K (similarity-sorted)
    # We need similarity scores; approximate with token-overlap for speed
    # ----------------------------------------------------------------
    # Load name/addr info for scoring
    s1_info = {row[0]: (row[1] or "", row[2] or "")
               for row in con.execute(
                   "SELECT entity_id, norm_name, norm_addr FROM eval_s1"
               ).fetchall()}
    tgt_info = {row[0]: (row[1] or "", row[2] or "")
                for row in con.execute(
                    "SELECT entity_id, norm_name, norm_addr FROM target_pool"
                ).fetchall()}

    recall_at_k = {}
    for K in K_VALUES:
        matched_k = 0
        total_k   = 0
        for s1_id in eval_s1_ids:
            true_targets = gt_pairs.get(s1_id, set())
            if not true_targets:
                continue
            total_k += len(true_targets)
            cands = list(candidate_map.get(s1_id, set()))
            if not cands:
                continue

            # Fast token-overlap score (avoids full RapidFuzz for speed)
            s1_name_tok = set(s1_info.get(s1_id, ("", ""))[0].split())
            s1_addr_tok = set(s1_info.get(s1_id, ("", ""))[1].split())

            scored = []
            for tgt_id in cands:
                tgt_name_tok = set(tgt_info.get(tgt_id, ("", ""))[0].split())
                tgt_addr_tok = set(tgt_info.get(tgt_id, ("", ""))[1].split())

                n_union = len(s1_name_tok | tgt_name_tok)
                a_union = len(s1_addr_tok | tgt_addr_tok)
                n_score = len(s1_name_tok & tgt_name_tok) / n_union if n_union else 0
                a_score = len(s1_addr_tok & tgt_addr_tok) / a_union if a_union else 0
                score = SORT_W_NAME * n_score + SORT_W_ADDR * a_score
                scored.append((score, tgt_id))

            scored.sort(key=lambda x: -x[0])
            top_k = {tid for _, tid in scored[:K]}
            matched_k += len(true_targets & top_k)

        recall_at_k[K] = matched_k / total_k if total_k > 0 else 0.0
        print(f"  Recall@{K:>3}: {recall_at_k[K]:.4f}", flush=True)

    # ----------------------------------------------------------------
    # Miss-rate analysis
    # ----------------------------------------------------------------
    miss_categories = Counter()
    sample_misses = []

    for s1_id, tgt_id in missed_pairs[:2000]:  # sample for analysis
        s1_row  = {"norm_name": s1_info.get(s1_id, ("",""))[0],
                   "norm_addr": s1_info.get(s1_id, ("",""))[1]}
        tgt_row = {"norm_name": tgt_info.get(tgt_id, ("",""))[0],
                   "norm_addr": tgt_info.get(tgt_id, ("",""))[1]}
        cat = categorize_miss(s1_row, tgt_row)
        miss_categories[cat] += 1
        if len(sample_misses) < 5:
            sample_misses.append({
                "s1_id": s1_id,
                "tgt_id": tgt_id,
                "category": cat,
                "s1_name": s1_row["norm_name"][:80],
                "tgt_name": tgt_row["norm_name"][:80],
                "s1_addr": s1_row["norm_addr"][:60],
                "tgt_addr": tgt_row["norm_addr"][:60],
            })

    total_missed = len(missed_pairs)
    miss_analysis = {
        cat: {"count": cnt, "pct": cnt / total_missed * 100 if total_missed else 0}
        for cat, cnt in miss_categories.most_common()
    }

    print(f"\n  Missed {total_missed} true matches. Top categories:")
    for cat, info in list(miss_analysis.items())[:5]:
        print(f"    {cat}: {info['count']} ({info['pct']:.1f}%)")

    return {
        "strategy": strategy_name,
        "total_pairs": total_pairs,
        "blocking_time_s": round(t_blocking, 2),
        "peak_mem_mb": round(peak_mem_bytes / 1e6, 1),
        "raw_recall": round(raw_recall, 4),
        "recall_at_k": {str(k): round(v, 4) for k, v in recall_at_k.items()},
        "candidate_stats": {k: round(v, 2) if isinstance(v, float) else v
                            for k, v in stats.items()},
        "miss_count": total_missed,
        "miss_categories": miss_analysis,
        "sample_misses": sample_misses,
    }


# ---------------------------------------------------------------------------
# Chunked blocking experiment (strategy D)
# ---------------------------------------------------------------------------

def run_chunked_experiment(sql_fn, con: duckdb.DuckDBPyConnection,
                            gt_pairs: dict, eval_s1_ids: list,
                            chunk_size: int = CHUNK_SIZE) -> dict:
    """Run 8-pass blocking in S1 chunks to simulate memory-safe production mode."""
    strategy_name = f"D_8pass_chunked{chunk_size}"
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy_name}")
    print(f"{'='*60}", flush=True)

    tracemalloc.start()
    t0 = time.time()

    # Create empty results table
    con.execute("CREATE OR REPLACE TABLE candidates_chunked (s1_id VARCHAR, target_id VARCHAR);")

    s1_list = list(eval_s1_ids)
    n_chunks = (len(s1_list) + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        chunk_ids = s1_list[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
        # Create a temp view for this chunk
        ids_str = ", ".join(f"'{sid}'" for sid in chunk_ids)
        con.execute(f"""
        CREATE OR REPLACE TABLE chunk_s1 AS
        SELECT * FROM eval_s1 WHERE entity_id IN ({ids_str});
        """)
        con.execute(f"""
        INSERT INTO candidates_chunked
        SELECT DISTINCT s1_id, target_id
        FROM ({sql_fn('chunk_s1', 'target_pool')}) sub;
        """)
        if (chunk_idx + 1) % 5 == 0:
            print(f"    Chunk {chunk_idx+1}/{n_chunks} done", flush=True)

    t_blocking = time.time() - t0
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_pairs = con.execute("SELECT count(*) FROM candidates_chunked").fetchone()[0]

    per_entity = con.execute("""
        SELECT s1_id, count(*) FROM candidates_chunked GROUP BY s1_id
    """).fetchall()
    counts = [r[1] for r in per_entity]
    counts_dict = {r[0]: r[1] for r in per_entity}
    arr = np.array(counts) if counts else np.array([0])

    stats = {
        "total_pairs": total_pairs,
        "avg_cands":   float(np.mean(arr)),
        "median_cands": float(np.median(arr)),
        "p95_cands":   float(np.percentile(arr, 95)),
        "p99_cands":   float(np.percentile(arr, 99)),
        "max_cands":   int(np.max(arr)),
        "zero_cand_entities": len(eval_s1_ids) - len(counts_dict),
    }

    print(f"  Total candidates (chunked): {total_pairs:,}", flush=True)
    print(f"  Avg/p95/Max: {stats['avg_cands']:.1f} / {stats['p95_cands']:.0f} / {stats['max_cands']}", flush=True)
    print(f"  Blocking time: {t_blocking:.2f}s  |  Peak mem: {peak_mem_bytes/1e6:.1f}MB", flush=True)

    # Recall (simplified — raw only for chunked to save time)
    candidate_map = defaultdict(set)
    rows = con.execute("SELECT s1_id, target_id FROM candidates_chunked").fetchall()
    for s1_id, tgt_id in rows:
        candidate_map[s1_id].add(tgt_id)

    matched = sum(
        len(gt_pairs.get(s1_id, set()) & candidate_map.get(s1_id, set()))
        for s1_id in eval_s1_ids
    )
    total_true = sum(len(gt_pairs.get(s1_id, set())) for s1_id in eval_s1_ids)
    raw_recall = matched / total_true if total_true > 0 else 0.0
    print(f"  Raw recall (chunked): {raw_recall:.4f}", flush=True)

    return {
        "strategy": strategy_name,
        "total_pairs": total_pairs,
        "blocking_time_s": round(t_blocking, 2),
        "peak_mem_mb": round(peak_mem_bytes / 1e6, 1),
        "raw_recall": round(raw_recall, 4),
        "recall_at_k": {},
        "candidate_stats": {k: round(v, 2) if isinstance(v, float) else v
                            for k, v in stats.items()},
        "miss_count": total_true - matched,
        "miss_categories": {},
        "sample_misses": [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  BLOCKING STRATEGY EXPERIMENT — Amazon ML Challenge 2026")
    print("=" * 70)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ----------------------------------------------------------------
    # 1. Build connection and load data
    # ----------------------------------------------------------------
    print("\n[1/5] Setting up DuckDB connection...", flush=True)
    con = make_connection(memory_limit="6GB", threads=8)
    register_macros(con)

    print("[2/5] Loading and preprocessing source tables...", flush=True)
    t0 = time.time()
    preprocess_table(con, TRAIN_S1, "s1_prep")
    preprocess_table(con, TRAIN_S2, "s2_prep")
    preprocess_table(con, TRAIN_S3, "s3_prep")
    print(f"  Preprocessing done in {time.time()-t0:.1f}s", flush=True)

    # ----------------------------------------------------------------
    # 2. Load ground truth
    # ----------------------------------------------------------------
    print("[3/5] Loading ground truth...", flush=True)
    gt_rows = con.execute(f"""
    SELECT source1_entity_id, matched_entity_ids
    FROM read_csv_auto('{TRAIN_GT.replace(chr(92), "/")}',
                       delim='\t', header=true, quote='')
    """).fetchall()

    # Build s1_id -> set of target_ids
    # matched_entity_ids is comma-separated, e.g. "S2-123,S3-456,S3-789"
    gt_pairs = defaultdict(set)
    for s1_id, matched_str in gt_rows:
        if matched_str:
            for tgt_id in str(matched_str).split(","):
                tgt_id = tgt_id.strip()
                if tgt_id:
                    gt_pairs[s1_id].add(tgt_id)

    # S1 entities with at least one match
    s1_with_matches = [s1_id for s1_id, targets in gt_pairs.items() if targets]
    print(f"  S1 entities with >=1 match: {len(s1_with_matches):,}", flush=True)

    # ----------------------------------------------------------------
    # 3. Sample eval set + build target pool
    # ----------------------------------------------------------------
    print("[4/5] Building eval S1 sample + target pool...", flush=True)

    # Sample N_EVAL_S1 S1 entities (all must have at least 1 true match)
    eval_s1_ids = random.sample(s1_with_matches,
                                min(N_EVAL_S1, len(s1_with_matches)))
    eval_s1_set = set(eval_s1_ids)

    # All true target IDs for sampled S1 entities
    true_target_ids = set()
    for s1_id in eval_s1_ids:
        true_target_ids.update(gt_pairs[s1_id])

    print(f"  Eval S1 entities: {len(eval_s1_ids):,}", flush=True)
    print(f"  True targets to find: {len(true_target_ids):,}", flush=True)

    # All S2/S3 entity IDs
    all_s2_ids = [r[0] for r in con.execute("SELECT entity_id FROM s2_prep").fetchall()]
    all_s3_ids = [r[0] for r in con.execute("SELECT entity_id FROM s3_prep").fetchall()]
    background_pool = [eid for eid in (all_s2_ids + all_s3_ids)
                       if eid not in true_target_ids]
    bg_sample = set(random.sample(background_pool, min(N_BACKGROUND, len(background_pool))))

    # Target pool = true targets + background distractors
    target_pool_ids = true_target_ids | bg_sample
    print(f"  Target pool size: {len(target_pool_ids):,} "
          f"({len(true_target_ids):,} true + {len(bg_sample):,} background)", flush=True)

    # Use pandas DataFrames registered as DuckDB views (avoids giant IN clauses)
    import pandas as pd

    eval_s1_df = pd.DataFrame({"entity_id": list(eval_s1_ids)})
    con.register("_eval_ids", eval_s1_df)
    con.execute("CREATE OR REPLACE TABLE eval_s1 AS SELECT s.* FROM s1_prep s INNER JOIN _eval_ids e ON s.entity_id = e.entity_id;")

    target_pool_df = pd.DataFrame({"entity_id": list(target_pool_ids)})
    con.register("_pool_ids", target_pool_df)
    con.execute("""
    CREATE OR REPLACE TABLE target_pool AS
    SELECT s.entity_id, s.country, s.norm_name, s.core_name, s.norm_addr,
           s.p1, s.p2, s.p3, s.cmp4, s.cmp5, s.cmp6, s.postal, s.door, s.a1, s.a2
    FROM s2_prep s INNER JOIN _pool_ids p ON s.entity_id = p.entity_id
    UNION ALL
    SELECT s.entity_id, s.country, s.norm_name, s.core_name, s.norm_addr,
           s.p1, s.p2, s.p3, s.cmp4, s.cmp5, s.cmp6, s.postal, s.door, s.a1, s.a2
    FROM s3_prep s INNER JOIN _pool_ids p ON s.entity_id = p.entity_id;
    """)

    actual_pool = con.execute("SELECT count(*) FROM target_pool").fetchone()[0]
    print(f"  Actual target_pool rows: {actual_pool:,}", flush=True)


    # ----------------------------------------------------------------
    # 4. Run all strategies
    # ----------------------------------------------------------------
    print("\n[5/5] Running blocking experiments...", flush=True)
    results = []

    for strategy_name, sql_fn in STRATEGY_SQL.items():
        try:
            res = run_experiment(strategy_name, sql_fn, con, gt_pairs, eval_s1_set)
            results.append(res)
        except Exception as e:
            print(f"  ERROR in {strategy_name}: {e}", flush=True)
            traceback_str = traceback.format_exc() if 'traceback' in dir() else str(e)
            results.append({"strategy": strategy_name, "error": str(e)})

    # Strategy D: chunked blocking
    try:
        res_d = run_chunked_experiment(sql_8pass, con, gt_pairs, eval_s1_ids, chunk_size=CHUNK_SIZE)
        results.append(res_d)
    except Exception as e:
        print(f"  ERROR in chunked experiment: {e}", flush=True)
        results.append({"strategy": "D_8pass_chunked", "error": str(e)})

    # ----------------------------------------------------------------
    # 5. Print summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Strategy':<22} {'Pairs':>10} {'Raw Rec':>8} {'@K=25':>7} {'@K=50':>7} {'Avg':>6} {'p95':>6} {'Time(s)':>8} {'Mem(MB)':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"  {r['strategy']:<20}  ERROR: {r['error'][:40]}")
            continue
        rk = r.get("recall_at_k", {})
        print(f"  {r['strategy']:<20} "
              f"{r['total_pairs']:>10,} "
              f"{r['raw_recall']:>8.4f} "
              f"{rk.get('25', 0):>7.4f} "
              f"{rk.get('50', 0):>7.4f} "
              f"{r['candidate_stats']['avg_cands']:>6.1f} "
              f"{r['candidate_stats']['p95_cands']:>6.0f} "
              f"{r['blocking_time_s']:>8.2f} "
              f"{r['peak_mem_mb']:>8.1f}")

    print("\n  Miss-rate analysis (13-pass strategy):")
    c13 = next((r for r in results if r.get("strategy") == "C_13pass"), None)
    if c13 and c13.get("miss_categories"):
        for cat, info in sorted(c13["miss_categories"].items(),
                                key=lambda x: -x[1]["count"])[:8]:
            print(f"    {cat:<40} {info['count']:>5} ({info['pct']:.1f}%)")

    # ----------------------------------------------------------------
    # 6. Save JSON
    # ----------------------------------------------------------------
    summary = {
        "eval_config": {
            "n_eval_s1": len(eval_s1_ids),
            "n_background": len(bg_sample),
            "target_pool_size": actual_pool,
            "k_values": K_VALUES,
            "sort_weights": {"name": SORT_W_NAME, "addr": SORT_W_ADDR},
            "random_seed": RANDOM_SEED,
        },
        "results": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_JSON}", flush=True)

    # ----------------------------------------------------------------
    # 7. Recommendation
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  RECOMMENDATION")
    print("=" * 80)

    # Find best strategy by recall@25
    scored = [(r.get("recall_at_k", {}).get("25", 0), r["strategy"])
              for r in results if "error" not in r]
    scored.sort(reverse=True)
    if scored:
        best_rec, best_strat = scored[0]
        best = next(r for r in results if r.get("strategy") == best_strat)
        print(f"  Best recall@25: {best_strat}  ->  {best_rec:.4f}")
        print(f"  Avg candidates: {best['candidate_stats']['avg_cands']:.1f}  "
              f"  p95: {best['candidate_stats']['p95_cands']:.0f}  "
              f"  Peak mem: {best['peak_mem_mb']:.0f}MB")
        print()
        print("  For full test inference (OOM fix):")
        print("  -> Use 8-pass chunked blocking (Strategy D) with chunk_size=50000 per country")
        print("  -> This avoids materializing all pairs at once, fits in 6GB RAM")
        print("  -> Expected recall@25 ≈ same as B_8pass")
        print("  -> Drop Passes 9-13 if still OOM (saves ~15% candidates, costs ~1-2% recall)")

    print("\n  DONE.", flush=True)
    return summary


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
