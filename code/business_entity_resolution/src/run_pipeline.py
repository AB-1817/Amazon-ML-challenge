# -*- coding: utf-8 -*-
"""
run_pipeline.py
===============
End-to-end entity resolution pipeline for the Amazon ML Challenge 2026.

OOM-SAFE ARCHITECTURE (v3):
  - Uses 8-pass blocking only (passes 1-8, no word permutations)
  - Processes S1 in chunks of S1_CHUNK_SIZE per country partition
  - Similarity-sorts candidates BEFORE truncation to K=25
  - Never materializes more than ~500MB of pairs at once

Empirical benchmarks (from blocking_experiment.py):
  - 8-pass recall@25 = 80.64%  (vs 84.40% for 13-pass)
  - 8-pass avg cands = 143.7   (vs 630.7 for 13-pass)
  - 8-pass is 4.4x more memory efficient
"""

import os
import sys
import time
import argparse
import duckdb
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH,
    STUDENT_RESOURCE_DIR, MAX_CANDIDATES_PER_ENTITY, DEFAULT_DECISION_THRESHOLD, NUM_WORKERS
)
from src.features import compute_pair_features, FEATURE_NAMES
from src.model import load_matching_model

# Chunk size for S1 entities per blocking round (memory-safe)
S1_CHUNK_SIZE = 50_000
# Scoring chunk size (RapidFuzz rows per batch)
SCORE_CHUNK_SIZE = 50_000
# Similarity ranking weights (empirically tuned)
W_NAME = 0.6
W_ADDR = 0.4


# ---------------------------------------------------------------------------
# DuckDB setup
# ---------------------------------------------------------------------------

def make_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("SET memory_limit='6GB';")
    con.execute(f"SET threads={NUM_WORKERS};")
    con.execute("SET preserve_insertion_order=false;")
    return con


def register_macros(con: duckdb.DuckDBPyConnection):
    """Register text cleaning macros. Must be called once per connection."""
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


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

PREPROCESS_SQL = """
SELECT
    entity_id,
    COALESCE(country, '')                                              AS country,
    norm_text(COALESCE(business_name,''))                              AS norm_name,
    clean_legal(COALESCE(business_name,''))                            AS core_name,
    clean_addr(COALESCE(business_address,''))                          AS norm_addr,
    get_token(norm_text(COALESCE(business_name,'')), 1)                AS p1,
    get_token(norm_text(COALESCE(business_name,'')), 2)                AS p2,
    get_token(norm_text(COALESCE(business_name,'')), 3)                AS p3,
    left(compact_name(COALESCE(business_name,'')), 4)                  AS cmp4,
    left(compact_name(COALESCE(business_name,'')), 5)                  AS cmp5,
    left(compact_name(COALESCE(business_name,'')), 6)                  AS cmp6,
    regexp_extract(clean_addr(COALESCE(business_address,'')),
                   '\\b(\\d{3,})\\b', 1)                              AS postal,
    regexp_extract(clean_addr(COALESCE(business_address,'')),
                   '^(\\d+)', 1)                                       AS door,
    get_token(clean_addr(COALESCE(business_address,'')), 1)            AS a1,
    get_token(clean_addr(COALESCE(business_address,'')), 2)            AS a2
FROM src_raw
WHERE COALESCE(country,'') = '{country}'
"""


def load_country_tables(con: duckdb.DuckDBPyConnection,
                        s1_path: str, s2_path: str, s3_path: str,
                        country: str):
    """Load and preprocess S1 + target (S2+S3) for one country into DuckDB tables."""
    # Load S1
    con.execute(f"""
    CREATE OR REPLACE TABLE src_raw AS
    SELECT * FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
    WHERE COALESCE(country,'') = '{country}';
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE s1_country AS {PREPROCESS_SQL.format(country=country)};
    """)
    s1_count = con.execute("SELECT count(*) FROM s1_country").fetchone()[0]

    # Load S2
    con.execute(f"""
    CREATE OR REPLACE TABLE src_raw AS
    SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true)
    WHERE COALESCE(country,'') = '{country}';
    """)
    con.execute(f"CREATE OR REPLACE TABLE s2_country AS {PREPROCESS_SQL.format(country=country)};")

    # Load S3
    con.execute(f"""
    CREATE OR REPLACE TABLE src_raw AS
    SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true)
    WHERE COALESCE(country,'') = '{country}';
    """)
    con.execute(f"CREATE OR REPLACE TABLE s3_country AS {PREPROCESS_SQL.format(country=country)};")

    # Combine into target table
    con.execute("""
    CREATE OR REPLACE TABLE target_country AS
    SELECT * FROM s2_country
    UNION ALL
    SELECT * FROM s3_country;
    """)
    con.execute("DROP TABLE IF EXISTS s2_country; DROP TABLE IF EXISTS s3_country; DROP TABLE IF EXISTS src_raw;")

    target_count = con.execute("SELECT count(*) FROM target_country").fetchone()[0]
    return s1_count, target_count


# ---------------------------------------------------------------------------
# 8-pass blocking SQL (OOM-safe, no word permutations)
# ---------------------------------------------------------------------------

def blocking_8pass_sql(s1_table: str, tgt_table: str) -> str:
    """Return UNION SQL for 8-pass OOM-safe blocking."""
    return f"""
    -- Pass 1: p1 + p2 match
    SELECT s.entity_id AS s1_id, t.entity_id AS target_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2
    WHERE length(s.p1)>=3 AND length(s.p2)>=3
    UNION
    -- Pass 2: p1 + postal
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p1 AND s.postal=t.postal
    WHERE length(s.p1)>=3 AND s.postal!=''
    UNION
    -- Pass 3: p1 + door
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p1 AND s.door=t.door
    WHERE length(s.p1)>=3 AND s.door!=''
    UNION
    -- Pass 4: cmp5 prefix
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.cmp5=t.cmp5
    WHERE length(s.cmp5)>=5
    UNION
    -- Pass 5: long p1 (>=5 chars)
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.p1=t.p1
    WHERE length(s.p1)>=5
    UNION
    -- Pass 6: door + postal (same building)
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.postal=t.postal AND s.door=t.door
    WHERE s.postal!='' AND length(s.door)>=2
    UNION
    -- Pass 7: door + street token 1
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.door=t.door AND s.a1=t.a1
    WHERE length(s.door)>=2 AND length(s.a1)>=4
    UNION
    -- Pass 8: door + street token 2
    SELECT s.entity_id, t.entity_id
    FROM {s1_table} s JOIN {tgt_table} t
      ON s.country=t.country AND s.door=t.door AND s.a2=t.a2
    WHERE length(s.door)>=2 AND length(s.a2)>=4
    """


# ---------------------------------------------------------------------------
# Chunked blocking + similarity-sorted K pruning
# ---------------------------------------------------------------------------

def run_chunked_blocking(con: duckdb.DuckDBPyConnection,
                         s1_ids: List[str],
                         chunk_size: int = S1_CHUNK_SIZE,
                         k: int = MAX_CANDIDATES_PER_ENTITY
                         ) -> Dict[str, List[str]]:
    """
    Run 8-pass blocking in S1 chunks and return top-K similarity-sorted
    candidates per S1 entity.

    Returns: dict {s1_id -> [target_id, ...]} (length <= K)
    """
    # Pre-load target info for scoring
    target_info = {
        row[0]: (row[1] or "", row[2] or "")
        for row in con.execute(
            "SELECT entity_id, norm_name, norm_addr FROM target_country"
        ).fetchall()
    }
    s1_info = {
        row[0]: (row[1] or "", row[2] or "")
        for row in con.execute(
            "SELECT entity_id, norm_name, norm_addr FROM s1_country"
        ).fetchall()
    }

    n_chunks = (len(s1_ids) + chunk_size - 1) // chunk_size
    print(f"  Chunked 8-pass blocking: {len(s1_ids):,} S1 in {n_chunks} chunk(s) of {chunk_size:,}", flush=True)

    # Accumulate per-entity candidate sets
    candidate_map: Dict[str, set] = defaultdict(set)

    for chunk_idx in range(n_chunks):
        t_chunk = time.time()
        chunk_ids = s1_ids[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]

        # Register chunk as DuckDB table via pandas
        chunk_df = pd.DataFrame({"entity_id": chunk_ids})
        con.register("_chunk_ids", chunk_df)
        con.execute("""
        CREATE OR REPLACE TABLE chunk_s1 AS
        SELECT * FROM s1_country
        WHERE entity_id IN (SELECT entity_id FROM _chunk_ids);
        """)

        # Run 8-pass blocking for this chunk
        sql = f"""
        SELECT DISTINCT s1_id, target_id FROM (
            {blocking_8pass_sql('chunk_s1', 'target_country')}
        ) sub;
        """
        rows = con.execute(sql).fetchall()
        for s1_id, tgt_id in rows:
            candidate_map[s1_id].add(tgt_id)

        elapsed = time.time() - t_chunk
        if (chunk_idx + 1) % 5 == 0 or chunk_idx == 0 or chunk_idx == n_chunks - 1:
            print(f"  Chunk {chunk_idx+1}/{n_chunks}: {len(rows):,} pairs in {elapsed:.2f}s", flush=True)

    con.execute("DROP TABLE IF EXISTS chunk_s1;")

    # Similarity-sort candidates and prune to top K
    print(f"  Similarity-sorting and pruning to top K={k}...", flush=True)
    result: Dict[str, List[str]] = {}

    for s1_id, cand_set in candidate_map.items():
        if not cand_set:
            result[s1_id] = []
            continue

        s1_name_tok = set((s1_info.get(s1_id, ("",""))[0] or "").split())
        s1_addr_tok = set((s1_info.get(s1_id, ("",""))[1] or "").split())

        scored = []
        for tgt_id in cand_set:
            tgt_name_tok = set((target_info.get(tgt_id, ("",""))[0] or "").split())
            tgt_addr_tok = set((target_info.get(tgt_id, ("",""))[1] or "").split())

            n_union = len(s1_name_tok | tgt_name_tok)
            a_union = len(s1_addr_tok | tgt_addr_tok)
            n_score = len(s1_name_tok & tgt_name_tok) / n_union if n_union else 0.0
            a_score = len(s1_addr_tok & tgt_addr_tok) / a_union if a_union else 0.0
            score = W_NAME * n_score + W_ADDR * a_score
            scored.append((score, tgt_id))

        scored.sort(key=lambda x: -x[0])
        result[s1_id] = [tid for _, tid in scored[:k]]

    # Singletons with no candidates
    for s1_id in s1_ids:
        if s1_id not in result:
            result[s1_id] = []

    total_cands = sum(len(v) for v in result.values())
    print(f"  Total candidates after K-pruning: {total_cands:,}  avg/S1={total_cands/len(s1_ids):.1f}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: Dict[str, List[str]],
    s1_info: Dict[str, tuple],
    target_info: Dict[str, tuple],
    model,
    threshold: float,
    chunk_size: int = SCORE_CHUNK_SIZE
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute RapidFuzz features for all candidate pairs and score with XGBoost.
    Returns (candidate_dict, matched_dict).
    """
    # Flatten to list of pairs
    pairs = [(s1_id, tgt_id)
             for s1_id, tgt_ids in candidates.items()
             for tgt_id in tgt_ids]

    total = len(pairs)
    print(f"  Scoring {total:,} pairs with XGBoost (chunk_size={chunk_size:,})...", flush=True)

    matched_dict: Dict[str, List[str]] = defaultdict(list)

    for i in range(0, total, chunk_size):
        chunk = pairs[i: i + chunk_size]
        n = len(chunk)
        X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)

        for j, (s1_id, tgt_id) in enumerate(chunk):
            s1_name, s1_addr = s1_info.get(s1_id, ("", ""))
            t_name,  t_addr  = target_info.get(tgt_id, ("", ""))
            X[j] = compute_pair_features(
                s1_name, s1_name, s1_addr, "", "",
                t_name,  t_name,  t_addr,  "", ""
            )

        probs = model.predict_proba(X)[:, 1]

        for (s1_id, tgt_id), prob in zip(chunk, probs):
            if prob >= threshold:
                matched_dict[s1_id].append(tgt_id)

        if (i // chunk_size + 1) % 20 == 0:
            print(f"    Scored {min(i+chunk_size, total):,}/{total:,}", flush=True)

    return candidates, matched_dict


# ---------------------------------------------------------------------------
# Country partition processor
# ---------------------------------------------------------------------------

def process_country_partition(
    con: duckdb.DuckDBPyConnection,
    model,
    country: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Process one country: load -> block -> score -> return dicts."""
    print(f"\n{'='*60}", flush=True)
    print(f"  COUNTRY: {country.upper()}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()

    # Load + preprocess
    s1_count, tgt_count = load_country_tables(con, s1_path, s2_path, s3_path, country)
    print(f"  S1={s1_count:,}  Targets={tgt_count:,}", flush=True)

    if s1_count == 0:
        print(f"  No S1 entities for {country}, skipping.", flush=True)
        return {}, {}

    # Fetch all S1 IDs
    s1_ids = [r[0] for r in con.execute("SELECT entity_id FROM s1_country").fetchall()]

    # Fetch lookup dicts for scoring
    s1_info = {r[0]: (r[1] or "", r[2] or "")
               for r in con.execute("SELECT entity_id, norm_name, norm_addr FROM s1_country").fetchall()}
    target_info = {r[0]: (r[1] or "", r[2] or "")
                   for r in con.execute("SELECT entity_id, norm_name, norm_addr FROM target_country").fetchall()}

    # Chunked 8-pass blocking + similarity-sorted K-pruning
    candidate_dict = run_chunked_blocking(con, s1_ids,
                                          chunk_size=S1_CHUNK_SIZE,
                                          k=MAX_CANDIDATES_PER_ENTITY)

    # XGBoost scoring
    candidate_dict, matched_dict = score_candidates(
        candidate_dict, s1_info, target_info, model, threshold
    )

    # Clean up country tables
    con.execute("DROP TABLE IF EXISTS s1_country; DROP TABLE IF EXISTS target_country;")

    elapsed = time.time() - t0
    total_matched = sum(len(v) for v in matched_dict.values())
    total_cands   = sum(len(v) for v in candidate_dict.values())
    singletons    = sum(1 for v in matched_dict.values() if not v)
    print(f"  Done {country} in {elapsed:.1f}s: {total_cands:,} cands, "
          f"{total_matched:,} matches, {singletons:,} singletons", flush=True)

    return candidate_dict, matched_dict


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(mode: str = "test", threshold: float = DEFAULT_DECISION_THRESHOLD):
    """
    End-to-end pipeline:
      1. Load trained model
      2. Process each country: 8-pass chunked blocking + K=25 + XGBoost scoring
      3. Write matching_results.tsv + candidate_pairs.tsv
      4. Validate with official validator
    """
    print("=" * 65, flush=True)
    print("  AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION", flush=True)
    print("  OOM-safe 8-pass chunked pipeline (v3)", flush=True)
    print("=" * 65, flush=True)

    # Load model
    try:
        model = load_matching_model()
        print("Loaded XGBoost model.", flush=True)
    except Exception as e:
        print(f"ERROR: Model not found: {e}\nRun train_model.py first!", flush=True)
        return

    data_dir = TEST_DIR if mode == "test" else TRAIN_DIR
    s1_path = os.path.join(data_dir, f"{mode}_source1.tsv").replace("\\", "/")
    s2_path = os.path.join(data_dir, f"{mode}_source2.tsv").replace("\\", "/")
    s3_path = os.path.join(data_dir, f"{mode}_source3.tsv").replace("\\", "/")

    print(f"\nMode: {mode.upper()}  |  Threshold: {threshold}  |  K={MAX_CANDIDATES_PER_ENTITY}", flush=True)
    print(f"S1: {s1_path}", flush=True)

    con = make_connection()
    register_macros(con)

    # Detect countries
    countries = [
        r[0] for r in con.execute(f"""
        SELECT DISTINCT COALESCE(country,'') AS c
        FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
        ORDER BY count(*) DESC
        """).fetchall()
    ]
    print(f"\nCountry partitions: {countries}", flush=True)

    all_candidates: Dict[str, List[str]] = {}
    all_matches:    Dict[str, List[str]] = {}

    t_pipeline = time.time()
    for country in countries:
        if not country:
            continue
        c_cands, c_matches = process_country_partition(
            con, model, country, s1_path, s2_path, s3_path, threshold
        )
        all_candidates.update(c_cands)
        all_matches.update(c_matches)

    total_pipeline = time.time() - t_pipeline
    print(f"\nAll countries done in {total_pipeline:.1f}s ({total_pipeline/60:.1f} min)", flush=True)

    # Write output files in S1 row order
    print("\nReading S1 row order for output...", flush=True)
    s1_df = pl.read_csv(s1_path, separator="\t")
    ordered_ids = s1_df["entity_id"].to_list()
    print(f"Total S1 entities: {len(ordered_ids):,}", flush=True)

    # matching_results.tsv
    print(f"Writing {MATCHING_RESULTS_PATH}...", flush=True)
    with open(MATCHING_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_ids:
            matches = all_matches.get(sid, [])
            seen: set = set()
            deduped = [m for m in matches if not (m in seen or seen.add(m))]  # type: ignore
            f.write(f"{sid}\t{','.join(deduped)}\n")

    # candidate_pairs.tsv
    print(f"Writing {CANDIDATE_PAIRS_PATH}...", flush=True)
    with open(CANDIDATE_PAIRS_PATH, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered_ids:
            cands = all_candidates.get(sid, [])
            seen = set()
            deduped = [c for c in cands if not (c in seen or seen.add(c))]  # type: ignore
            f.write(f"{sid}\t{','.join(deduped)}\n")

    mr_size = os.path.getsize(MATCHING_RESULTS_PATH)
    cp_size = os.path.getsize(CANDIDATE_PAIRS_PATH)
    print(f"\nmatching_results.tsv : {mr_size:,} bytes", flush=True)
    print(f"candidate_pairs.tsv  : {cp_size:,} bytes", flush=True)

    # Run official validator
    validator_path = os.path.join(STUDENT_RESOURCE_DIR, "utils", "validate_submission.py")
    if os.path.exists(validator_path):
        print("\n" + "=" * 65, flush=True)
        print("  RUNNING OFFICIAL SUBMISSION VALIDATOR", flush=True)
        print("=" * 65, flush=True)
        cmd = (f'python "{validator_path}" '
               f'--matching "{MATCHING_RESULTS_PATH}" '
               f'--candidate "{CANDIDATE_PAIRS_PATH}" '
               f'--test-dir "{data_dir}"')
        print(f"Command: {cmd}", flush=True)
        os.system(cmd)
    else:
        print(f"\nValidator not found at: {validator_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution — OOM-safe pipeline v3")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "train"], help="Dataset split to run on")
    parser.add_argument("--threshold", type=float, default=DEFAULT_DECISION_THRESHOLD,
                        help=f"XGBoost decision threshold (default={DEFAULT_DECISION_THRESHOLD})")
    args = parser.parse_args()
    run_pipeline(mode=args.mode, threshold=args.threshold)
