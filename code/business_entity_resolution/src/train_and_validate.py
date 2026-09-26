# -*- coding: utf-8 -*-
"""
train_and_validate.py
=====================
Complete model training and validation pipeline for Amazon ML Challenge 2026.

Features:
- Configurable DATA_ROOT, S1 entity limit (10k, 50k, 150k, None), and random seed.
- Proper S1-level train/validation split (80/20) with zero candidate leakage.
- Final 8-pass blocking candidate generation with token-Jaccard pre-ranking and K=25.
- Hard-negative prioritization and sampling.
- 24 rich candidate-pair similarity features (RapidFuzz only on top-K).
- Model comparison: Logistic Regression vs XGBoost.
- Threshold search (0.30 to 0.95) optimizing exact challenge Macro F0.5.
- Comprehensive error analysis (FP and FN categories).
- Full export of models to artifacts/ and tables/reports to results/.
"""

import os
import sys
import time
import json
import argparse
import random
import tracemalloc
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import joblib

# Set UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuration Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "6ab10eb3b23ba_student_resource", "student_resource", "dataset", "train")
)
DEFAULT_SEED = 42
DEFAULT_S1_LIMIT = 150000
DEFAULT_VAL_FRACTION = 0.20
DEFAULT_CHUNK_SIZE = 50000
DEFAULT_TOP_K = 25
DEFAULT_HARD_NEG_RATIO = 3.0

FEATURE_NAMES = [
    # Name features
    "name_exact_raw",
    "name_exact_norm",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_levenshtein_ratio",
    "name_char_sim",
    "name_substring_containment",
    # Address features
    "addr_exact_norm",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_char_sim",
    "addr_num_overlap",
    "addr_postal_match",
    # Other features
    "country_agreement",
    "s1_name_missing",
    "t_name_missing",
    "s1_addr_missing",
    "t_addr_missing",
    "candidate_rank",
    "blocking_pass_count",
    "name_x_addr",
    "name_diff_addr"
]


# ---------------------------------------------------------------------------
# Preprocessing SQL Macros
# ---------------------------------------------------------------------------
def register_duckdb_macros(con: duckdb.DuckDBPyConnection):
    con.execute("""
    CREATE OR REPLACE MACRO norm_text(s) AS
        lower(trim(regexp_replace(regexp_replace(
            regexp_replace(coalesce(s,''), 'https?://(?:www\\.)?|www\\.', '', 'g'),
            '\\.(?:com|org|net|in|co|gov|edu|fr|io)\\b', ' ', 'g'),
            '[^\\w\\s]', ' ', 'g')));

    CREATE OR REPLACE MACRO clean_legal(s) AS
        trim(regexp_replace(norm_text(s),
        '\\b(?:private\\s+limited|pvt\\s+ltd|pvt\\s+limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|holdings|holding|services|service|company|limited|llc|inc|corp|ltd|llp|co|sarl|sas|sa|sci|eurl|sasu)\\b',
        ' ', 'g'));

    CREATE OR REPLACE MACRO compact_name(s) AS
        regexp_replace(norm_text(s), '\\s+', '', 'g');

    CREATE OR REPLACE MACRO get_token(s, n) AS
        CASE WHEN array_length(string_split(trim(coalesce(s,'')), ' ')) >= n
             THEN string_split(trim(coalesce(s,'')), ' ')[n]
             ELSE '' END;

    CREATE OR REPLACE MACRO clean_addr(s) AS
        lower(trim(regexp_replace(
            regexp_replace(coalesce(s,''), '[^\\w\\s]', ' ', 'g'),
            '\\s+', ' ', 'g')));
    """)


# ---------------------------------------------------------------------------
# Feature Extraction Functions
# ---------------------------------------------------------------------------
def compute_pair_features(
    s1_raw_name: str, s1_norm_name: str, s1_addr: str, s1_postal: str, s1_door: str,
    t_raw_name: str, t_norm_name: str, t_addr: str, t_postal: str, t_door: str,
    s1_country: str, t_country: str, rank: int, pass_count: int
) -> List[float]:
    """Compute 24 rich candidate-pair similarity features."""
    # 1. Name features
    exact_raw = 1.0 if s1_raw_name and s1_raw_name == t_raw_name else 0.0
    exact_norm = 1.0 if s1_norm_name and s1_norm_name == t_norm_name else 0.0
    name_sort = fuzz.token_sort_ratio(s1_norm_name, t_norm_name) / 100.0
    name_set = fuzz.token_set_ratio(s1_norm_name, t_norm_name) / 100.0
    jw = distance.JaroWinkler.similarity(s1_norm_name, t_norm_name)
    lev = distance.Levenshtein.normalized_similarity(s1_norm_name, t_norm_name)
    name_char_sim = fuzz.ratio(s1_norm_name, t_norm_name) / 100.0
    name_substr = 1.0 if (s1_norm_name and t_norm_name and (s1_norm_name in t_norm_name or t_norm_name in s1_norm_name)) else 0.0

    # 2. Address features
    addr_exact = 1.0 if s1_addr and s1_addr == t_addr else 0.0
    addr_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
    addr_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0

    s1_a_tok = set(s1_addr.split()) if s1_addr else set()
    t_a_tok = set(t_addr.split()) if t_addr else set()
    addr_jaccard = len(s1_a_tok & t_a_tok) / len(s1_a_tok | t_a_tok) if (s1_a_tok or t_a_tok) else 0.0
    addr_char_sim = fuzz.ratio(s1_addr, t_addr) / 100.0

    # Digit / number overlap
    s1_nums = {w for w in s1_a_tok if w.isdigit()}
    t_nums = {w for w in t_a_tok if w.isdigit()}
    if s1_nums and t_nums:
        num_overlap = len(s1_nums & t_nums) / len(s1_nums | t_nums)
    elif not s1_nums and not t_nums:
        num_overlap = 0.5
    else:
        num_overlap = 0.0

    postal_match = 1.0 if s1_postal and t_postal and s1_postal == t_postal else 0.0

    # 3. Other features
    country_agree = 1.0 if s1_country and s1_country == t_country else 0.0
    s1_name_miss = 1.0 if not s1_norm_name else 0.0
    t_name_miss = 1.0 if not t_norm_name else 0.0
    s1_addr_miss = 1.0 if not s1_addr else 0.0
    t_addr_miss = 1.0 if not t_addr else 0.0
    rank_feat = 1.0 / rank if rank > 0 else 0.0
    pass_cnt_feat = float(pass_count)
    name_x_addr = name_sort * addr_sort
    name_diff_addr = abs(name_sort - addr_sort)

    return [
        exact_raw, exact_norm, name_sort, name_set, jw, lev, name_char_sim, name_substr,
        addr_exact, addr_sort, addr_set, addr_jaccard, addr_char_sim, num_overlap, postal_match,
        country_agree, s1_name_miss, t_name_miss, s1_addr_miss, t_addr_miss,
        rank_feat, pass_cnt_feat, name_x_addr, name_diff_addr
    ]


# ---------------------------------------------------------------------------
# Metric Evaluation Functions
# ---------------------------------------------------------------------------
def compute_single_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """Exact Macro F0.5 per S1 entity with competition singleton rules."""
    if not true_matches:
        return 1.0 if not pred_matches else 0.0
    if not pred_matches:
        return 0.0
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_matches)
    rec = tp / len(true_matches)
    denom = 0.25 * prec + rec
    return (1.25 * prec * rec) / denom if denom > 0 else 0.0


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Tuple[float, float, float, int, int]:
    """Calculate macro F0.5, micro precision, micro recall, total FP, total FN."""
    f05_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05_scores.append(compute_single_entity_f05(true_set, pred_set))
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f05 = float(np.mean(f05_scores))
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    return macro_f05, precision, recall, total_fp, total_fn


# ---------------------------------------------------------------------------
# Main Training & Validation Runner
# ---------------------------------------------------------------------------
def run_training_pipeline(
    data_root: str = DEFAULT_DATA_ROOT,
    seed: int = DEFAULT_SEED,
    s1_limit: int = DEFAULT_S1_LIMIT,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    top_k: int = DEFAULT_TOP_K,
    hard_neg_ratio: float = DEFAULT_HARD_NEG_RATIO,
    project_root: str = None
):
    start_time = time.time()
    tracemalloc.start()

    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    artifacts_dir = os.path.join(project_root, "artifacts")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(os.path.join(artifacts_dir, "logistic_regression"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "xgboost"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70, flush=True)
    print("  AMAZON ML CHALLENGE 2026: CANDIDATE-PAIR MODEL TRAINING  ", flush=True)
    print("=" * 70, flush=True)
    print(f"Data root:            {data_root}")
    print(f"Seed:                 {seed}")
    print(f"S1 entity limit:      {s1_limit if s1_limit else 'Full Dataset'}")
    print(f"Validation fraction:  {val_fraction * 100:.0f}%")
    print(f"Top K candidates:     {top_k}")
    print(f"Hard negative ratio:  {hard_neg_ratio}:1")
    print("-" * 70, flush=True)

    # 1. Verify Dataset Files
    s1_path = os.path.join(data_root, "train_source1.tsv").replace("\\", "/")
    s2_path = os.path.join(data_root, "train_source2.tsv").replace("\\", "/")
    s3_path = os.path.join(data_root, "train_source3.tsv").replace("\\", "/")
    gt_path = os.path.join(data_root, "train_ground_truth.tsv").replace("\\", "/")

    for name, path in [("S1", s1_path), ("S2", s2_path), ("S3", s3_path), ("GT", gt_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file {name} not found at: {path}")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [OK] {name:<4} exists: {path} ({size_mb:.1f} MB)", flush=True)

    # 2. Setup DuckDB Connection
    db_file = os.path.join(project_root, "scratch", "train_val_pipeline.duckdb")
    if os.path.exists(db_file):
        try: os.remove(db_file)
        except: pass
    con = duckdb.connect(db_file)
    con.execute("SET memory_limit='6GB';")
    con.execute("SET threads=8;")
    con.execute("SET preserve_insertion_order=false;")
    register_duckdb_macros(con)

    # 3. Load & Sample S1 Entities
    print("\n[Step 1/8] Loading and sampling Source 1 entities...", flush=True)
    sample_clause = f"USING SAMPLE {s1_limit} (reservoir, {seed})" if s1_limit else ""
    con.execute(f"""
    CREATE TABLE s1_sample AS
    SELECT 
        entity_id,
        coalesce(country, '') as country,
        business_name as raw_name,
        business_address as raw_addr,
        norm_text(business_name) as norm_name,
        clean_legal(business_name) as core_name,
        clean_addr(business_address) as norm_addr,
        get_token(clean_legal(business_name), 1) as p1,
        get_token(clean_legal(business_name), 2) as p2,
        get_token(clean_legal(business_name), 3) as p3,
        left(compact_name(business_name), 4) as cmp4,
        left(compact_name(business_name), 5) as cmp5,
        regexp_extract(clean_addr(business_address), '\\b(\\d{{3,}})\\b', 1) as postal,
        regexp_extract(clean_addr(business_address), '^(\\d+)', 1) as door,
        get_token(clean_addr(business_address), 1) as a1,
        get_token(clean_addr(business_address), 2) as a2
    FROM read_csv_auto('{s1_path}', delim='\t', header=true, quote='')
    {sample_clause};
    """)
    total_s1 = con.execute("SELECT count(*) FROM s1_sample").fetchone()[0]
    print(f"Sampled {total_s1:,} Source 1 entities.", flush=True)

    # 4. Join Ground Truth and Analyze Distributions
    print("\n[Step 2/8] Joining Ground Truth & Stratified S1 Train/Validation Split...", flush=True)
    con.execute(f"""
    CREATE TABLE s1_gt AS
    SELECT 
        s.*,
        g.matched_entity_ids,
        case when g.matched_entity_ids is null or trim(g.matched_entity_ids) = '' then 0
             else length(g.matched_entity_ids) - length(replace(g.matched_entity_ids, ',', '')) + 1 end as match_count,
        case when g.matched_entity_ids is null or trim(g.matched_entity_ids) = '' then 1 else 0 end as is_singleton
    FROM s1_sample s
    LEFT JOIN read_csv_auto('{gt_path}', delim='\t', header=true, quote='') g
    ON s.entity_id = g.source1_entity_id;
    """)

    # S1 Split: 80% Train, 20% Validation by S1 entity
    con.execute(f"""
    CREATE TABLE s1_split AS
    SELECT *,
        case when row_number() over (partition by country, is_singleton order by hash(entity_id)) <= count(*) over (partition by country, is_singleton) * (1.0 - {val_fraction})
             then 'train' else 'val' end as split
    FROM s1_gt;
    """)

    # Print distributions
    split_dist = con.execute("""
    SELECT split, country, count(*) as s1_cnt, sum(is_singleton) as singleton_cnt, sum(match_count) as total_true_matches
    FROM s1_split GROUP BY split, country ORDER BY split, country
    """).fetchall()
    print("Train/Validation S1 Distributions:")
    for sp, co, scnt, sngcnt, tmatch in split_dist:
        print(f"  [{sp.upper():<5}] Country: {co:<6} | S1 Count: {scnt:>6,} | Singletons: {sngcnt:>5,} ({sngcnt/scnt*100:4.1f}%) | True Matches: {tmatch:>7,}")

    train_s1_count = con.execute("SELECT count(*) FROM s1_split WHERE split='train'").fetchone()[0]
    val_s1_count = con.execute("SELECT count(*) FROM s1_split WHERE split='val'").fetchone()[0]

    # Map ground truth for validation
    val_gt_rows = con.execute("SELECT entity_id, matched_entity_ids FROM s1_split WHERE split='val'").fetchall()
    val_gt: Dict[str, Set[str]] = {}
    for sid, mstr in val_gt_rows:
        if mstr and str(mstr).strip():
            val_gt[sid] = {m.strip() for m in str(mstr).split(",") if m.strip()}
        else:
            val_gt[sid] = set()

    # 5. Build Target Pool (True Targets + Realistic Background Distractors)
    print("\n[Step 3/8] Building Target Candidate Pool...", flush=True)
    con.execute("""
    CREATE TABLE true_target_ids AS
    SELECT DISTINCT unnest(string_split(matched_entity_ids, ',')) as target_id
    FROM s1_gt WHERE matched_entity_ids is not null and trim(matched_entity_ids) != '';
    """)
    true_tgt_count = con.execute("SELECT count(*) FROM true_target_ids").fetchone()[0]
    print(f"Identified {true_tgt_count:,} true target IDs required across S1 entities.", flush=True)

    # Target pool = true targets + 300k random distractors
    con.execute(f"""
    CREATE TABLE target_pool AS
    SELECT 
        entity_id, coalesce(country,'') as country,
        business_name as raw_name, business_address as raw_addr,
        norm_text(business_name) as norm_name,
        clean_legal(business_name) as core_name,
        clean_addr(business_address) as norm_addr,
        get_token(clean_legal(business_name), 1) as p1,
        get_token(clean_legal(business_name), 2) as p2,
        get_token(clean_legal(business_name), 3) as p3,
        left(compact_name(business_name), 4) as cmp4,
        left(compact_name(business_name), 5) as cmp5,
        regexp_extract(clean_addr(business_address), '\\b(\\d{{3,}})\\b', 1) as postal,
        regexp_extract(clean_addr(business_address), '^(\\d+)', 1) as door,
        get_token(clean_addr(business_address), 1) as a1,
        get_token(clean_addr(business_address), 2) as a2
    FROM (
        SELECT * FROM read_csv_auto('{s2_path}', delim='\t', header=true, quote='')
        WHERE entity_id IN (SELECT target_id FROM true_target_ids)
        UNION ALL
        SELECT * FROM read_csv_auto('{s3_path}', delim='\t', header=true, quote='')
        WHERE entity_id IN (SELECT target_id FROM true_target_ids)
        UNION ALL
        (SELECT * FROM read_csv_auto('{s2_path}', delim='\t', header=true, quote='') USING SAMPLE 150000 (reservoir, {seed}))
        UNION ALL
        (SELECT * FROM read_csv_auto('{s3_path}', delim='\t', header=true, quote='') USING SAMPLE 150000 (reservoir, {seed}))
    );
    """)
    target_pool_count = con.execute("SELECT count(*) FROM target_pool").fetchone()[0]
    print(f"Target candidate pool created: {target_pool_count:,} records.", flush=True)

    # 6. Candidate Generation (8-Pass Blocking + Token Jaccard Pre-ranking K=25)
    print("\n[Step 4/8] Candidate Generation: 8-pass blocking with Token-Jaccard K=25 pre-ranking...", flush=True)
    t_block_start = time.time()

    con.execute("""
    CREATE TABLE raw_candidate_pairs AS
    SELECT sub.entity_id as s1_id, sub.target_id, count(*) as pass_count
    FROM (
        SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2 WHERE length(s.p1)>=3 AND length(s.p2)>=3
        UNION ALL
        SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.postal=t.postal WHERE length(s.p1)>=3 AND s.postal!=''
        UNION ALL
        SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.door=t.door WHERE length(s.p1)>=3 AND s.door!=''
        UNION ALL
        SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.cmp5=t.cmp5 WHERE length(s.cmp5)>=5
        UNION ALL
        SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 WHERE length(s.p1)>=5
        UNION ALL
        SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.postal=t.postal AND s.door=t.door WHERE s.postal!='' AND length(s.door)>=2
        UNION ALL
        SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.door=t.door AND s.a1=t.a1 WHERE length(s.door)>=2 AND length(s.a1)>=4
        UNION ALL
        SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.door=t.door AND s.a2=t.a2 WHERE length(s.door)>=2 AND length(s.a2)>=4
    ) sub
    GROUP BY sub.entity_id, sub.target_id;
    """)
    raw_pair_count = con.execute("SELECT count(*) FROM raw_candidate_pairs").fetchone()[0]
    print(f"Generated {raw_pair_count:,} raw blocking pairs in {time.time()-t_block_start:.2f}s.", flush=True)

    # Join attributes for token Jaccard pre-ranking
    print("Ranking candidates with token Jaccard (0.6*name + 0.4*addr)...", flush=True)
    con.execute(f"""
    CREATE TABLE ranked_candidates AS
    WITH pair_scored AS (
        SELECT 
            c.s1_id, c.target_id, c.pass_count,
            s.split, s.country as s1_country, s.raw_name as s1_raw_name, s.raw_addr as s1_raw_addr, s.norm_name as s1_norm_name, s.norm_addr as s1_norm_addr, s.postal as s1_postal, s.door as s1_door,
            t.country as t_country, t.raw_name as t_raw_name, t.raw_addr as t_raw_addr, t.norm_name as t_norm_name, t.norm_addr as t_norm_addr, t.postal as t_postal, t.door as t_door,
            -- Fast token Jaccard
            (0.6 * (len(list_intersect(string_split(s.norm_name, ' '), string_split(t.norm_name, ' ')))::float / 
                    nullif(len(list_distinct(list_concat(string_split(s.norm_name, ' '), string_split(t.norm_name, ' ')))), 0)) +
             0.4 * (len(list_intersect(string_split(s.norm_addr, ' '), string_split(t.norm_addr, ' ')))::float / 
                    nullif(len(list_distinct(list_concat(string_split(s.norm_addr, ' '), string_split(t.norm_addr, ' ')))), 0))) as jaccard_score
        FROM raw_candidate_pairs c
        JOIN s1_split s ON c.s1_id = s.entity_id
        JOIN target_pool t ON c.target_id = t.entity_id
    )
    SELECT *,
        row_number() over (partition by s1_id order by coalesce(jaccard_score, 0) desc, pass_count desc) as candidate_rank
    FROM pair_scored;

    CREATE TABLE topk_candidates AS
    SELECT * FROM ranked_candidates WHERE candidate_rank <= {top_k};
    """)
    topk_pair_count = con.execute("SELECT count(*) FROM topk_candidates").fetchone()[0]

    # Candidate Statistics
    per_s1_counts = np.array([r[0] for r in con.execute("SELECT count(*) FROM topk_candidates GROUP BY s1_id").fetchall()])
    cands_avg = float(np.mean(per_s1_counts)) if len(per_s1_counts) else 0.0
    cands_med = float(np.median(per_s1_counts)) if len(per_s1_counts) else 0.0
    cands_p95 = float(np.percentile(per_s1_counts, 95)) if len(per_s1_counts) else 0.0
    cands_max = int(np.max(per_s1_counts)) if len(per_s1_counts) else 0
    print(f"Top-K Candidate Pairs: {topk_pair_count:,} across {len(per_s1_counts):,} S1 entities.")
    print(f"  Avg: {cands_avg:.1f} | Med: {cands_med:.0f} | p95: {cands_p95:.0f} | Max: {cands_max}")

    # 7. Positive / Negative Label Generation & Hard-Negative Prioritization
    print("\n[Step 5/8] Labeling Candidate Pairs & Hard-Negative Sampling...", flush=True)
    con.execute("""
    CREATE TABLE labeled_topk AS
    SELECT 
        c.*,
        case when g.target_id is not null then 1 else 0 end as label
    FROM topk_candidates c
    LEFT JOIN (
        SELECT entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) as target_id
        FROM s1_gt WHERE matched_entity_ids is not null
    ) g ON c.s1_id = g.s1_id AND c.target_id = g.target_id;
    """)

    label_counts = con.execute("SELECT split, label, count(*) FROM labeled_topk GROUP BY split, label ORDER BY split, label").fetchall()
    print("Candidate Label Distribution:")
    for sp, lb, cnt in label_counts:
        print(f"  {sp.upper()} | Label {lb} ({'Positive' if lb==1 else 'Negative'}): {cnt:>8,}")

    train_pos_cnt = con.execute("SELECT count(*) FROM labeled_topk WHERE split='train' AND label=1").fetchone()[0]
    train_neg_cnt = con.execute("SELECT count(*) FROM labeled_topk WHERE split='train' AND label=0").fetchone()[0]

    # Hard-negative sampling:
    # Prioritize negatives with high candidate rank (<= 5), high name similarity with different address, or same postal/door
    max_train_negs = int(train_pos_cnt * hard_neg_ratio)
    con.execute(f"""
    CREATE TABLE train_pairs_selected AS
    SELECT * FROM labeled_topk WHERE split='train' AND label=1
    UNION ALL
    (
        SELECT * FROM labeled_topk 
        WHERE split='train' AND label=0
        ORDER BY candidate_rank ASC, coalesce(jaccard_score, 0) DESC
        LIMIT {max_train_negs}
    );
    """)
    sampled_train_cnt = con.execute("SELECT count(*) FROM train_pairs_selected").fetchone()[0]
    sampled_neg_cnt = sampled_train_cnt - train_pos_cnt
    print(f"Sampled balanced training set: {sampled_train_cnt:,} pairs ({train_pos_cnt:,} Positives, {sampled_neg_cnt:,} Hard Negatives, Ratio: {sampled_neg_cnt/train_pos_cnt:.2f}:1)")

    # 8. Feature Extraction
    print("\n[Step 6/8] Extracting 24 RapidFuzz features...", flush=True)
    t_feat_start = time.time()

    # Load train and validation pairs into memory for feature computation
    df_train_pairs = con.execute("SELECT * FROM train_pairs_selected").df()
    df_val_pairs = con.execute("SELECT * FROM labeled_topk WHERE split='val'").df()

    def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(df)
        X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.int32)

        s1_raw = df["s1_raw_name"].fillna("").values
        s1_norm = df["s1_norm_name"].fillna("").values
        s1_addr = df["s1_norm_addr"].fillna("").values
        s1_postal = df["s1_postal"].fillna("").values
        s1_door = df["s1_door"].fillna("").values
        s1_country = df["s1_country"].fillna("").values

        t_raw = df["t_raw_name"].fillna("").values
        t_norm = df["t_norm_name"].fillna("").values
        t_addr = df["t_norm_addr"].fillna("").values
        t_postal = df["t_postal"].fillna("").values
        t_door = df["t_door"].fillna("").values
        t_country = df["t_country"].fillna("").values

        ranks = df["candidate_rank"].values
        pass_counts = df["pass_count"].values

        for i in range(n):
            X[i] = compute_pair_features(
                s1_raw[i], s1_norm[i], s1_addr[i], s1_postal[i], s1_door[i],
                t_raw[i], t_norm[i], t_addr[i], t_postal[i], t_door[i],
                s1_country[i], t_country[i], ranks[i], pass_counts[i]
            )
        return X, y

    X_train, y_train = build_feature_matrix(df_train_pairs)
    X_val, y_val = build_feature_matrix(df_val_pairs)
    print(f"Features extracted in {time.time()-t_feat_start:.2f}s! X_train: {X_train.shape}, X_val: {X_val.shape}")

    # 9. Model Training: Logistic Regression vs XGBoost
    print("\n[Step 7/8] Training Candidate Matching Models...", flush=True)

    # --- Model A: Logistic Regression ---
    print("  Training Model A: Logistic Regression...", flush=True)
    t_lr_start = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr_model = LogisticRegression(C=1.0, max_iter=500, random_state=seed, solver="lbfgs")
    lr_model.fit(X_train_scaled, y_train)
    lr_time = time.time() - t_lr_start
    lr_val_probs = lr_model.predict_proba(X_val_scaled)[:, 1]
    print(f"  Logistic Regression trained in {lr_time:.2f}s.")

    # --- Model B: XGBoost ---
    print("  Training Model B: XGBoost...", flush=True)
    t_xgb_start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=seed,
        n_jobs=8
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - t_xgb_start
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    print(f"  XGBoost trained in {xgb_time:.2f}s.")

    # Feature Importance for XGBoost
    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 XGBoost Features:")
    for rank_idx in range(min(10, len(FEATURE_NAMES))):
        fi_idx = sorted_idx[rank_idx]
        print(f"  {rank_idx+1:>2}. {FEATURE_NAMES[fi_idx]:<26} : {importances[fi_idx]*100:5.2f}%")

    # 10. Threshold Search & Evaluation (0.30 to 0.95)
    print("\n[Step 8/8] Threshold Search on Validation Macro F0.5...", flush=True)
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]

    val_s1_ids = df_val_pairs["s1_id"].values
    val_target_ids = df_val_pairs["target_id"].values

    def run_threshold_grid(val_probs: np.ndarray, model_name: str):
        # Group scores by S1 entity
        scores_by_s1 = defaultdict(list)
        for s1_id, tgt_id, prob in zip(val_s1_ids, val_target_ids, val_probs):
            scores_by_s1[s1_id].append((tgt_id, float(prob)))

        results = []
        best_f05 = -1.0
        best_row = None

        for tau in thresholds:
            preds = {}
            for s1_id in val_gt.keys():
                cands = scores_by_s1.get(s1_id, [])
                matched = {t_id for t_id, prob in cands if prob >= tau}
                preds[s1_id] = matched

            macro_f05, prec, rec, fp, fn = evaluate_predictions(val_gt, preds)
            avg_matches = sum(len(p) for p in preds.values()) / len(val_gt)

            row = {
                "model": model_name,
                "threshold": tau,
                "macro_f05": round(macro_f05, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "false_positives": fp,
                "false_negatives": fn,
                "avg_matches_per_s1": round(avg_matches, 3)
            }
            results.append(row)
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_row = row

        return results, best_row

    lr_thresh_results, lr_best = run_threshold_grid(lr_val_probs, "LogisticRegression")
    xgb_thresh_results, xgb_best = run_threshold_grid(xgb_val_probs, "XGBoost")

    # Overall Metric Comparison
    lr_roc = roc_auc_score(y_val, lr_val_probs)
    lr_pr = average_precision_score(y_val, lr_val_probs)
    xgb_roc = roc_auc_score(y_val, xgb_val_probs)
    xgb_pr = average_precision_score(y_val, xgb_val_probs)

    lr_f1 = f1_score(y_val, (lr_val_probs >= lr_best["threshold"]).astype(int))
    xgb_f1 = f1_score(y_val, (xgb_val_probs >= xgb_best["threshold"]).astype(int))

    print("\n" + "=" * 80)
    print("  VALIDATION MODEL COMPARISON  (Exact Macro F0.5)")
    print("=" * 80)
    print(f"{'Model':<20} {'Optimal tau*':>12} {'Macro F0.5':>12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 80)
    print(f"{'LogisticRegression':<20} {lr_best['threshold']:>12.2f} {lr_best['macro_f05']:>12.4f} {lr_best['precision']:>10.4f} {lr_best['recall']:>10.4f} {lr_f1:>8.4f} {lr_roc:>10.4f} {lr_pr:>10.4f}")
    print(f"{'XGBoost':<20} {xgb_best['threshold']:>12.2f} {xgb_best['macro_f05']:>12.4f} {xgb_best['precision']:>10.4f} {xgb_best['recall']:>10.4f} {xgb_f1:>8.4f} {xgb_roc:>10.4f} {xgb_pr:>10.4f}")
    print("=" * 80)

    # Determine Winning Model
    if xgb_best["macro_f05"] >= lr_best["macro_f05"]:
        winning_model_name = "XGBoost"
        winning_best = xgb_best
        winning_probs = xgb_val_probs
    else:
        winning_model_name = "LogisticRegression"
        winning_best = lr_best
        winning_probs = lr_val_probs

    print(f"\nWINNING MODEL: {winning_model_name} with Macro F0.5 = {winning_best['macro_f05']:.4f} at tau* = {winning_best['threshold']:.2f}")

    # 11. Error Analysis on Winning Model
    print("\nAnalyzing False Positives and False Negatives for Error Analysis...", flush=True)
    df_val_eval = df_val_pairs.copy()
    df_val_eval["pred_prob"] = winning_probs
    df_val_eval["pred_match"] = (winning_probs >= winning_best["threshold"]).astype(int)

    # Category counts
    fp_df = df_val_eval[(df_val_eval["label"] == 0) & (df_val_eval["pred_match"] == 1)]
    fn_df = df_val_eval[(df_val_eval["label"] == 1) & (df_val_eval["pred_match"] == 0)]

    # Categorize FP
    fp_categories = defaultdict(int)
    fp_samples = []
    for _, r in fp_df.iterrows():
        s1_n, t_n = r["s1_norm_name"], r["t_norm_name"]
        s1_a, t_a = r["s1_norm_addr"], r["t_norm_addr"]
        name_sim = fuzz.token_sort_ratio(s1_n, t_n)
        addr_sim = fuzz.token_sort_ratio(s1_a, t_a)

        if name_sim >= 85 and addr_sim < 40:
            cat = "same_name_diff_address"
        elif addr_sim >= 80 and name_sim < 40:
            cat = "same_address_diff_business"
        elif r["s1_door"] and r["t_door"] and r["s1_door"] != r["t_door"] and name_sim >= 70:
            cat = "address_number_conflict"
        elif len(s1_n) <= 6 or len(t_n) <= 6:
            cat = "abbreviation_collision"
        else:
            cat = "generic_name_collision"

        fp_categories[cat] += 1
        if len(fp_samples) < 5:
            fp_samples.append({
                "category": cat,
                "s1_name": r["s1_raw_name"],
                "target_name": r["t_raw_name"],
                "s1_addr": r["s1_raw_addr"],
                "target_addr": r["t_raw_addr"],
                "prob": round(float(r["pred_prob"]), 3)
            })

    # Categorize FN
    fn_categories = defaultdict(int)
    fn_samples = []
    for _, r in fn_df.iterrows():
        s1_n, t_n = r["s1_norm_name"], r["t_norm_name"]
        s1_a, t_a = r["s1_norm_addr"], r["t_norm_addr"]
        name_sim = fuzz.token_sort_ratio(s1_n, t_n)

        if not s1_n or not t_n:
            cat = "missing_name"
        elif not s1_a or not t_a:
            cat = "missing_address"
        elif name_sim < 50:
            cat = "trade_name_or_transliteration"
        elif name_sim < 75:
            cat = "severe_typo_or_abbrev"
        else:
            cat = "address_variation"

        fn_categories[cat] += 1
        if len(fn_samples) < 5:
            fn_samples.append({
                "category": cat,
                "s1_name": r["s1_raw_name"],
                "target_name": r["t_raw_name"],
                "s1_addr": r["s1_raw_addr"],
                "target_addr": r["t_raw_addr"],
                "prob": round(float(r["pred_prob"]), 3)
            })

    top_fp_cat = max(fp_categories.items(), key=lambda x: x[1])[0] if fp_categories else "None"
    top_fn_cat = max(fn_categories.items(), key=lambda x: x[1])[0] if fn_categories else "None"

    # 12. Save Model Artifacts
    print("\nSaving Model Artifacts to artifacts/...", flush=True)
    joblib.dump(lr_model, os.path.join(artifacts_dir, "logistic_regression", "model.joblib"))
    joblib.dump(scaler, os.path.join(artifacts_dir, "logistic_regression", "scaler.joblib"))
    xgb_model.save_model(os.path.join(artifacts_dir, "xgboost", "xgboost_model.json"))

    with open(os.path.join(artifacts_dir, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_names": FEATURE_NAMES, "count": len(FEATURE_NAMES)}, f, indent=2)

    with open(os.path.join(artifacts_dir, "preprocessing_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "k_candidates": top_k,
            "seed": seed,
            "s1_limit": s1_limit,
            "val_fraction": val_fraction,
            "blocking_passes": 8,
            "pre_ranking_weights": {"name": 0.6, "address": 0.4}
        }, f, indent=2)

    with open(os.path.join(artifacts_dir, "selected_model.json"), "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": winning_model_name,
            "optimal_threshold": winning_best["threshold"],
            "validation_macro_f05": winning_best["macro_f05"],
            "validation_precision": winning_best["precision"],
            "validation_recall": winning_best["recall"],
            "validation_s1_count": val_s1_count,
            "train_s1_count": train_s1_count,
            "artifact_path": "xgboost/xgboost_model.json" if winning_model_name == "XGBoost" else "logistic_regression/model.joblib"
        }, f, indent=2)

    # 13. Save Results
    print("Saving Experiment Results to results/...", flush=True)
    # Model comparison CSV
    model_comp_rows = [
        {"model": "LogisticRegression", "optimal_threshold": lr_best["threshold"], "macro_f05": lr_best["macro_f05"], "precision": lr_best["precision"], "recall": lr_best["recall"], "f1": round(lr_f1, 4), "roc_auc": round(lr_roc, 4), "pr_auc": round(lr_pr, 4), "train_time_sec": round(lr_time, 2)},
        {"model": "XGBoost", "optimal_threshold": xgb_best["threshold"], "macro_f05": xgb_best["macro_f05"], "precision": xgb_best["precision"], "recall": xgb_best["recall"], "f1": round(xgb_f1, 4), "roc_auc": round(xgb_roc, 4), "pr_auc": round(xgb_pr, 4), "train_time_sec": round(xgb_time, 2)}
    ]
    pd.DataFrame(model_comp_rows).to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

    # Threshold results CSV
    pd.DataFrame(lr_thresh_results + xgb_thresh_results).to_csv(os.path.join(results_dir, "threshold_results.csv"), index=False)

    # Validation predictions sample CSV
    df_val_eval[["s1_id", "target_id", "label", "pred_prob", "pred_match"]].head(10000).to_csv(
        os.path.join(results_dir, "validation_predictions.csv"), index=False
    )

    # Error analysis Markdown
    with open(os.path.join(results_dir, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("# Error Analysis — Amazon ML Challenge 2026\n\n")
        f.write(f"**Model**: {winning_model_name} | **Threshold**: {winning_best['threshold']:.2f} | **Macro F0.5**: {winning_best['macro_f05']:.4f}\n\n")
        f.write("## 1. False Positive Categories\n\n")
        f.write("| Category | Count | Percentage |\n|---|---:|---:|\n")
        total_fps = sum(fp_categories.values())
        for cat, cnt in sorted(fp_categories.items(), key=lambda x: -x[1]):
            f.write(f"| {cat} | {cnt:,} | {cnt/max(1, total_fps)*100:.1f}% |\n")
        f.write("\n### Sample False Positives:\n\n")
        for s in fp_samples:
            f.write(f"- **Category**: {s['category']} (Prob: {s['prob']})\n  - S1: `{s['s1_name']}` | Addr: `{s['s1_addr']}`\n  - Target: `{s['target_name']}` | Addr: `{s['target_addr']}`\n\n")

        f.write("## 2. False Negative Categories\n\n")
        f.write("| Category | Count | Percentage |\n|---|---:|---:|\n")
        total_fns = sum(fn_categories.values())
        for cat, cnt in sorted(fn_categories.items(), key=lambda x: -x[1]):
            f.write(f"| {cat} | {cnt:,} | {cnt/max(1, total_fns)*100:.1f}% |\n")
        f.write("\n### Sample False Negatives:\n\n")
        for s in fn_samples:
            f.write(f"- **Category**: {s['category']} (Prob: {s['prob']})\n  - S1: `{s['s1_name']}` | Addr: `{s['s1_addr']}`\n  - Target: `{s['target_name']}` | Addr: `{s['target_addr']}`\n\n")

    # Training report Markdown
    total_elapsed = time.time() - start_time
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    with open(os.path.join(results_dir, "training_report.md"), "w", encoding="utf-8") as f:
        f.write("# Training Report — Amazon ML Challenge 2026\n\n")
        f.write("## 1. Dataset & Split Configuration\n\n")
        f.write(f"- **Total S1 Sample Size**: {total_s1:,}\n")
        f.write(f"- **Training S1 Count (80%)**: {train_s1_count:,}\n")
        f.write(f"- **Validation S1 Count (20%)**: {val_s1_count:,}\n")
        f.write(f"- **Random Seed**: {seed}\n\n")
        f.write("## 2. Candidate Generation & Sampling\n\n")
        f.write(f"- **Blocking Passes**: 8 passes (passes 1-8)\n")
        f.write(f"- **Raw Candidate Pairs**: {raw_pair_count:,}\n")
        f.write(f"- **Top-K Candidates Retained**: {topk_pair_count:,} (K={top_k})\n")
        f.write(f"- **Candidate Stats**: Avg={cands_avg:.1f}, Med={cands_med:.0f}, p95={cands_p95:.0f}, Max={cands_max}\n")
        f.write(f"- **Train Positive Pairs**: {train_pos_cnt:,}\n")
        f.write(f"- **Train Negative Pairs (Raw)**: {train_neg_cnt:,}\n")
        f.write(f"- **Train Hard Negatives (Sampled)**: {sampled_neg_cnt:,} ({sampled_neg_cnt/train_pos_cnt:.2f}:1 ratio)\n\n")
        f.write("## 3. Model Comparison\n\n")
        f.write("| Model | Optimal Threshold | Validation Macro F0.5 | Precision | Recall | F1 | ROC-AUC | PR-AUC | Runtime (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(f"| Logistic Regression | {lr_best['threshold']:.2f} | **{lr_best['macro_f05']:.4f}** | {lr_best['precision']:.4f} | {lr_best['recall']:.4f} | {lr_f1:.4f} | {lr_roc:.4f} | {lr_pr:.4f} | {lr_time:.2f}s |\n")
        f.write(f"| XGBoost | {xgb_best['threshold']:.2f} | **{xgb_best['macro_f05']:.4f}** | {xgb_best['precision']:.4f} | {xgb_best['recall']:.4f} | {xgb_f1:.4f} | {xgb_roc:.4f} | {xgb_pr:.4f} | {xgb_time:.2f}s |\n\n")
        f.write(f"## 4. Final Decision\n\n")
        f.write(f"- **Best Model**: **{winning_model_name}**\n")
        f.write(f"- **Optimal Decision Threshold (tau*)**: **{winning_best['threshold']:.2f}**\n")
        f.write(f"- **Validation Macro F0.5**: **{winning_best['macro_f05']:.4f}**\n")
        f.write(f"- **Validation Precision**: {winning_best['precision']:.4f}\n")
        f.write(f"- **Validation Recall**: {winning_best['recall']:.4f}\n")
        f.write(f"- **Total Runtime**: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)\n")
        f.write(f"- **Peak Memory**: {peak_mem_mb:.1f} MB\n")

    con.close()
    print("\n" + "=" * 70)
    print("  TRAINING & VALIDATION PIPELINE COMPLETE!  ")
    print("=" * 70)
    print(f"BEST MODEL:               {winning_model_name}")
    print(f"BEST THRESHOLD:           {winning_best['threshold']:.2f}")
    print(f"VALIDATION PRECISION:     {winning_best['precision']:.4f}")
    print(f"VALIDATION RECALL:        {winning_best['recall']:.4f}")
    print(f"VALIDATION F0.5:          {winning_best['macro_f05']:.4f}")
    print(f"POSITIVE PAIRS:           {train_pos_cnt:,}")
    print(f"NEGATIVE PAIRS:           {train_neg_cnt:,}")
    print(f"HARD NEGATIVES (SAMPLED): {sampled_neg_cnt:,}")
    print(f"TRAINING S1 COUNT:        {train_s1_count:,}")
    print(f"VALIDATION S1 COUNT:      {val_s1_count:,}")
    print(f"TOP FALSE-POSITIVE:       {top_fp_cat}")
    print(f"TOP FALSE-NEGATIVE:       {top_fn_cat}")
    print(f"RUNTIME:                  {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
    print(f"PEAK MEMORY:              {peak_mem_mb:.1f} MB")
    print("=" * 70, flush=True)

    return {
        "best_model": winning_model_name,
        "best_threshold": winning_best["threshold"],
        "val_precision": winning_best["precision"],
        "val_recall": winning_best["recall"],
        "val_f05": winning_best["macro_f05"],
        "lr_f05": lr_best["macro_f05"],
        "xgb_f05": xgb_best["macro_f05"],
        "pos_pairs": train_pos_cnt,
        "neg_pairs": train_neg_cnt,
        "hard_negs": sampled_neg_cnt,
        "train_s1": train_s1_count,
        "val_s1": val_s1_count,
        "top_fp": top_fp_cat,
        "top_fn": top_fn_cat,
        "runtime_s": total_elapsed,
        "peak_mem_mb": peak_mem_mb,
        "cand_pairs": topk_pair_count
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candidate-Pair Matching Model Training & Validation")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Path to training data directory")
    parser.add_argument("--s1-limit", type=int, default=DEFAULT_S1_LIMIT, help="S1 sample size limit (e.g. 10000, 50000, 150000, or 0 for all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--hard-neg-ratio", type=float, default=DEFAULT_HARD_NEG_RATIO, help="Ratio of hard negatives to positives")
    args = parser.parse_args()

    s1_limit_val = args.s1_limit if args.s1_limit > 0 else None
    run_training_pipeline(
        data_root=args.data_root,
        seed=args.seed,
        s1_limit=s1_limit_val,
        hard_neg_ratio=args.hard_neg_ratio
    )
