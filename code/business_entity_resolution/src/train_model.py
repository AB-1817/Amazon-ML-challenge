# -*- coding: utf-8 -*-
"""
train_model.py
==============
Full training pipeline for the entity matching XGBoost model.

Key improvements over v1:
  - Trains on 200,000 S1 entities (vs 45,000 before)
  - Uses 8-pass OOM-safe chunked blocking (vs 13-pass full blocking)
  - Proper GT loading (source1_entity_id + matched_entity_ids comma-sep format)
  - Balanced train/val split with realistic background distractors
  - Similarity-sorted K=25 pruning before feature extraction
"""

import os
import sys
import time
import random
import duckdb
import numpy as np
import pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "code", "business_entity_resolution"))

from src.config import TRAIN_DIR, NUM_WORKERS
from src.features import compute_pair_features, FEATURE_NAMES
from src.model import train_matching_model, MODEL_PATH
from src.evaluate import find_optimal_threshold

# -----------------------------------------------------------------------
# Training parameters
# -----------------------------------------------------------------------
N_TRAIN_S1      = 150_000   # S1 entities for training pairs
N_VAL_S1        = 20_000    # S1 entities for validation (held-out)
N_BACKGROUND    = 300_000   # Random S2/S3 distractors in target pool
K_BUDGET        = 25        # Top-K candidates per S1 (similarity-sorted)
S1_CHUNK_SIZE   = 10_000    # Chunk size for blocking (memory safe)
SCORE_CHUNK     = 100_000   # Rows per feature-extraction batch
MAX_NEG_RATIO   = 3         # Hard negatives per positive in training
RANDOM_SEED     = 42
W_NAME          = 0.6
W_ADDR          = 0.4

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv").replace("\\", "/")
gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv").replace("\\", "/")


# -----------------------------------------------------------------------
# DuckDB helpers
# -----------------------------------------------------------------------

def make_con():
    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit='7GB';")
    con.execute(f"SET threads={NUM_WORKERS};")
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


PREP_SELECT = """
SELECT
    entity_id,
    COALESCE(country,'')                                       AS country,
    norm_text(COALESCE(business_name,''))                      AS norm_name,
    clean_legal(COALESCE(business_name,''))                    AS core_name,
    clean_addr(COALESCE(business_address,''))                  AS norm_addr,
    get_token(norm_text(COALESCE(business_name,'')),1)         AS p1,
    get_token(norm_text(COALESCE(business_name,'')),2)         AS p2,
    get_token(norm_text(COALESCE(business_name,'')),3)         AS p3,
    left(compact_name(COALESCE(business_name,'')),4)           AS cmp4,
    left(compact_name(COALESCE(business_name,'')),5)           AS cmp5,
    left(compact_name(COALESCE(business_name,'')),6)           AS cmp6,
    regexp_extract(clean_addr(COALESCE(business_address,'')),
                   '\\b(\\d{3,})\\b', 1)                      AS postal,
    regexp_extract(clean_addr(COALESCE(business_address,'')),
                   '^(\\d+)', 1)                               AS door,
    get_token(clean_addr(COALESCE(business_address,'')),1)     AS a1,
    get_token(clean_addr(COALESCE(business_address,'')),2)     AS a2
FROM src_raw
"""


def _8pass_sql(s1_tbl, tgt_tbl):
    return f"""
    SELECT s.entity_id AS s1_id, t.entity_id AS target_id
    FROM {s1_tbl} s JOIN {tgt_tbl} t ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2
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
    """


# -----------------------------------------------------------------------
# STEP 1: Load ground truth
# -----------------------------------------------------------------------
print("=" * 65, flush=True)
print("  STEP 1 — LOAD GROUND TRUTH", flush=True)
print("=" * 65, flush=True)
t0 = time.time()

con = make_con()
register_macros(con)

gt_rows = con.execute(f"""
SELECT source1_entity_id, matched_entity_ids
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
""").fetchall()

# Parse: matched_entity_ids is comma-separated e.g. "S2-123,S3-456"
gt_pairs = defaultdict(set)   # s1_id -> set of target_ids
singletons = set()
for s1_id, matched_str in gt_rows:
    if not matched_str or str(matched_str).strip() == "":
        singletons.add(s1_id)
    else:
        for tid in str(matched_str).split(","):
            tid = tid.strip()
            if tid:
                gt_pairs[s1_id].add(tid)

all_s1_with_matches = list(gt_pairs.keys())
all_singletons      = list(singletons)
print(f"GT loaded in {time.time()-t0:.1f}s", flush=True)
print(f"  S1 with matches: {len(all_s1_with_matches):,}", flush=True)
print(f"  S1 singletons:   {len(all_singletons):,}", flush=True)

# -----------------------------------------------------------------------
# STEP 2: Sample train / val split
# -----------------------------------------------------------------------
print("\n" + "=" * 65, flush=True)
print("  STEP 2 — TRAIN/VAL SPLIT", flush=True)
print("=" * 65, flush=True)

random.shuffle(all_s1_with_matches)
val_s1_ids   = set(all_s1_with_matches[:N_VAL_S1])
train_s1_ids = set(all_s1_with_matches[N_VAL_S1: N_VAL_S1 + N_TRAIN_S1])

# Include some singletons in both sets (for realistic training)
random.shuffle(all_singletons)
n_sing_val   = min(2_000, len(all_singletons) // 4)
n_sing_train = min(5_000, len(all_singletons) - n_sing_val)
val_s1_ids   |= set(all_singletons[:n_sing_val])
train_s1_ids |= set(all_singletons[n_sing_val: n_sing_val + n_sing_train])

all_selected_s1 = train_s1_ids | val_s1_ids
print(f"  Train S1: {len(train_s1_ids):,}  (incl. {n_sing_train:,} singletons)", flush=True)
print(f"  Val   S1: {len(val_s1_ids):,}  (incl. {n_sing_val:,} singletons)", flush=True)

# -----------------------------------------------------------------------
# STEP 3: Load and preprocess all needed records
# -----------------------------------------------------------------------
print("\n" + "=" * 65, flush=True)
print("  STEP 3 — LOAD & PREPROCESS RECORDS", flush=True)
print("=" * 65, flush=True)
t0 = time.time()

# All true target IDs for selected S1 entities
all_true_targets: set = set()
for s1_id in all_selected_s1:
    all_true_targets.update(gt_pairs.get(s1_id, set()))

print(f"  True targets needed: {len(all_true_targets):,}", flush=True)

# Load S1
s1_sel_df = pd.DataFrame({"entity_id": list(all_selected_s1)})
con.register("_s1_sel", s1_sel_df)
con.execute(f"""
CREATE OR REPLACE TABLE src_raw AS
SELECT * FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true);
""")
con.execute(f"""
CREATE OR REPLACE TABLE s1_prep AS {PREP_SELECT}
WHERE entity_id IN (SELECT entity_id FROM _s1_sel);
""")
s1_n = con.execute("SELECT count(*) FROM s1_prep").fetchone()[0]
print(f"  S1 preprocessed: {s1_n:,}", flush=True)

# Load S2 full (need for background distractors)
con.execute(f"""
CREATE OR REPLACE TABLE src_raw AS
SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true);
""")
con.execute(f"CREATE OR REPLACE TABLE s2_prep AS {PREP_SELECT};")
s2_n = con.execute("SELECT count(*) FROM s2_prep").fetchone()[0]

# Load S3 full
con.execute(f"""
CREATE OR REPLACE TABLE src_raw AS
SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true);
""")
con.execute(f"CREATE OR REPLACE TABLE s3_prep AS {PREP_SELECT};")
s3_n = con.execute("SELECT count(*) FROM s3_prep").fetchone()[0]
print(f"  S2: {s2_n:,}  S3: {s3_n:,}", flush=True)

# Sample background distractors (exclude true targets)
all_s2_ids = [r[0] for r in con.execute("SELECT entity_id FROM s2_prep").fetchall()]
all_s3_ids = [r[0] for r in con.execute("SELECT entity_id FROM s3_prep").fetchall()]
bg_pool = [eid for eid in (all_s2_ids + all_s3_ids) if eid not in all_true_targets]
bg_sample = set(random.sample(bg_pool, min(N_BACKGROUND, len(bg_pool))))

# Build target pool = true targets + background
target_ids = all_true_targets | bg_sample
target_df = pd.DataFrame({"entity_id": list(target_ids)})
con.register("_tgt_ids", target_df)

con.execute("""
CREATE OR REPLACE TABLE target_prep AS
SELECT s.* FROM s2_prep s INNER JOIN _tgt_ids t ON s.entity_id = t.entity_id
UNION ALL
SELECT s.* FROM s3_prep s INNER JOIN _tgt_ids t ON s.entity_id = t.entity_id;
""")
tgt_n = con.execute("SELECT count(*) FROM target_prep").fetchone()[0]
print(f"  Target pool: {tgt_n:,} ({len(all_true_targets):,} true + {len(bg_sample):,} bg)", flush=True)
print(f"  Preprocessing done in {time.time()-t0:.1f}s", flush=True)

# -----------------------------------------------------------------------
# STEP 4: Chunked 8-pass blocking + similarity-sorted K=25 pruning
# -----------------------------------------------------------------------
print("\n" + "=" * 65, flush=True)
print("  STEP 4 — CHUNKED BLOCKING + K=25 PRUNING", flush=True)
print("=" * 65, flush=True)
t0 = time.time()

# Pre-load info for similarity scoring
s1_info = {r[0]: (r[1] or "", r[2] or "")
           for r in con.execute("SELECT entity_id, norm_name, norm_addr FROM s1_prep").fetchall()}
target_info = {r[0]: (r[1] or "", r[2] or "")
               for r in con.execute("SELECT entity_id, norm_name, norm_addr FROM target_prep").fetchall()}

all_s1_ids = list(all_selected_s1 & set(s1_info.keys()))
n_chunks = (len(all_s1_ids) + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
print(f"  Blocking {len(all_s1_ids):,} S1 in {n_chunks} chunks of {S1_CHUNK_SIZE:,}", flush=True)

candidate_map: dict = defaultdict(set)   # s1_id -> set of target_ids

for ci in range(n_chunks):
    chunk_ids = all_s1_ids[ci * S1_CHUNK_SIZE: (ci + 1) * S1_CHUNK_SIZE]
    chunk_df = pd.DataFrame({"entity_id": chunk_ids})
    con.register("_chunk", chunk_df)
    con.execute("""
    CREATE OR REPLACE TABLE chunk_s1 AS
    SELECT * FROM s1_prep WHERE entity_id IN (SELECT entity_id FROM _chunk);
    """)
    rows = con.execute(f"""
    SELECT DISTINCT s1_id, target_id FROM ({_8pass_sql('chunk_s1','target_prep')}) sub
    """).fetchall()
    for s1_id, tgt_id in rows:
        candidate_map[s1_id].add(tgt_id)
    if (ci + 1) % 10 == 0 or ci == 0 or ci == n_chunks - 1:
        print(f"  Chunk {ci+1}/{n_chunks}: {len(rows):,} pairs", flush=True)

con.execute("DROP TABLE IF EXISTS chunk_s1;")
print(f"  Blocking done in {time.time()-t0:.1f}s  |  Total raw pairs: {sum(len(v) for v in candidate_map.values()):,}", flush=True)

# Similarity-sort and prune to K=25
print(f"  Similarity-sorting + pruning to K={K_BUDGET}...", flush=True)
pruned_map: dict = {}   # s1_id -> [target_id, ...] (length <= K)

for s1_id, cand_set in candidate_map.items():
    if not cand_set:
        pruned_map[s1_id] = []
        continue
    s1_name_tok = set((s1_info.get(s1_id, ("",""))[0] or "").split())
    s1_addr_tok = set((s1_info.get(s1_id, ("",""))[1] or "").split())
    scored = []
    for tgt_id in cand_set:
        tgt_name_tok = set((target_info.get(tgt_id, ("",""))[0] or "").split())
        tgt_addr_tok = set((target_info.get(tgt_id, ("",""))[1] or "").split())
        n_union = len(s1_name_tok | tgt_name_tok)
        a_union = len(s1_addr_tok | tgt_addr_tok)
        n_sc = len(s1_name_tok & tgt_name_tok) / n_union if n_union else 0.0
        a_sc = len(s1_addr_tok & tgt_addr_tok) / a_union if a_union else 0.0
        scored.append((W_NAME * n_sc + W_ADDR * a_sc, tgt_id))
    scored.sort(key=lambda x: -x[0])
    pruned_map[s1_id] = [tid for _, tid in scored[:K_BUDGET]]

for s1_id in all_s1_ids:
    if s1_id not in pruned_map:
        pruned_map[s1_id] = []

total_cands = sum(len(v) for v in pruned_map.values())
print(f"  After K-pruning: {total_cands:,} pairs  avg={total_cands/len(all_s1_ids):.1f}/S1", flush=True)

# -----------------------------------------------------------------------
# STEP 5: Build labeled training pairs
# -----------------------------------------------------------------------
print("\n" + "=" * 65, flush=True)
print("  STEP 5 — BUILD LABELED PAIRS", flush=True)
print("=" * 65, flush=True)

train_pairs = []   # (s1_id, tgt_id, label, split)
val_pairs   = []

for s1_id, tgt_ids in pruned_map.items():
    true_set = gt_pairs.get(s1_id, set())
    split = "val" if s1_id in val_s1_ids else "train"
    for tgt_id in tgt_ids:
        label = 1 if tgt_id in true_set else 0
        (val_pairs if split == "val" else train_pairs).append(
            (s1_id, tgt_id, label)
        )
    # Also inject any true targets missed by blocking (recall ceiling)
    for tgt_id in true_set:
        if tgt_id not in set(tgt_ids) and tgt_id in target_info:
            (val_pairs if split == "val" else train_pairs).append(
                (s1_id, tgt_id, 1)
            )

pos_train = sum(1 for _, _, l in train_pairs if l == 1)
neg_train = sum(1 for _, _, l in train_pairs if l == 0)
pos_val   = sum(1 for _, _, l in val_pairs   if l == 1)
print(f"  Train pairs: {len(train_pairs):,}  (pos={pos_train:,} neg={neg_train:,})", flush=True)
print(f"  Val pairs:   {len(val_pairs):,}  (pos={pos_val:,})", flush=True)

# Balance: cap negatives at MAX_NEG_RATIO × positives
random.shuffle(train_pairs)
pos_list = [(s, t, l) for s, t, l in train_pairs if l == 1]
neg_list = [(s, t, l) for s, t, l in train_pairs if l == 0]
neg_list = neg_list[:MAX_NEG_RATIO * len(pos_list)]
train_pairs_balanced = pos_list + neg_list
random.shuffle(train_pairs_balanced)
print(f"  Balanced train: {len(train_pairs_balanced):,} pairs", flush=True)

# -----------------------------------------------------------------------
# STEP 6: Extract RapidFuzz features
# -----------------------------------------------------------------------
print("\n" + "=" * 65, flush=True)
print("  STEP 6 — FEATURE EXTRACTION", flush=True)
print("=" * 65, flush=True)
t0 = time.time()


def extract_features(pairs):
    n = len(pairs)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty(n, dtype=np.int32)
    for i, (s1_id, tgt_id, label) in enumerate(pairs):
        s1_name, s1_addr = s1_info.get(s1_id, ("", ""))
        t_name,  t_addr  = target_info.get(tgt_id, ("", ""))
        X[i] = compute_pair_features(
            s1_name, s1_name, s1_addr, "", "",
            t_name,  t_name,  t_addr,  "", ""
        )
        y[i] = label
        if (i + 1) % 200_000 == 0:
            print(f"    {i+1:,}/{n:,} features extracted", flush=True)
    return X, y


X_train, y_train = extract_features(train_pairs_balanced)
X_val,   y_val   = extract_features(val_pairs)
print(f"  Features done in {time.time()-t0:.1f}s  "
      f"X_train={X_train.shape}  X_val={X_val.shape}", flush=True)

# -----------------------------------------------------------------------
# STEP 7: Train XGBoost
# -----------------------------------------------------------------------
print("\n" + "=" * 65, flush=True)
print("  STEP 7 — TRAIN XGBOOST", flush=True)
print("=" * 65, flush=True)

clf = train_matching_model(X_train, y_train, X_val, y_val)

# -----------------------------------------------------------------------
# STEP 8: Tune F0.5 threshold on validation set
# -----------------------------------------------------------------------
print("\n" + "=" * 65, flush=True)
print("  STEP 8 — TUNE F0.5 THRESHOLD", flush=True)
print("=" * 65, flush=True)

val_probs = clf.predict_proba(X_val)[:, 1]

# Build per-entity candidate score lists
val_cand_scores = defaultdict(list)
for (s1_id, tgt_id, _), prob in zip(val_pairs, val_probs):
    val_cand_scores[s1_id].append((tgt_id, float(prob)))

# Build ground truth map for val entities
val_gt: dict = defaultdict(set)
for s1_id in val_s1_ids:
    val_gt[s1_id] = gt_pairs.get(s1_id, set())

thresholds = [round(x, 2) for x in np.arange(0.40, 0.92, 0.04)]
best_tau, best_f05 = find_optimal_threshold(val_gt, val_cand_scores, thresholds=thresholds)

print("\n" + "=" * 65, flush=True)
print(f"  TRAINING COMPLETE!", flush=True)
print(f"  Optimal tau*  : {best_tau:.2f}", flush=True)
print(f"  Val Macro F0.5: {best_f05:.4f}", flush=True)
print(f"  Model saved   : {MODEL_PATH}", flush=True)
print("=" * 65, flush=True)
