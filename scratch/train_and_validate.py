import duckdb
import os
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\code\business_entity_resolution")
from src.config import TRAIN_DIR, NUM_WORKERS
from src.features import compute_pair_features, FEATURE_NAMES
from src.model import train_matching_model
from src.evaluate import find_optimal_threshold, evaluate_macro_f05

s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv").replace("\\", "/")
gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("SET memory_limit = '4GB';")
con.execute(f"SET threads = {NUM_WORKERS};")

print("1. Loading ground truth sample (25,000 entities)...", flush=True)
t0 = time.time()
con.execute(f"""
CREATE TABLE gt_sample AS
SELECT 
    source1_entity_id,
    UNNEST(string_split(matched_entity_ids, ',')) AS target_id
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
LIMIT 40000;
""")

con.execute(f"""
CREATE TABLE s1_sample AS
SELECT * FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT DISTINCT source1_entity_id FROM gt_sample);
""")

con.execute(f"""
CREATE TABLE target_sample AS
SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT target_id FROM gt_sample)
UNION ALL
SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT target_id FROM gt_sample)
UNION ALL
(SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true) LIMIT 60000)
UNION ALL
(SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true) LIMIT 60000);
""")

print(f"Data loaded in {time.time()-t0:.2f}s. S1 count: {con.execute('SELECT count(*) FROM s1_sample').fetchone()[0]:,}, Target pool: {con.execute('SELECT count(*) FROM target_sample').fetchone()[0]:,}", flush=True)

con.execute("""
CREATE MACRO norm_text(str) AS (
    lower(regexp_replace(regexp_replace(coalesce(str, ''), '[^a-zA-Z0-9 ]', ' ', 'g'), ' +', ' ', 'g'))
);

CREATE MACRO strip_domain(str) AS (
    regexp_replace(norm_text(str), '\\b(com|org|net|in|co|gov|edu|fr|io)\\b', '', 'g')
);

CREATE MACRO clean_legal(str) AS (
    trim(regexp_replace(strip_domain(str), '\\b(private limited|pvt ltd|pvt limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|company|limited|llc|inc|corp|ltd|llp|sarl|sas|sa|sci|eurl|the|ms|smt|shree|sri|dr|pvt)\\b', '', 'g'))
);

CREATE MACRO compact_name(str) AS (
    regexp_replace(clean_legal(str), ' ', '', 'g')
);

CREATE MACRO get_token(str, idx) AS (
    string_split(trim(str), ' ')[idx]
);

CREATE MACRO clean_addr(str) AS (
    trim(regexp_replace(norm_text(str), '\\b(road|street|drive|avenue|lane|near|opp|opposite|floor|building|shop|plot|nagar|colony|block|west|east|north|south)\\b', '', 'g'))
);
""")

con.execute("""
CREATE TABLE s1_prep AS
SELECT 
    entity_id,
    country,
    norm_text(business_name) AS norm_name,
    clean_legal(business_name) AS core_name,
    norm_text(business_address) AS norm_addr,
    get_token(clean_legal(business_name), 1) AS p1,
    CASE WHEN length(get_token(clean_legal(business_name), 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END AS p2,
    substring(compact_name(business_name), 1, 6) AS cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door,
    get_token(clean_addr(business_address), 1) AS a1
FROM s1_sample;
""")

con.execute("""
CREATE TABLE target_prep AS
SELECT 
    entity_id,
    country,
    norm_text(business_name) AS norm_name,
    clean_legal(business_name) AS core_name,
    norm_text(business_address) AS norm_addr,
    get_token(clean_legal(business_name), 1) AS p1,
    CASE WHEN length(get_token(clean_legal(business_name), 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END AS p2,
    substring(compact_name(business_name), 1, 6) AS cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door,
    get_token(clean_addr(business_address), 1) AS a1
FROM target_sample;
""")

# Split S1 into Train (80%) and Val (20%)
con.execute("""
CREATE TABLE s1_val_ids AS SELECT DISTINCT entity_id FROM s1_prep ORDER BY entity_id LIMIT 2500;
CREATE TABLE s1_train_ids AS SELECT DISTINCT entity_id FROM s1_prep WHERE entity_id NOT IN (SELECT entity_id FROM s1_val_ids);
""")

print("2. Generating candidate pairs via blocking passes...", flush=True)
from src.blocking import run_blocking
run_blocking(con, "s1_prep", "target_prep", "candidates_all")

print("3. Building balanced feature dataset...", flush=True)
con.execute("""
CREATE TABLE candidate_features AS
SELECT 
    c.s1_id,
    c.target_id,
    CASE WHEN g.target_id IS NOT NULL THEN 1 ELSE 0 END AS label,
    s.norm_name AS s1_name,
    s.core_name AS s1_core,
    s.norm_addr AS s1_addr,
    s.postal AS s1_postal,
    s.door AS s1_door,
    t.norm_name AS t_name,
    t.core_name AS t_core,
    t.norm_addr AS t_addr,
    t.postal AS t_postal,
    t.door AS t_door,
    CASE WHEN c.s1_id IN (SELECT entity_id FROM s1_val_ids) THEN 'val' ELSE 'train' END AS split
FROM candidates_all c
LEFT JOIN gt_sample g ON c.s1_id = g.source1_entity_id AND c.target_id = g.target_id
JOIN s1_prep s ON c.s1_id = s.entity_id
JOIN target_prep t ON c.target_id = t.entity_id;
""")

# Sample to keep memory light and training fast
# Keep ALL positive pairs + up to 3x hard negatives for train
con.execute("""
CREATE TABLE train_samples AS
SELECT * FROM candidate_features WHERE split = 'train' AND label = 1
UNION ALL
(SELECT * FROM candidate_features WHERE split = 'train' AND label = 0 USING SAMPLE 80000);
""")

# For val, keep all candidates for the 2500 validation entities (up to 50k pairs)
con.execute("""
CREATE TABLE val_samples AS
SELECT * FROM candidate_features WHERE split = 'val' LIMIT 60000;
""")

print("Training samples count:", con.execute("SELECT count(*) FROM train_samples").fetchone()[0], flush=True)
print("Validation samples count:", con.execute("SELECT count(*) FROM val_samples").fetchone()[0], flush=True)

df_train = con.execute("SELECT * FROM train_samples").df()
df_val = con.execute("SELECT * FROM val_samples").df()

print("4. Calculating rapidfuzz similarity features in Python...", flush=True)
t_feat = time.time()

def extract_features(df):
    X = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    s1_names = df["s1_name"].values
    s1_cores = df["s1_core"].values
    s1_addrs = df["s1_addr"].values
    s1_postals = df["s1_postal"].values
    s1_doors = df["s1_door"].values
    t_names = df["t_name"].values
    t_cores = df["t_core"].values
    t_addrs = df["t_addr"].values
    t_postals = df["t_postal"].values
    t_doors = df["t_door"].values
    
    for i in range(len(df)):
        X[i] = compute_pair_features(
            s1_names[i], s1_cores[i], s1_addrs[i], s1_postals[i], s1_doors[i],
            t_names[i], t_cores[i], t_addrs[i], t_postals[i], t_doors[i]
        )
    return X

X_train = extract_features(df_train)
y_train = df_train["label"].values
X_val = extract_features(df_val)
y_val = df_val["label"].values
print(f"Features extracted in {time.time()-t_feat:.2f}s. X_train: {X_train.shape}, X_val: {X_val.shape}", flush=True)

# Train XGBoost Model
print("\n5. Training XGBoost Model...", flush=True)
clf = train_matching_model(X_train, y_train, X_val, y_val)

# Run threshold tuning on validation set
print("\n6. Running Threshold Optimization for Macro F_0.5...", flush=True)
val_probs = clf.predict_proba(X_val)[:, 1]

val_cand_scores = defaultdict(list)
for s1_id, t_id, prob in zip(df_val["s1_id"], df_val["target_id"], val_probs):
    val_cand_scores[s1_id].append((t_id, float(prob)))

val_gt_df = con.execute("SELECT source1_entity_id, target_id FROM gt_sample WHERE source1_entity_id IN (SELECT entity_id FROM s1_val_ids);").df()
val_gt = defaultdict(set)
for _, r in val_gt_df.iterrows():
    val_gt[r["source1_entity_id"]].add(r["target_id"])

for s1_id in con.execute("SELECT entity_id FROM s1_val_ids;").df()["entity_id"]:
    if s1_id not in val_gt:
        val_gt[s1_id] = set()

best_tau, best_score = find_optimal_threshold(val_gt, val_cand_scores)
print(f"\n==========================================", flush=True)
print(f"VALIDATION MACRO F_0.5: {best_score:.4f} at tau* = {best_tau:.2f}", flush=True)
print(f"==========================================", flush=True)
