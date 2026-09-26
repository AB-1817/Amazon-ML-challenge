import os
import sys
import json
import time
import duckdb
import numpy as np
import polars as pl
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RESOURCE_DIR = os.path.join(ROOT_DIR, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR = os.path.join(RESOURCE_DIR, "dataset", "train")

gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv").replace("\\", "/")
s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("SET memory_limit = '6GB';")
con.execute("SET threads = 14;")

print("=== 7. EXPERIMENTAL BLOCKING FEASIBILITY AUDIT ===", flush=True)

# 1. Setup Ground Truth evaluation slice: 10,000 S1 entities
print("Loading 10,000 evaluation S1 entities and true links...", flush=True)
con.execute(f"""
CREATE TABLE eval_gt_raw AS
SELECT source1_entity_id as s1_id, matched_entity_ids
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) != ''
LIMIT 10000;

CREATE TABLE eval_gt_links AS
SELECT s1_id, UNNEST(string_split(matched_entity_ids, ',')) as target_id
FROM eval_gt_raw;
""")

total_true_links = con.execute("SELECT count(*) FROM eval_gt_links;").fetchone()[0]
total_s1 = con.execute("SELECT count(*) FROM eval_gt_raw;").fetchone()[0]
print(f"Eval set: {total_s1:,} S1 entities with {total_true_links:,} true matches.", flush=True)

# 2. Setup S1 records
con.execute(f"""
CREATE TABLE eval_s1 AS
SELECT * FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT s1_id FROM eval_gt_raw);
""")

# 3. Setup Target pool: ALL true targets for eval_s1 + 200,000 background distractors from S2 and S3
print("Loading Target pool (all true targets + 200,000 background distractors)...", flush=True)
con.execute(f"""
CREATE TABLE eval_targets AS
SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT target_id FROM eval_gt_links)
UNION ALL
SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT target_id FROM eval_gt_links)
UNION ALL
(SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true) LIMIT 100000)
UNION ALL
(SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true) LIMIT 100000);
""")
total_targets = con.execute("SELECT count(*) FROM eval_targets;").fetchone()[0]
print(f"Target pool size: {total_targets:,} records.", flush=True)

# Total Cartesian comparison space for reduction ratio calculation
cartesian_space = total_s1 * total_targets
print(f"Cartesian space: {cartesian_space:,} pairs.", flush=True)

# Cleaning Macros
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

# Preprocess S1 and Targets
con.execute("""
CREATE TABLE s1_prep AS
SELECT 
    entity_id,
    country,
    norm_text(business_name) as norm_name,
    clean_legal(business_name) as core_name,
    norm_text(business_address) as norm_addr,
    get_token(clean_legal(business_name), 1) as p1,
    CASE WHEN length(get_token(clean_legal(business_name), 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END as p2,
    substring(compact_name(business_name), 1, 6) as cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') as postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') as door,
    get_token(clean_addr(business_address), 1) as a1
FROM eval_s1;

CREATE TABLE target_prep AS
SELECT 
    entity_id,
    country,
    norm_text(business_name) as norm_name,
    clean_legal(business_name) as core_name,
    norm_text(business_address) as norm_addr,
    get_token(clean_legal(business_name), 1) as p1,
    CASE WHEN length(get_token(clean_legal(business_name), 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END as p2,
    substring(compact_name(business_name), 1, 6) as cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') as postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') as door,
    get_token(clean_addr(business_address), 1) as a1
FROM eval_targets;
""")

strategies = {
    "1. Exact Normalized Name": """
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.norm_name = t.norm_name
        WHERE length(s.norm_name) >= 3
    """,
    
    "2. First Significant Name Token (p1)": """
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1
        WHERE length(s.p1) >= 4
    """,
    
    "3. Two-Token Name Prefix (p2)": """
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.p2 = t.p2
        WHERE length(s.p2) >= 4
    """,
    
    "4. Address Anchor (Door + Postal)": """
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.door = t.door
        WHERE s.postal != '' AND length(s.door) >= 2
    """,
    
    "5. Address Anchor (Door + Addr Token)": """
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.door = t.door AND s.a1 = t.a1
        WHERE length(s.door) >= 2 AND length(s.a1) >= 4
    """,
    
    "6. Compact Name Prefix (cmp6)": """
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.cmp6 = t.cmp6
        WHERE length(s.cmp6) >= 5
    """,
    
    "7. Combined Multi-Pass Blocking": """
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.p2 = t.p2
        WHERE length(s.p2) >= 4
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1 AND s.postal = t.postal
        WHERE length(s.p1) >= 3 AND s.postal != ''
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1 AND s.door = t.door
        WHERE length(s.p1) >= 3 AND s.door != ''
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.cmp6 = t.cmp6
        WHERE length(s.cmp6) >= 5
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1
        WHERE length(s.p1) >= 5
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.door = t.door
        WHERE s.postal != '' AND length(s.door) >= 2
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.door = t.door AND s.a1 = t.a1
        WHERE length(s.door) >= 2 AND length(s.a1) >= 4
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND get_token(s.p2, 2) = t.p1
        WHERE length(get_token(s.p2, 2)) >= 4
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.p1 = get_token(t.p2, 2)
        WHERE length(s.p1) >= 4
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.cmp6[1:3] = t.cmp6[1:3]
        WHERE s.postal != '' AND length(s.cmp6) >= 3 AND length(t.cmp6) >= 3
        UNION
        SELECT s.entity_id as s1_id, t.entity_id as target_id
        FROM s1_prep s
        JOIN target_prep t ON s.country = t.country AND s.door = t.door AND s.cmp6[1:3] = t.cmp6[1:3]
        WHERE length(s.door) >= 2 AND length(s.cmp6) >= 3 AND length(t.cmp6) >= 3
    """
}

results = {}

for name, query in strategies.items():
    print(f"\n--- Testing Strategy: {name} ---", flush=True)
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TABLE current_cands AS {query};")
    elapsed = time.time() - t0
    
    total_cands = con.execute("SELECT count(*) FROM current_cands;").fetchone()[0]
    
    # Calculate recall against ground truth
    hits = con.execute("""
    SELECT count(*) 
    FROM eval_gt_links g
    JOIN current_cands c ON g.s1_id = c.s1_id AND g.target_id = c.target_id;
    """).fetchone()[0]
    recall = (hits / total_true_links) * 100
    
    # Reduction ratio
    reduction_ratio = (1.0 - (total_cands / cartesian_space)) * 100
    
    # Distribution of candidates per S1 entity
    cand_dist = con.execute("""
    SELECT 
        round(avg(c), 2) as avg_c,
        median(c) as med_c,
        quantile_cont(c, 0.95) as p95_c,
        quantile_cont(c, 0.99) as p99_c,
        max(c) as max_c
    FROM (
        SELECT s1_id, count(*) as c
        FROM current_cands
        GROUP BY s1_id
    );
    """).df().to_dict(orient='records')
    
    dist_stats = cand_dist[0] if cand_dist else {'avg_c': 0, 'med_c': 0, 'p95_c': 0, 'p99_c': 0, 'max_c': 0}
    
    print(f"  Time taken:        {elapsed:.2f}s")
    print(f"  Candidate pairs:   {total_cands:,}")
    print(f"  Candidate Recall:  {recall:.2f}% ({hits:,}/{total_true_links:,})")
    print(f"  Reduction Ratio:   {reduction_ratio:.6f}%")
    print(f"  Per-entity stats:  avg={dist_stats['avg_c']}, med={dist_stats['med_c']}, p95={dist_stats['p95_c']}, p99={dist_stats['p99_c']}, max={dist_stats['max_c']}")
    
    results[name] = {
        "recall_pct": recall,
        "hits": hits,
        "total_cands": total_cands,
        "reduction_ratio_pct": reduction_ratio,
        "dist": dist_stats
    }

# 4. Candidate Budget Analysis for the Combined Strategy
print("\n=== CANDIDATE BUDGET SENSITIVITY (COMBINED STRATEGY) ===", flush=True)
budgets = [10, 15, 25, 50, 100]
budget_results = {}

for b in budgets:
    # Rank candidates by a simple composite score: Jaccard on name tokens
    con.execute(f"""
    CREATE OR REPLACE TABLE budgeted_cands AS
    SELECT s1_id, target_id
    FROM (
        SELECT 
            s1_id, target_id,
            ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY target_id) as rnk
        FROM current_cands
    )
    WHERE rnk <= {b};
    """)
    
    b_hits = con.execute("""
    SELECT count(*) 
    FROM eval_gt_links g
    JOIN budgeted_cands c ON g.s1_id = c.s1_id AND g.target_id = c.target_id;
    """).fetchone()[0]
    b_recall = (b_hits / total_true_links) * 100
    b_total = con.execute("SELECT count(*) FROM budgeted_cands;").fetchone()[0]
    
    print(f"Budget K = {b:3d} -> Candidate Recall: {b_recall:.2f}% ({b_hits:,}/{total_true_links:,}) | Total pairs: {b_total:,}")
    budget_results[b] = {
        "recall_pct": b_recall,
        "hits": b_hits,
        "total_pairs": b_total
    }

out_file = os.path.join(os.path.dirname(__file__), "blocking_experiments_summary.json")
with open(out_file, "w") as f:
    json.dump({
        "strategies": results,
        "budgets": budget_results,
        "eval_s1": total_s1,
        "eval_true_links": total_true_links,
        "target_pool": total_targets
    }, f, indent=2)
print(f"\nSaved blocking feasibility audit to: {out_file}")
