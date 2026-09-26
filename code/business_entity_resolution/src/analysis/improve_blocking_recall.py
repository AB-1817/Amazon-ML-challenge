import os
import sys
import time
import duckdb

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

print("=== IMPROVING BLOCKING RECALL: TARGETED EXPERIMENTS ===", flush=True)

# 1. Setup 10,000 S1 eval entities
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

con.execute(f"""
CREATE TABLE eval_s1 AS
SELECT * FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT s1_id FROM eval_gt_raw);
""")

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
cartesian_space = total_s1 * total_targets

# Macros
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
    norm_text(business_name) as norm_name,
    clean_legal(business_name) as core_name,
    norm_text(business_address) as norm_addr,
    get_token(clean_legal(business_name), 1) as p1,
    get_token(clean_legal(business_name), 2) as p2,
    get_token(clean_legal(business_name), 3) as p3,
    substring(compact_name(business_name), 1, 4) as cmp4,
    substring(compact_name(business_name), 1, 5) as cmp5,
    substring(compact_name(business_name), 1, 6) as cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') as postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') as door,
    get_token(clean_addr(business_address), 1) as a1,
    get_token(clean_addr(business_address), 2) as a2
FROM eval_s1;

CREATE TABLE target_prep AS
SELECT 
    entity_id,
    country,
    norm_text(business_name) as norm_name,
    clean_legal(business_name) as core_name,
    norm_text(business_address) as norm_addr,
    get_token(clean_legal(business_name), 1) as p1,
    get_token(clean_legal(business_name), 2) as p2,
    get_token(clean_legal(business_name), 3) as p3,
    substring(compact_name(business_name), 1, 4) as cmp4,
    substring(compact_name(business_name), 1, 5) as cmp5,
    substring(compact_name(business_name), 1, 6) as cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') as postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') as door,
    get_token(clean_addr(business_address), 1) as a1,
    get_token(clean_addr(business_address), 2) as a2
FROM eval_targets;
""")

# Test Enhanced Blocking Passes
print("Evaluating Enhanced Multi-Pass Blocking with Address & Cross-Token Coverage...", flush=True)
t0 = time.time()
con.execute("""
CREATE OR REPLACE TABLE candidates_enhanced AS
-- Pass 1: Country + 2-token prefix match (p1 + p2)
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1 AND s.p2 = t.p2
WHERE length(s.p1) >= 3 AND length(s.p2) >= 3

UNION

-- Pass 2: Country + 1st token + Postal code
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1 AND s.postal = t.postal
WHERE length(s.p1) >= 3 AND s.postal != ''

UNION

-- Pass 3: Country + 1st token + Door number
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1 AND s.door = t.door
WHERE length(s.p1) >= 3 AND s.door != ''

UNION

-- Pass 4: Compact name prefix 5-char (catches domain names, merged text, typos)
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.cmp5 = t.cmp5
WHERE length(s.cmp5) >= 5

UNION

-- Pass 5: Single long core name word (p1 >= 5)
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1
WHERE length(s.p1) >= 5

UNION

-- Pass 6: Address Anchor: Door + Postal code (identical building)
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.door = t.door
WHERE s.postal != '' AND length(s.door) >= 2

UNION

-- Pass 7: Address Anchor: Door + Street token 1 (catches Indic script & DBA name changes)
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.door = t.door AND s.a1 = t.a1
WHERE length(s.door) >= 2 AND length(s.a1) >= 4

UNION

-- Pass 8: Address Anchor: Door + Street token 2
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.door = t.door AND s.a2 = t.a2
WHERE length(s.door) >= 2 AND length(s.a2) >= 4

UNION

-- Pass 9: Word permutation: S1 p2 = Target p1 (e.g. 'Hotel Sheron' vs 'Sheron')
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p2 = t.p1
WHERE length(s.p2) >= 4

UNION

-- Pass 10: Word permutation: S1 p1 = Target p2
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p2
WHERE length(s.p1) >= 4

UNION

-- Pass 11: Word permutation: S1 p3 = Target p1
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p3 = t.p1
WHERE length(s.p3) >= 4

UNION

-- Pass 12: Postal match + 4-char compact name prefix
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.cmp4 = t.cmp4
WHERE s.postal != '' AND length(s.cmp4) >= 4

UNION

-- Pass 13: Door match + 4-char compact name prefix
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.door = t.door AND s.cmp4 = t.cmp4
WHERE length(s.door) >= 2 AND length(s.cmp4) >= 4;
""")
elapsed = time.time() - t0

total_cands = con.execute("SELECT count(*) FROM candidates_enhanced;").fetchone()[0]
hits = con.execute("""
SELECT count(*) 
FROM eval_gt_links g
JOIN candidates_enhanced c ON g.s1_id = c.s1_id AND g.target_id = c.target_id;
""").fetchone()[0]

recall = (hits / total_true_links) * 100
reduction_ratio = (1.0 - (total_cands / cartesian_space)) * 100

print(f"\n--- ENHANCED BLOCKING RESULTS ---")
print(f"Time Taken:        {elapsed:.2f} seconds")
print(f"Candidate Pairs:   {total_cands:,}")
print(f"Recall:            {recall:.2f}% ({hits:,}/{total_true_links:,})")
print(f"Reduction Ratio:   {reduction_ratio:.6f}%")

# Stats per S1
dist_stats = con.execute("""
SELECT 
    round(avg(c), 2) as avg_c,
    median(c) as med_c,
    quantile_cont(c, 0.95) as p95_c,
    quantile_cont(c, 0.99) as p99_c,
    max(c) as max_c
FROM (
    SELECT s1_id, count(*) as c
    FROM candidates_enhanced
    GROUP BY s1_id
);
""").df().to_dict(orient='records')[0]
print(f"Per-entity stats:  avg={dist_stats['avg_c']}, med={dist_stats['med_c']}, p95={dist_stats['p95_c']}, p99={dist_stats['p99_c']}, max={dist_stats['max_c']}")
