import duckdb
import os
import time

base_dir = r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
s1_path = os.path.join(base_dir, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(base_dir, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(base_dir, "train_source3.tsv").replace("\\", "/")
gt_path = os.path.join(base_dir, "train_ground_truth.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("SET memory_limit = '8GB';")
con.execute("SET threads = 14;")

print("Loading Ground Truth sample...")
con.execute(f"""
CREATE TABLE gt_sample AS
SELECT 
    source1_entity_id,
    UNNEST(string_split(matched_entity_ids, ',')) AS target_id
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
LIMIT 20000;
""")
n_gt = con.execute("SELECT count(*) FROM gt_sample;").fetchone()[0]

con.execute(f"""
CREATE TABLE s1_sample AS
SELECT * FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT DISTINCT source1_entity_id FROM gt_sample);
""")

# Load Target sample: true targets + background records from S2 and S3
t0 = time.time()
print("Scanning targets from S2 and S3...")
con.execute(f"""
CREATE TABLE target_sample AS
SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT target_id FROM gt_sample)
UNION ALL
SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT target_id FROM gt_sample)
UNION ALL
(SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true) LIMIT 100000)
UNION ALL
(SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true) LIMIT 100000);
""")
print(f"Target sample loaded in {time.time()-t0:.2f}s, count: {con.execute('SELECT count(*) FROM target_sample;').fetchone()[0]:,}")

con.execute("""
CREATE MACRO norm_text(str) AS (
    lower(regexp_replace(regexp_replace(coalesce(str, ''), '[^a-zA-Z0-9 ]', ' ', 'g'), ' +', ' ', 'g'))
);

CREATE MACRO clean_legal(str) AS (
    regexp_replace(norm_text(str), '\\b(private limited|pvt ltd|pvt limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|company|limited|llc|inc|corp|ltd|llp|sarl|sas|sa|sci|eurl)\\b', '', 'g')
);

CREATE MACRO get_token(str, idx) AS (
    string_split(trim(norm_text(str)), ' ')[idx]
);
""")

con.execute("""
CREATE TABLE s1_prep AS
SELECT 
    entity_id,
    country,
    business_name,
    business_address,
    norm_text(business_name) AS norm_name,
    trim(clean_legal(business_name)) AS core_name,
    get_token(clean_legal(business_name), 1) AS p1,
    CASE WHEN length(get_token(clean_legal(business_name), 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END AS p2,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door
FROM s1_sample;
""")

con.execute("""
CREATE TABLE target_prep AS
SELECT 
    entity_id,
    country,
    business_name,
    business_address,
    norm_text(business_name) AS norm_name,
    trim(clean_legal(business_name)) AS core_name,
    get_token(clean_legal(business_name), 1) AS p1,
    CASE WHEN length(get_token(clean_legal(business_name), 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END AS p2,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door
FROM target_sample;
""")

print("Testing Multi-Key Blocking Recall...")
t1 = time.time()
# Join on multiple blocking rules
con.execute("""
CREATE TABLE candidates AS
SELECT DISTINCT
    s.entity_id AS s1_id,
    t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country
WHERE 
    -- Rule 1: Exact 2-word core prefix match (very high precision)
    (length(s.p2) >= 4 AND s.p2 = t.p2)
    OR
    -- Rule 2: 1-word core prefix match + Postal code match
    (length(s.p1) >= 3 AND s.p1 = t.p1 AND s.postal != '' AND s.postal = t.postal)
    OR
    -- Rule 3: 1-word core prefix match + Door number match
    (length(s.p1) >= 3 AND s.p1 = t.p1 AND s.door != '' AND s.door = t.door)
    OR
    -- Rule 4: Postal code match + Door number match (same address)
    (s.postal != '' AND s.postal = t.postal AND s.door != '' AND s.door = t.door AND length(s.p1) >= 3 AND length(t.p1) >= 3 AND s.p1[1:3] = t.p1[1:3])
    OR
    -- Rule 5: 1-word core prefix match if word is long (>= 6 chars)
    (length(s.p1) >= 6 AND s.p1 = t.p1);
""")
cand_time = time.time() - t1
n_cand = con.execute("SELECT count(*) FROM candidates;").fetchone()[0]
print(f"Candidate generation took {cand_time:.2f}s, produced {n_cand:,} candidate pairs")

# Measure Recall against Ground Truth
hits = con.execute("""
SELECT count(*) 
FROM gt_sample g
JOIN candidates c ON g.source1_entity_id = c.s1_id AND g.target_id = c.target_id;
""").fetchone()[0]

recall = hits / n_gt if n_gt > 0 else 0
print(f"Recall: {recall*100:.2f}% ({hits:,}/{n_gt:,})")
avg_cands = con.execute("SELECT avg(cnt) FROM (SELECT count(*) as cnt FROM candidates GROUP BY s1_id);").fetchone()[0]
print(f"Average candidates per S1 entity: {avg_cands:.2f}")
