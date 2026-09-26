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
    string_split(trim(clean_legal(str)), ' ')[idx]
);
""")

# Create explicit equi-join blocking keys!
con.execute("""
CREATE TABLE s1_prep AS
SELECT 
    entity_id,
    country,
    get_token(business_name, 1) AS p1,
    CASE WHEN length(get_token(business_name, 2)) > 0 
         THEN get_token(business_name, 1) || ' ' || get_token(business_name, 2) 
         ELSE get_token(business_name, 1) END AS p2,
    substring(compact_name(business_name), 1, 5) AS cmp5,
    substring(compact_name(business_name), 1, 7) AS cmp7,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door
FROM s1_sample;
""")

con.execute("""
CREATE TABLE target_prep AS
SELECT 
    entity_id,
    country,
    get_token(business_name, 1) AS p1,
    CASE WHEN length(get_token(business_name, 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END AS p2,
    substring(compact_name(business_name), 1, 5) AS cmp5,
    substring(compact_name(business_name), 1, 7) AS cmp7,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door
FROM target_sample;
""")

print("Running fast equi-join blocking passes...")
t1 = time.time()
# Each pass is an explicit equi-join with UNION:
con.execute("""
CREATE TABLE candidates AS
-- Pass 1: Country + Exact 2-word core prefix (length >= 4)
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p2 = t.p2
WHERE length(s.p2) >= 4

UNION

-- Pass 2: Country + 1st word + Postal code (length p1 >= 3, postal != '')
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1 AND s.postal = t.postal
WHERE length(s.p1) >= 3 AND s.postal != ''

UNION

-- Pass 3: Country + 1st word + Door number (length p1 >= 3, door != '')
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1 AND s.door = t.door
WHERE length(s.p1) >= 3 AND s.door != ''

UNION

-- Pass 4: Country + Compact name 7-char prefix (handles merged domain names and no-space variations)
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.cmp7 = t.cmp7
WHERE length(s.cmp7) >= 6

UNION

-- Pass 5: Country + Door + Postal (same address) and same 1st letter of name
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.door = t.door AND s.p1[1:1] = t.p1[1:1]
WHERE s.postal != '' AND s.door != '' AND length(s.p1) >= 1

UNION

-- Pass 6: Long single word prefix match (length >= 5)
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1
WHERE length(s.p1) >= 5

UNION

-- Pass 7: Word order swap (e.g. Word1 Word2 vs Word2 Word1)
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country 
                  AND get_token(s.p2, 1) = get_token(t.p2, 2) 
                  AND get_token(s.p2, 2) = get_token(t.p2, 1)
WHERE length(get_token(s.p2, 1)) >= 3 AND length(get_token(s.p2, 2)) >= 3

UNION

-- Pass 8: Exact same address (Postal + Door)
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.door = t.door
WHERE s.postal != '' AND s.door != '';

""")

cand_time = time.time() - t1
n_cand = con.execute("SELECT count(*) FROM candidates;").fetchone()[0]
print(f"Candidate generation took {cand_time:.2f}s, produced {n_cand:,} candidate pairs")

hits = con.execute("""
SELECT count(*) 
FROM gt_sample g
JOIN candidates c ON g.source1_entity_id = c.s1_id AND g.target_id = c.target_id;
""").fetchone()[0]

recall = hits / n_gt if n_gt > 0 else 0
print(f"Recall: {recall*100:.2f}% ({hits:,}/{n_gt:,})")
avg_cands = con.execute("SELECT avg(cnt) FROM (SELECT count(*) as cnt FROM candidates GROUP BY s1_id);").fetchone()[0]
print(f"Average candidates per S1 entity: {avg_cands:.2f}")
