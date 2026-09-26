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
    string_split(trim(str), ' ')[idx]
);

-- Extract address tokens, skipping generic address words like 'road', 'street', 'near'
CREATE MACRO clean_addr(str) AS (
    trim(regexp_replace(norm_text(str), '\\b(road|street|drive|avenue|lane|near|opp|opposite|floor|building|shop|plot|nagar|colony|block|west|east|north|south)\\b', '', 'g'))
);
""")

con.execute("""
CREATE TABLE s1_prep AS
SELECT 
    entity_id,
    country,
    get_token(clean_legal(business_name), 1) AS p1,
    CASE WHEN length(get_token(clean_legal(business_name), 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END AS p2,
    substring(compact_name(business_name), 1, 6) AS cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door,
    get_token(clean_addr(business_address), 1) AS a1,
    get_token(clean_addr(business_address), 2) AS a2
FROM s1_sample;
""")

con.execute("""
CREATE TABLE target_prep AS
SELECT 
    entity_id,
    country,
    get_token(clean_legal(business_name), 1) AS p1,
    CASE WHEN length(get_token(clean_legal(business_name), 2)) > 0 
         THEN get_token(clean_legal(business_name), 1) || ' ' || get_token(clean_legal(business_name), 2) 
         ELSE get_token(clean_legal(business_name), 1) END AS p2,
    substring(compact_name(business_name), 1, 6) AS cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door,
    get_token(clean_addr(business_address), 1) AS a1,
    get_token(clean_addr(business_address), 2) AS a2
FROM target_sample;
""")

print("Running fast multi-key blocking passes (Name + Address Anchors)...")
t1 = time.time()
con.execute("""
CREATE TABLE candidates AS
-- Pass 1: Country + 2-word core prefix (length >= 4)
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

-- Pass 4: Country + Compact name 6-char prefix (handles concatenated domains)
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.cmp6 = t.cmp6
WHERE length(s.cmp6) >= 5

UNION

-- Pass 5: Single long core name word (length >= 5)
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = t.p1
WHERE length(s.p1) >= 5

UNION

-- Pass 6: Address Anchor: Door + Postal (length door >= 2, postal != '')
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.door = t.door
WHERE s.postal != '' AND length(s.door) >= 2

UNION

-- Pass 7: Address Anchor: Door + Significant Address Word (catches Indic script name changes!)
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.door = t.door AND s.a1 = t.a1
WHERE length(s.door) >= 2 AND length(s.a1) >= 4

UNION

-- Pass 8: 2nd word of S1 matches 1st word of Target (e.g. 'Sri Balaji' vs 'Balaji' or 'New York Pizza' vs 'York Pizza')
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND get_token(s.p2, 2) = t.p1
WHERE length(get_token(s.p2, 2)) >= 4

UNION

-- Pass 9: 1st word of S1 matches 2nd word of Target
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.p1 = get_token(t.p2, 2)
WHERE length(s.p1) >= 4

UNION

-- Pass 10: Postal code match + 3-char prefix of name
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.postal = t.postal AND s.cmp6[1:3] = t.cmp6[1:3]
WHERE s.postal != '' AND length(s.cmp6) >= 3 AND length(t.cmp6) >= 3

UNION

-- Pass 11: Door number match + 3-char prefix of name
SELECT s.entity_id AS s1_id, t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country AND s.door = t.door AND s.cmp6[1:3] = t.cmp6[1:3]
WHERE length(s.door) >= 2 AND length(s.cmp6) >= 3 AND length(t.cmp6) >= 3;

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
