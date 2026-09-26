import duckdb
import os

base_dir = r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
s1_path = os.path.join(base_dir, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(base_dir, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(base_dir, "train_source3.tsv").replace("\\", "/")
gt_path = os.path.join(base_dir, "train_ground_truth.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("SET memory_limit = '8GB';")
con.execute("SET threads = 14;")

con.execute(f"""
CREATE TABLE gt_sample AS
SELECT 
    source1_entity_id,
    UNNEST(string_split(matched_entity_ids, ',')) AS target_id
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
LIMIT 5000;
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
WHERE entity_id IN (SELECT target_id FROM gt_sample);
""")

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

con.execute("""
CREATE TABLE candidates AS
SELECT DISTINCT
    s.entity_id AS s1_id,
    t.entity_id AS target_id
FROM s1_prep s
JOIN target_prep t ON s.country = t.country
WHERE 
    (length(s.p2) >= 4 AND s.p2 = t.p2)
    OR
    (length(s.p1) >= 3 AND s.p1 = t.p1 AND s.postal != '' AND s.postal = t.postal)
    OR
    (length(s.p1) >= 3 AND s.p1 = t.p1 AND s.door != '' AND s.door = t.door)
    OR
    (s.postal != '' AND s.postal = t.postal AND s.door != '' AND s.door = t.door AND length(s.p1) >= 3 AND length(t.p1) >= 3 AND s.p1[1:3] = t.p1[1:3])
    OR
    (length(s.p1) >= 6 AND s.p1 = t.p1);
""")

# Inspect missed matches
missed = con.execute("""
SELECT 
    s.entity_id AS s1_id,
    s.business_name AS s1_name,
    s.business_address AS s1_addr,
    t.entity_id AS target_id,
    t.business_name AS target_name,
    t.business_address AS target_addr
FROM gt_sample g
JOIN s1_prep s ON g.source1_entity_id = s.entity_id
JOIN target_prep t ON g.target_id = t.entity_id
WHERE NOT EXISTS (
    SELECT 1 FROM candidates c WHERE c.s1_id = g.source1_entity_id AND c.target_id = g.target_id
)
LIMIT 10;
""").df()

for idx, r in missed.iterrows():
    print(f"\n--- Missed #{idx+1} ---")
    print(f"S1: [{r['s1_id']}] '{r['s1_name']}' | '{r['s1_addr']}'")
    print(f"Tg: [{r['target_id']}] '{r['target_name']}' | '{r['target_addr']}'")
