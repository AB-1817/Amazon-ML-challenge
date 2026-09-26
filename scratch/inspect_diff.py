import duckdb
import os

base_dir = r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
s1_path = os.path.join(base_dir, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(base_dir, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(base_dir, "train_source3.tsv").replace("\\", "/")
gt_path = os.path.join(base_dir, "train_ground_truth.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute(f"""
CREATE TABLE gt_sample AS
SELECT 
    source1_entity_id,
    UNNEST(string_split(matched_entity_ids, ',')) AS target_id
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
LIMIT 2000;
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

CREATE MACRO strip_domain(str) AS (
    regexp_replace(norm_text(str), '\\b(com|org|net|in|co|gov|edu|fr|io)\\b', '', 'g')
);

CREATE MACRO clean_legal(str) AS (
    trim(regexp_replace(strip_domain(str), '\\b(private limited|pvt ltd|pvt limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|company|limited|llc|inc|corp|ltd|llp|sarl|sas|sa|sci|eurl|the|ms|smt|shree|sri|dr|pvt)\\b', '', 'g'))
);

CREATE MACRO get_token(str, idx) AS (
    string_split(trim(clean_legal(str)), ' ')[idx]
);

CREATE TABLE s1_prep AS
SELECT entity_id, country, business_name, business_address,
       get_token(business_name, 1) as p1,
       get_token(business_name, 2) as p2
FROM s1_sample;

CREATE TABLE target_prep AS
SELECT entity_id, country, business_name, business_address,
       get_token(business_name, 1) as p1,
       get_token(business_name, 2) as p2
FROM target_sample;
""")

diff = con.execute("""
SELECT s.business_name as s1_name, s.business_address as s1_addr,
       t.business_name as t_name, t.business_address as t_addr
FROM gt_sample g
JOIN s1_prep s ON g.source1_entity_id = s.entity_id
JOIN target_prep t ON g.target_id = t.entity_id
WHERE s.p1 != t.p1
LIMIT 10;
""").df()

for idx, r in diff.iterrows():
    s1_n = r["s1_name"].encode("ascii", "replace").decode("ascii")
    t_n = r["t_name"].encode("ascii", "replace").decode("ascii")
    s1_a = r["s1_addr"].encode("ascii", "replace").decode("ascii")
    t_a = r["t_addr"].encode("ascii", "replace").decode("ascii")
    print(f"S1: {s1_n} | {s1_a}")
    print(f"Tg: {t_n} | {t_a}")
    print("-" * 50)
