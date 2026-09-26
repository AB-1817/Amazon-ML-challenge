import duckdb
import os
import sys
import time

base_dir = r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
s1_path = os.path.join(base_dir, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(base_dir, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(base_dir, "train_source3.tsv").replace("\\", "/")
gt_path = os.path.join(base_dir, "train_ground_truth.tsv").replace("\\", "/")

print("Connecting to DuckDB...")
con = duckdb.connect()

# Set memory limit to 8GB to be safe and use 14 threads
con.execute("SET memory_limit = '8GB';")
con.execute("SET threads = 14;")

t0 = time.time()
print("Loading Ground Truth sample (10,000 S1 records)...")
con.execute(f"""
CREATE TABLE gt_sample AS
SELECT 
    source1_entity_id,
    UNNEST(string_split(matched_entity_ids, ',')) AS target_id
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
LIMIT 25000;
""")
n_gt = con.execute("SELECT count(*) FROM gt_sample;").fetchone()[0]
print(f"Loaded {n_gt:,} true pairs in {time.time()-t0:.2f}s")

# Let's inspect the target IDs
con.execute(f"""
CREATE TABLE s1_sample AS
SELECT * FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT DISTINCT source1_entity_id FROM gt_sample);
""")

print("S1 sample count:", con.execute("SELECT count(*) FROM s1_sample;").fetchone()[0])

# Preprocess strings inside DuckDB using regex functions!
con.execute("""
CREATE MACRO norm_text(str) AS (
    lower(regexp_replace(regexp_replace(coalesce(str, ''), '[^a-zA-Z0-9 ]', ' ', 'g'), ' +', ' ', 'g'))
);

CREATE MACRO get_token(str, idx) AS (
    string_split(norm_text(str), ' ')[idx]
);
""")

# Test creating blocking keys for S1
con.execute("""
CREATE TABLE s1_keys AS
SELECT 
    entity_id,
    country,
    norm_text(business_name) AS norm_name,
    norm_text(business_address) AS norm_addr,
    get_token(business_name, 1) AS p1,
    CASE WHEN length(get_token(business_name, 2)) > 0 
         THEN get_token(business_name, 1) || ' ' || get_token(business_name, 2) 
         ELSE get_token(business_name, 1) END AS p2,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') AS postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') AS door
FROM s1_sample;
""")

print("Sample S1 keys:")
print(con.execute("SELECT entity_id, country, p1, p2, postal, door FROM s1_keys LIMIT 5;").df())
