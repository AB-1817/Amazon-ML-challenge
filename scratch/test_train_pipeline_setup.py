"""
Testing 8-pass blocking and candidate generation for 150k S1 entities.
"""
import os
import sys
import time
import json
import duckdb
import numpy as np
import pandas as pd
from collections import defaultdict

t_start = time.time()

TRAIN_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv").replace("\\", "/")
gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv").replace("\\", "/")

# DuckDB connection on disk to allow smooth disk spilling without OOM
db_path = "scratch/colab_pipeline.duckdb"
if os.path.exists(db_path):
    try: os.remove(db_path)
    except: pass

con = duckdb.connect(db_path)
con.execute("SET memory_limit='6GB';")
con.execute("SET threads=8;")
con.execute("SET preserve_insertion_order=false;")

print("[1] Registering preprocessing macros...", flush=True)
con.execute("""
CREATE OR REPLACE MACRO norm_text(s) AS
    lower(trim(regexp_replace(regexp_replace(
        regexp_replace(coalesce(s,''), 'https?://(?:www\\.)?|www\\.', '', 'g'),
        '\\.(?:com|org|net|in|co|gov|edu|fr|io)\\b', ' ', 'g'),
        '[^\\w\\s]', ' ', 'g')));

CREATE OR REPLACE MACRO clean_legal(s) AS
    trim(regexp_replace(norm_text(s),
    '\\b(?:private\\s+limited|pvt\\s+ltd|pvt\\s+limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|holdings|holding|services|service|company|limited|llc|inc|corp|ltd|llp|co|sarl|sas|sa|sci|eurl|sasu)\\b',
    ' ', 'g'));

CREATE OR REPLACE MACRO compact_name(s) AS
    regexp_replace(norm_text(s), '\\s+', '', 'g');

CREATE OR REPLACE MACRO get_token(s, n) AS
    CASE WHEN array_length(string_split(trim(coalesce(s,'')), ' ')) >= n
         THEN string_split(trim(coalesce(s,'')), ' ')[n]
         ELSE '' END;

CREATE OR REPLACE MACRO clean_addr(s) AS
    lower(trim(regexp_replace(
        regexp_replace(coalesce(s,''), '[^\\w\\s]', ' ', 'g'),
        '\\s+', ' ', 'g')));
""")

print("[2] Sampling 150,000 S1 entities with seed 42...", flush=True)
con.execute(f"""
CREATE TABLE s1_sample AS
SELECT 
    entity_id,
    coalesce(country, '') as country,
    business_name as raw_name,
    business_address as raw_addr,
    norm_text(business_name) as norm_name,
    clean_legal(business_name) as core_name,
    clean_addr(business_address) as norm_addr,
    get_token(clean_legal(business_name), 1) as p1,
    get_token(clean_legal(business_name), 2) as p2,
    get_token(clean_legal(business_name), 3) as p3,
    left(compact_name(business_name), 4) as cmp4,
    left(compact_name(business_name), 5) as cmp5,
    regexp_extract(clean_addr(business_address), '\\b(\\d{{3,}})\\b', 1) as postal,
    regexp_extract(clean_addr(business_address), '^(\\d+)', 1) as door,
    get_token(clean_addr(business_address), 1) as a1,
    get_token(clean_addr(business_address), 2) as a2
FROM read_csv_auto('{s1_path}', delim='\t', header=true, quote='')
USING SAMPLE 150000 (reservoir, 42);
""")

s1_count = con.execute("SELECT count(*) FROM s1_sample").fetchone()[0]
print(f"Sampled {s1_count:,} S1 entities. Countries:", flush=True)
country_dist = con.execute("SELECT country, count(*) FROM s1_sample GROUP BY country").fetchall()
for c, cnt in country_dist:
    print(f"  - {c}: {cnt:,} ({cnt/s1_count*100:.1f}%)", flush=True)

print("[3] Loading Ground Truth for S1 sample...", flush=True)
con.execute(f"""
CREATE TABLE gt_sample AS
SELECT 
    s.entity_id as s1_id,
    g.matched_entity_ids
FROM s1_sample s
LEFT JOIN read_csv_auto('{gt_path}', delim='\t', header=true, quote='') g
ON s.entity_id = g.source1_entity_id;
""")

gt_stats = con.execute("""
SELECT 
    count(*) as total,
    sum(case when matched_entity_ids is not null and trim(matched_entity_ids) != '' then 1 else 0 end) as with_match,
    sum(case when matched_entity_ids is null or trim(matched_entity_ids) = '' then 1 else 0 end) as singletons
FROM gt_sample;
""").fetchone()
print(f"GT: {gt_stats[0]:,} S1 entities | {gt_stats[1]:,} with matches | {gt_stats[2]:,} singletons ({gt_stats[2]/gt_stats[0]*100:.2f}%)", flush=True)

# Build set of true targets needed
con.execute("""
CREATE TABLE true_target_ids AS
SELECT DISTINCT unnest(string_split(matched_entity_ids, ',')) as target_id
FROM gt_sample
WHERE matched_entity_ids is not null and trim(matched_entity_ids) != '';
""")
true_tgt_count = con.execute("SELECT count(*) FROM true_target_ids").fetchone()[0]
print(f"True target entities to retrieve: {true_tgt_count:,}", flush=True)

# Create train/val S1 split (80% train, 20% validation)
print("[4] Splitting S1 entities: 80% train / 20% validation...", flush=True)
con.execute("""
CREATE TABLE s1_split AS
SELECT 
    entity_id,
    country,
    norm_name, core_name, norm_addr, raw_name, raw_addr,
    p1, p2, p3, cmp4, cmp5, postal, door, a1, a2,
    case when row_number() over (partition by country order by hash(entity_id)) <= count(*) over (partition by country) * 0.8
         then 'train' else 'val' end as split
FROM s1_sample;
""")
split_counts = con.execute("SELECT split, country, count(*) FROM s1_split GROUP BY split, country ORDER BY split, country").fetchall()
for sp, co, ct in split_counts:
    print(f"  {sp.upper()} | {co}: {ct:,}", flush=True)

print(f"Done setup in {time.time()-t_start:.2f}s", flush=True)
