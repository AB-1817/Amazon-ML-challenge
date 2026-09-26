import os, time, duckdb
from collections import defaultdict

t0 = time.time()
con = duckdb.connect(':memory:')
con.execute("SET memory_limit='6GB'; SET threads=8; SET preserve_insertion_order=false;")

s1_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv'
s2_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv'
s3_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv'
gt_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv'

# Register macros
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

# Sample 10k S1
con.execute(f"""
CREATE TABLE s1_sample AS
SELECT 
    entity_id, coalesce(country,'') as country,
    business_name as raw_name, business_address as raw_addr,
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
USING SAMPLE 10000 (reservoir, 42);
""")

# True targets
con.execute(f"""
CREATE TABLE true_targets AS
SELECT DISTINCT unnest(string_split(g.matched_entity_ids, ',')) as target_id
FROM s1_sample s
JOIN read_csv_auto('{gt_path}', delim='\t', header=true, quote='') g
ON s.entity_id = g.source1_entity_id
WHERE g.matched_entity_ids is not null and trim(g.matched_entity_ids) != '';
""")

# Build target pool: true targets + 100k random distractors
con.execute(f"""
CREATE TABLE target_pool AS
SELECT 
    entity_id, coalesce(country,'') as country,
    business_name as raw_name, business_address as raw_addr,
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
FROM (
    SELECT * FROM read_csv_auto('{s2_path}', delim='\t', header=true, quote='')
    WHERE entity_id IN (SELECT target_id FROM true_targets)
    UNION ALL
    SELECT * FROM read_csv_auto('{s3_path}', delim='\t', header=true, quote='')
    WHERE entity_id IN (SELECT target_id FROM true_targets)
    UNION ALL
    (SELECT * FROM read_csv_auto('{s2_path}', delim='\t', header=true, quote='') USING SAMPLE 50000 (reservoir, 42))
    UNION ALL
    (SELECT * FROM read_csv_auto('{s3_path}', delim='\t', header=true, quote='') USING SAMPLE 50000 (reservoir, 42))
);
""")

pool_cnt = con.execute("SELECT count(*) FROM target_pool").fetchone()[0]
print(f"Sample 10k S1, Target pool: {pool_cnt:,} records, setup took {time.time()-t0:.2f}s", flush=True)

# Run 8-pass blocking
t_block = time.time()
con.execute("""
CREATE TABLE raw_pairs AS
SELECT sub.entity_id as s1_id, sub.target_id, count(*) as pass_count
FROM (
    SELECT s.entity_id, t.entity_id as target_id FROM s1_sample s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2 WHERE length(s.p1)>=3 AND length(s.p2)>=3
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_sample s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.postal=t.postal WHERE length(s.p1)>=3 AND s.postal!=''
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_sample s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.door=t.door WHERE length(s.p1)>=3 AND s.door!=''
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_sample s JOIN target_pool t ON s.country=t.country AND s.cmp5=t.cmp5 WHERE length(s.cmp5)>=5
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_sample s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 WHERE length(s.p1)>=5
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_sample s JOIN target_pool t ON s.country=t.country AND s.postal=t.postal AND s.door=t.door WHERE s.postal!='' AND length(s.door)>=2
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_sample s JOIN target_pool t ON s.country=t.country AND s.door=t.door AND s.a1=t.a1 WHERE length(s.door)>=2 AND length(s.a1)>=4
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_sample s JOIN target_pool t ON s.country=t.country AND s.door=t.door AND s.a2=t.a2 WHERE length(s.door)>=2 AND length(s.a2)>=4
) sub
GROUP BY sub.entity_id, sub.target_id;
""")
pair_cnt = con.execute("SELECT count(*) FROM raw_pairs").fetchone()[0]
print(f"Generated {pair_cnt:,} distinct candidate pairs in {time.time()-t_block:.2f}s!", flush=True)
