import os, time, duckdb

t0 = time.time()
db_path = "scratch/colab_pipeline.duckdb"
con = duckdb.connect(db_path)
con.execute("SET memory_limit='6GB'; SET threads=8; SET preserve_insertion_order=false;")

s2_path = "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv"
s3_path = "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv"

print("Building target candidate pool (all 519k true targets + 300k random distractors)...", flush=True)
# Target pool consists of true targets for our 150k S1 sample + 300k random distractors from S2 and S3
con.execute(f"""
CREATE TABLE target_pool AS
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
FROM (
    SELECT * FROM read_csv_auto('{s2_path}', delim='\t', header=true, quote='')
    WHERE entity_id IN (SELECT target_id FROM true_target_ids)
    UNION ALL
    SELECT * FROM read_csv_auto('{s3_path}', delim='\t', header=true, quote='')
    WHERE entity_id IN (SELECT target_id FROM true_target_ids)
    UNION ALL
    (SELECT * FROM read_csv_auto('{s2_path}', delim='\t', header=true, quote='') USING SAMPLE 150000 (reservoir, 42))
    UNION ALL
    (SELECT * FROM read_csv_auto('{s3_path}', delim='\t', header=true, quote='') USING SAMPLE 150000 (reservoir, 42))
);
""")

pool_count = con.execute("SELECT count(*) FROM target_pool").fetchone()[0]
print(f"Target pool built: {pool_count:,} records in {time.time()-t0:.2f}s", flush=True)

# Test blocking for India partition
print("Testing 8-pass blocking on India partition (59k S1 x target pool)...", flush=True)
t_b = time.time()
con.execute("""
CREATE TABLE cands_india AS
SELECT s.entity_id as s1_id, t.entity_id as target_id,
       s.split, s.country,
       s.raw_name as s1_raw_name, s.norm_name as s1_norm_name, s.norm_addr as s1_norm_addr, s.postal as s1_postal, s.door as s1_door,
       t.raw_name as t_raw_name, t.norm_name as t_norm_name, t.norm_addr as t_norm_addr, t.postal as t_postal, t.door as t_door,
       count(*) as pass_count
FROM (
    SELECT s.entity_id, t.entity_id as target_id, 1 as pass_num
    FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2
    WHERE s.country='India' AND length(s.p1)>=3 AND length(s.p2)>=3
    UNION ALL
    SELECT s.entity_id, t.entity_id, 2
    FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.postal=t.postal
    WHERE s.country='India' AND length(s.p1)>=3 AND s.postal!=''
    UNION ALL
    SELECT s.entity_id, t.entity_id, 3
    FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.door=t.door
    WHERE s.country='India' AND length(s.p1)>=3 AND s.door!=''
    UNION ALL
    SELECT s.entity_id, t.entity_id, 4
    FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.cmp5=t.cmp5
    WHERE s.country='India' AND length(s.cmp5)>=5
    UNION ALL
    SELECT s.entity_id, t.entity_id, 5
    FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1
    WHERE s.country='India' AND length(s.p1)>=5
    UNION ALL
    SELECT s.entity_id, t.entity_id, 6
    FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.postal=t.postal AND s.door=t.door
    WHERE s.country='India' AND s.postal!='' AND length(s.door)>=2
    UNION ALL
    SELECT s.entity_id, t.entity_id, 7
    FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.door=t.door AND s.a1=t.a1
    WHERE s.country='India' AND length(s.door)>=2 AND length(s.a1)>=4
    UNION ALL
    SELECT s.entity_id, t.entity_id, 8
    FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.door=t.door AND s.a2=t.a2
    WHERE s.country='India' AND length(s.door)>=2 AND length(s.a2)>=4
) sub
JOIN s1_split s ON sub.entity_id = s.entity_id
JOIN target_pool t ON sub.target_id = t.entity_id
GROUP BY s.entity_id, t.entity_id, s.split, s.country,
         s.raw_name, s.norm_name, s.norm_addr, s.postal, s.door,
         t.raw_name, t.norm_name, t.norm_addr, t.postal, t.door;
""")
in_cands = con.execute("SELECT count(*) FROM cands_india").fetchone()[0]
print(f"India candidate pairs generated: {in_cands:,} in {time.time()-t_b:.2f}s", flush=True)
