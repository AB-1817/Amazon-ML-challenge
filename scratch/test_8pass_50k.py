import duckdb, time

s1_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv'
s2_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv'
s3_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv'

con = duckdb.connect(':memory:')
con.execute("SET threads=8; SET memory_limit='6GB';")

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

print("Testing 8-pass blocking on India partition (smaller to check speed and pair count)...")
t0 = time.time()
con.execute(f"""
CREATE TABLE s1_in AS
SELECT 
    entity_id, country,
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
WHERE country = 'India'
LIMIT 50000;
""")
print(f"Loaded 50k India S1 in {time.time()-t0:.2f}s")

t0 = time.time()
con.execute(f"""
CREATE TABLE target_in AS
SELECT 
    entity_id, country,
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
FROM read_csv_auto('{s2_path}', delim='\t', header=true, quote='')
WHERE country = 'India'
UNION ALL
SELECT 
    entity_id, country,
    norm_name, core_name, norm_addr, p1, p2, p3, cmp4, cmp5, postal, door, a1, a2
FROM (
    SELECT 
        entity_id, country,
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
    FROM read_csv_auto('{s3_path}', delim='\t', header=true, quote='')
    WHERE country = 'India'
);
""")
tgt_count = con.execute("SELECT count(*) FROM target_in").fetchone()[0]
print(f"Loaded {tgt_count:,} India targets in {time.time()-t0:.2f}s")

t0 = time.time()
print("Executing 8-pass blocking...")
con.execute("""
CREATE TABLE raw_cands AS
SELECT s.entity_id as s1_id, t.entity_id as target_id
FROM s1_in s JOIN target_in t ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2
WHERE length(s.p1)>=3 AND length(s.p2)>=3
UNION
SELECT s.entity_id, t.entity_id FROM s1_in s JOIN target_in t ON s.country=t.country AND s.p1=t.p1 AND s.postal=t.postal
WHERE length(s.p1)>=3 AND s.postal!=''
UNION
SELECT s.entity_id, t.entity_id FROM s1_in s JOIN target_in t ON s.country=t.country AND s.p1=t.p1 AND s.door=t.door
WHERE length(s.p1)>=3 AND s.door!=''
UNION
SELECT s.entity_id, t.entity_id FROM s1_in s JOIN target_in t ON s.country=t.country AND s.cmp5=t.cmp5
WHERE length(s.cmp5)>=5
UNION
SELECT s.entity_id, t.entity_id FROM s1_in s JOIN target_in t ON s.country=t.country AND s.p1=t.p1
WHERE length(s.p1)>=5
UNION
SELECT s.entity_id, t.entity_id FROM s1_in s JOIN target_in t ON s.country=t.country AND s.postal=t.postal AND s.door=t.door
WHERE s.postal!='' AND length(s.door)>=2
UNION
SELECT s.entity_id, t.entity_id FROM s1_in s JOIN target_in t ON s.country=t.country AND s.door=t.door AND s.a1=t.a1
WHERE length(s.door)>=2 AND length(s.a1)>=4
UNION
SELECT s.entity_id, t.entity_id FROM s1_in s JOIN target_in t ON s.country=t.country AND s.door=t.door AND s.a2=t.a2
WHERE length(s.door)>=2 AND length(s.a2)>=4;
""")
pair_count = con.execute("SELECT count(*) FROM raw_cands").fetchone()[0]
print(f"8-pass blocking generated {pair_count:,} candidate pairs in {time.time()-t0:.2f}s!")
