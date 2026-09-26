import duckdb
import os
import sys
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RESOURCE_DIR = os.path.join(ROOT_DIR, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR = os.path.join(RESOURCE_DIR, "dataset", "train")

gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv").replace("\\", "/")
s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv").replace("\\", "/")
s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("SET memory_limit = '4GB';")
con.execute("SET threads = 14;")

print("Testing Score-Sorted Candidate Budget Recall...", flush=True)

con.execute(f"""
CREATE TABLE eval_gt AS
SELECT source1_entity_id as s1_id, UNNEST(string_split(matched_entity_ids, ',')) as target_id
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) != ''
LIMIT 3000;
""")

s1_sample_ids = con.execute("SELECT DISTINCT s1_id FROM eval_gt LIMIT 1000;").df()['s1_id'].tolist()
total_gt_matches = con.execute(f"SELECT count(*) FROM eval_gt WHERE s1_id IN {tuple(s1_sample_ids)};").fetchone()[0]

con.execute(f"""
CREATE TABLE s1_sub AS SELECT * FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true) WHERE entity_id IN {tuple(s1_sample_ids)};
CREATE TABLE targets_sub AS
SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true) WHERE entity_id IN (SELECT target_id FROM eval_gt WHERE s1_id IN {tuple(s1_sample_ids)})
UNION ALL
SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true) WHERE entity_id IN (SELECT target_id FROM eval_gt WHERE s1_id IN {tuple(s1_sample_ids)})
UNION ALL
(SELECT * FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true) LIMIT 40000)
UNION ALL
(SELECT * FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true) LIMIT 40000);
""")

con.execute("""
CREATE MACRO norm_text(str) AS (lower(regexp_replace(regexp_replace(coalesce(str, ''), '[^a-zA-Z0-9 ]', ' ', 'g'), ' +', ' ', 'g')));
CREATE MACRO strip_domain(str) AS (regexp_replace(norm_text(str), '\\b(com|org|net|in|co|gov|edu|fr|io)\\b', '', 'g'));
CREATE MACRO clean_legal(str) AS (trim(regexp_replace(strip_domain(str), '\\b(private limited|pvt ltd|pvt limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|company|limited|llc|inc|corp|ltd|llp|sarl|sas|sa|sci|eurl|the|ms|smt|shree|sri|dr|pvt)\\b', '', 'g')));
CREATE MACRO compact_name(str) AS (regexp_replace(clean_legal(str), ' ', '', 'g'));
CREATE MACRO get_token(str, idx) AS (string_split(trim(str), ' ')[idx]);

CREATE MACRO clean_addr(str) AS (trim(regexp_replace(norm_text(str), '\\b(road|street|drive|avenue|lane|near|opp|opposite|floor|building|shop|plot|nagar|colony|block|west|east|north|south)\\b', '', 'g')));

CREATE TABLE s1_p AS SELECT entity_id, country, business_name, business_address,
    norm_text(business_name) as n_name, clean_legal(business_name) as c_name,
    norm_text(business_address) as n_addr,
    get_token(clean_legal(business_name), 1) as p1,
    get_token(clean_legal(business_name), 1) || ' ' || coalesce(get_token(clean_legal(business_name), 2), '') as p2,
    substring(compact_name(business_name), 1, 6) as cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') as postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') as door,
    get_token(clean_addr(business_address), 1) as a1
FROM s1_sub;

CREATE TABLE tgt_p AS SELECT entity_id, country, business_name, business_address,
    norm_text(business_name) as n_name, clean_legal(business_name) as c_name,
    norm_text(business_address) as n_addr,
    get_token(clean_legal(business_name), 1) as p1,
    get_token(clean_legal(business_name), 1) || ' ' || coalesce(get_token(clean_legal(business_name), 2), '') as p2,
    substring(compact_name(business_name), 1, 6) as cmp6,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{5,6}\\b') as postal,
    regexp_extract(coalesce(business_address, ''), '\\b[0-9]{1,4}\\b') as door,
    get_token(clean_addr(business_address), 1) as a1
FROM targets_sub;
""")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.blocking import run_blocking
run_blocking(con, 's1_p', 'tgt_p', 'cands')

cand_df = con.execute("""
SELECT c.s1_id, c.target_id, s.c_name as s1_c, s.n_addr as s1_a, t.c_name as t_c, t.n_addr as t_a
FROM cands c
JOIN s1_p s ON c.s1_id = s.entity_id
JOIN tgt_p t ON c.target_id = t.entity_id;
""").df()

print(f"Generated candidate pairs: {len(cand_df):,}", flush=True)

# Compute similarity score
scores = []
for s1_c, s1_a, t_c, t_a in zip(cand_df['s1_c'], cand_df['s1_a'], cand_df['t_c'], cand_df['t_a']):
    sim = 0.60 * (fuzz.token_sort_ratio(s1_c, t_c) / 100.0) + 0.40 * (fuzz.token_sort_ratio(s1_a, t_a) / 100.0)
    scores.append(sim)
cand_df['sim'] = scores

gt_pairs = set(zip(con.execute("SELECT s1_id, target_id FROM eval_gt").df()['s1_id'], con.execute("SELECT s1_id, target_id FROM eval_gt").df()['target_id']))

# Total unbudgeted recall
all_retrieved = set(zip(cand_df['s1_id'], cand_df['target_id']))
total_unbudgeted_hits = len(all_retrieved & gt_pairs)
print(f"Total Unbudgeted Candidate Recall: {total_unbudgeted_hits / total_gt_matches * 100:.2f}% ({total_unbudgeted_hits}/{total_gt_matches})\n")

# Score-sorted recall at different budgets
cand_df = cand_df.sort_values(['s1_id', 'sim'], ascending=[True, False])

for K in [5, 10, 15, 20, 25, 30, 50, 100]:
    top_k = cand_df.groupby('s1_id').head(K)
    retrieved = set(zip(top_k['s1_id'], top_k['target_id']))
    hits = len(retrieved & gt_pairs)
    print(f"Similarity-Sorted Budget K = {K:3d} -> Candidate Recall: {hits / total_gt_matches * 100:.2f}% ({hits}/{total_gt_matches})")
