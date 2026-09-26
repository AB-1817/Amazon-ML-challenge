import os
import sys
import json
import duckdb

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RESOURCE_DIR = os.path.join(ROOT_DIR, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR = os.path.join(RESOURCE_DIR, "dataset", "train")

gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv").replace("\\", "/")
s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv").replace("\\", "/")
s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("SET memory_limit = '6GB';")
con.execute("SET threads = 14;")

print("=== 6. HARD NEGATIVE ANALYSIS ===", flush=True)

# 1. Same exact business name, but DIFFERENT businesses (not in ground truth)
print("Searching for hard negatives: Same Name, Different Entities...", flush=True)
con.execute(f"""
CREATE TABLE s1_sub AS 
SELECT entity_id, country, business_name, business_address
FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
LIMIT 50000;

CREATE TABLE s2_sub AS 
SELECT entity_id, country, business_name, business_address
FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true)
LIMIT 200000;

CREATE TABLE gt_links AS
SELECT source1_entity_id as s1_id, UNNEST(string_split(matched_entity_ids, ',')) as target_id
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
LIMIT 200000;
""")

# Same exact name, same country, but NOT a true match
same_name_diff_entity = con.execute("""
SELECT 
    s.entity_id as s1_id,
    s.business_name as s1_name,
    s.business_address as s1_addr,
    t.entity_id as t_id,
    t.business_name as t_name,
    t.business_address as t_addr,
    s.country
FROM s1_sub s
JOIN s2_sub t ON s.country = t.country AND lower(trim(s.business_name)) = lower(trim(t.business_name))
LEFT JOIN gt_links g ON s.entity_id = g.s1_id AND t.entity_id = g.target_id
WHERE g.target_id IS NULL AND s.entity_id != t.entity_id
LIMIT 5;
""").df().to_dict(orient='records')

print(f"Found same-name false matches (hard negatives): {len(same_name_diff_entity)}")
for idx, r in enumerate(same_name_diff_entity):
    print(f"\n--- Hard Negative (Same Name, Diff Entity) #{idx+1} [{r['country']}] ---")
    print(f"  S1: [{r['s1_id']}] '{r['s1_name']}' | '{r['s1_addr']}'")
    print(f"  Tg: [{r['t_id']}] '{r['t_name']}' | '{r['t_addr']}'")

# 2. Same address, but DIFFERENT businesses (multi-tenant building / office park)
print("\nSearching for hard negatives: Same Address, Different Entities...", flush=True)
same_addr_diff_entity = con.execute("""
SELECT 
    s.entity_id as s1_id,
    s.business_name as s1_name,
    s.business_address as s1_addr,
    t.entity_id as t_id,
    t.business_name as t_name,
    t.business_address as t_addr,
    s.country
FROM s1_sub s
JOIN s2_sub t ON s.country = t.country AND lower(trim(s.business_address)) = lower(trim(t.business_address))
LEFT JOIN gt_links g ON s.entity_id = g.s1_id AND t.entity_id = g.target_id
WHERE g.target_id IS NULL AND length(s.business_address) > 15
LIMIT 5;
""").df().to_dict(orient='records')

print(f"Found same-address false matches (hard negatives): {len(same_addr_diff_entity)}")
for idx, r in enumerate(same_addr_diff_entity):
    print(f"\n--- Hard Negative (Same Address, Diff Entity) #{idx+1} [{r['country']}] ---")
    print(f"  S1: [{r['s1_id']}] '{r['s1_name']}' | '{r['s1_addr']}'")
    print(f"  Tg: [{r['t_id']}] '{r['t_name']}' | '{r['t_addr']}'")

# 3. Generic common names in the dataset
generic_names = con.execute("""
SELECT business_name, count(*) as freq
FROM s1_sub
GROUP BY business_name
ORDER BY freq DESC
LIMIT 10;
""").df().to_dict(orient='records')
print(f"\nTop Generic Business Names in S1 sample:\n{generic_names}")

out_file = os.path.join(os.path.dirname(__file__), "hard_negatives_summary.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "same_name_diff_entity": same_name_diff_entity,
        "same_addr_diff_entity": same_addr_diff_entity,
        "generic_names": generic_names
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved hard negatives summary to: {out_file}")
