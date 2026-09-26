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
s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv").replace("\\", "/")

con = duckdb.connect()
con.execute("SET memory_limit = '6GB';")
con.execute("SET threads = 14;")

print("=== 3. GROUND TRUTH ANALYSIS ===", flush=True)

# 1. Load GT and count matches per S1 entity
con.execute(f"""
CREATE TABLE gt_raw AS
SELECT 
    source1_entity_id,
    matched_entity_ids,
    CASE 
        WHEN matched_entity_ids IS NULL OR trim(matched_entity_ids) = '' THEN 0
        ELSE length(matched_entity_ids) - length(replace(matched_entity_ids, ',', '')) + 1
    END as num_matches
FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true);
""")

total_s1 = con.execute("SELECT count(*) FROM gt_raw;").fetchone()[0]
print(f"Total Source 1 entities in ground truth: {total_s1:,}")

# Match distribution
m_dist = con.execute("""
SELECT 
    num_matches,
    count(*) as count,
    round(count(*)*100.0 / sum(count(*)) over (), 2) as pct
FROM gt_raw
GROUP BY num_matches
ORDER BY num_matches;
""").df()
print("\nMatch Count Distribution:")
print(m_dist.to_string(index=False))

# Bucketed stats
singletons = con.execute("SELECT count(*) FROM gt_raw WHERE num_matches = 0;").fetchone()[0]
one_match = con.execute("SELECT count(*) FROM gt_raw WHERE num_matches = 1;").fetchone()[0]
two_matches = con.execute("SELECT count(*) FROM gt_raw WHERE num_matches = 2;").fetchone()[0]
three_plus = con.execute("SELECT count(*) FROM gt_raw WHERE num_matches >= 3;").fetchone()[0]

print(f"\nSummary Categories:")
print(f"  0 matches (Singletons): {singletons:,} ({singletons/total_s1*100:.2f}%)")
print(f"  1 match:                {one_match:,} ({one_match/total_s1*100:.2f}%)")
print(f"  2 matches:              {two_matches:,} ({two_matches/total_s1*100:.2f}%)")
print(f"  3+ matches:             {three_plus:,} ({three_plus/total_s1*100:.2f}%)")

# Detailed stats on match count
stats = con.execute("""
SELECT 
    min(num_matches) as min_m,
    max(num_matches) as max_m,
    round(avg(num_matches), 3) as avg_m,
    median(num_matches) as med_m,
    quantile_cont(num_matches, 0.75) as p75_m,
    quantile_cont(num_matches, 0.90) as p90_m,
    quantile_cont(num_matches, 0.95) as p95_m,
    quantile_cont(num_matches, 0.99) as p99_m,
    sum(num_matches) as total_links
FROM gt_raw;
""").df().to_dict(orient='records')[0]

print(f"\nDistribution Statistics:")
print(f"  Total ground-truth links: {stats['total_links']:,}")
print(f"  Average matches per S1:   {stats['avg_m']}")
print(f"  Median matches per S1:    {stats['med_m']}")
print(f"  Min / Max matches:        {stats['min_m']} / {stats['max_m']}")
print(f"  75th percentile:          {stats['p75_m']}")
print(f"  90th percentile:          {stats['p90_m']}")
print(f"  95th percentile:          {stats['p95_m']}")
print(f"  99th percentile:          {stats['p99_m']}")

# Separate S2 and S3 match counts
print("\nUnnesting links to analyze targets...", flush=True)
con.execute("""
CREATE TABLE gt_links AS
SELECT 
    source1_entity_id,
    UNNEST(string_split(matched_entity_ids, ',')) as target_id
FROM gt_raw
WHERE num_matches > 0;
""")

total_unpacked_links = con.execute("SELECT count(*) FROM gt_links;").fetchone()[0]
print(f"Total unpacked links: {total_unpacked_links:,}")

# Prefix counts (S2 vs S3 vs unexpected)
prefix_counts = con.execute("""
SELECT 
    substr(target_id, 1, 3) as prefix,
    count(*) as count,
    round(count(*)*100.0 / sum(count(*)) over (), 2) as pct
FROM gt_links
GROUP BY prefix
ORDER BY count DESC;
""").df()
print("\nTarget ID Prefix Distribution:")
print(prefix_counts.to_string(index=False))

# Check for unexpected/invalid IDs:
# 1. Any S1 target IDs?
s1_targets = con.execute("SELECT count(*) FROM gt_links WHERE target_id LIKE 'S1-%';").fetchone()[0]
print(f"\nInvalid checks:")
print(f"  S1 self-matches in ground truth: {s1_targets}")

# 2. Check existence in train_source2 and train_source3
print("Verifying that all ground truth target IDs exist in train_source2 and train_source3...", flush=True)
con.execute(f"""
CREATE TABLE s2_ids AS SELECT entity_id FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true);
CREATE TABLE s3_ids AS SELECT entity_id FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true);
""")

missing_s2 = con.execute("""
SELECT count(*) 
FROM gt_links 
WHERE target_id LIKE 'S2-%' AND target_id NOT IN (SELECT entity_id FROM s2_ids);
""").fetchone()[0]

missing_s3 = con.execute("""
SELECT count(*) 
FROM gt_links 
WHERE target_id LIKE 'S3-%' AND target_id NOT IN (SELECT entity_id FROM s3_ids);
""").fetchone()[0]

print(f"  S2 IDs in ground truth missing from train_source2: {missing_s2}")
print(f"  S3 IDs in ground truth missing from train_source3: {missing_s3}")

# Check for duplicate targets within a single S1 row
dup_targets = con.execute("""
SELECT count(*) FROM (
    SELECT source1_entity_id, target_id, count(*) as c
    FROM gt_links
    GROUP BY source1_entity_id, target_id
    HAVING c > 1
);
""").fetchone()[0]
print(f"  Duplicate targets within same S1 entity: {dup_targets}")

# Check Country consistency between S1 and GT matches
print("\nChecking country consistency of all ground truth matches...", flush=True)
con.execute(f"""
CREATE TABLE s1_meta AS SELECT entity_id, country FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true);
CREATE TABLE s2_meta AS SELECT entity_id, country FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true);
CREATE TABLE s3_meta AS SELECT entity_id, country FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true);
""")

cross_country = con.execute("""
SELECT count(*) FROM (
    SELECT l.source1_entity_id, l.target_id, s1.country as c1, coalesce(s2.country, s3.country) as c2
    FROM gt_links l
    JOIN s1_meta s1 ON l.source1_entity_id = s1.entity_id
    LEFT JOIN s2_meta s2 ON l.target_id = s2.entity_id
    LEFT JOIN s3_meta s3 ON l.target_id = s3.entity_id
    WHERE s1.country != coalesce(s2.country, s3.country)
);
""").fetchone()[0]
print(f"  Cross-country matches across ALL {total_unpacked_links:,} links: {cross_country}")

out_path = os.path.join(os.path.dirname(__file__), "ground_truth_summary.json")
with open(out_path, "w") as f:
    json.dump({
        "total_s1": total_s1,
        "singletons": singletons,
        "one_match": one_match,
        "two_matches": two_matches,
        "three_plus": three_plus,
        "total_links": stats['total_links'],
        "avg_matches": stats['avg_m'],
        "cross_country": cross_country
    }, f, indent=2)
print(f"\nGround truth audit saved to: {out_path}")
