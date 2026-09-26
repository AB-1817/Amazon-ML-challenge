import os
import sys
import json
import duckdb
import polars as pl
import numpy as np

# Reconfigure stdout for UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Set directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RESOURCE_DIR = os.path.join(ROOT_DIR, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR = os.path.join(RESOURCE_DIR, "dataset", "train")
TEST_DIR = os.path.join(RESOURCE_DIR, "dataset", "test")

print("=== 1. DATASET STRUCTURE & TSV INTEGRITY ===", flush=True)

files = [
    ("train_source1.tsv", os.path.join(TRAIN_DIR, "train_source1.tsv")),
    ("train_source2.tsv", os.path.join(TRAIN_DIR, "train_source2.tsv")),
    ("train_source3.tsv", os.path.join(TRAIN_DIR, "train_source3.tsv")),
    ("train_ground_truth.tsv", os.path.join(TRAIN_DIR, "train_ground_truth.tsv")),
    ("test_source1.tsv", os.path.join(TEST_DIR, "test_source1.tsv")),
    ("test_source2.tsv", os.path.join(TEST_DIR, "test_source2.tsv")),
    ("test_source3.tsv", os.path.join(TEST_DIR, "test_source3.tsv")),
]

con = duckdb.connect()
con.execute("SET memory_limit = '6GB';")
con.execute("SET threads = 14;")

results = {}

for name, path in files:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    # Check with duckdb
    p_sql = path.replace("\\", "/")
    df_desc = con.execute(f"DESCRIBE SELECT * FROM read_csv('{p_sql}', delim='\t', header=true, ignore_errors=true);").df()
    cols = df_desc["column_name"].tolist()
    dtypes = df_desc["column_type"].tolist()
    row_count = con.execute(f"SELECT count(*) FROM read_csv('{p_sql}', delim='\t', header=true, ignore_errors=true);").fetchone()[0]
    
    print(f"\nFile: {name}")
    print(f"  Path: {path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Rows: {row_count:,}")
    print(f"  Columns ({len(cols)}): {dict(zip(cols, dtypes))}")
    
    # Confirm TSV parsing by checking sample rows and ensuring no column concatenation
    sample = con.execute(f"SELECT * FROM read_csv('{p_sql}', delim='\t', header=true, ignore_errors=true) LIMIT 2;").df()
    print(f"  Sample row:\n{sample.to_dict(orient='records')}")
    
    results[name] = {
        "path": path,
        "size_mb": size_mb,
        "rows": row_count,
        "columns": dict(zip(cols, dtypes))
    }

print("\n=== 2. DETAILED SOURCE ANALYSIS (TRAIN & TEST) ===", flush=True)

for name, path in files:
    if "ground_truth" in name:
        continue
    p_sql = path.replace("\\", "/")
    print(f"\n--- Analyzing {name} ---", flush=True)
    
    # Missing values
    stats = con.execute(f"""
    SELECT 
        count(*) as total,
        count(entity_id) as non_null_id,
        count(business_name) as non_null_name,
        count(business_address) as non_null_addr,
        count(country) as non_null_country,
        count(distinct entity_id) as unique_ids,
        count(distinct business_name) as unique_names
    FROM read_csv('{p_sql}', delim='\t', header=true, ignore_errors=true);
    """).df().to_dict(orient='records')[0]
    
    total = stats['total']
    null_name = total - stats['non_null_name']
    null_addr = total - stats['non_null_addr']
    null_country = total - stats['non_null_country']
    
    print(f"  Total records: {total:,}")
    print(f"  Unique entity_ids: {stats['unique_ids']:,}")
    print(f"  Unique business names: {stats['unique_names']:,} ({stats['unique_names']/total*100:.2f}%)")
    print(f"  Duplicate business names: {total - stats['unique_names']:,} ({(total - stats['unique_names'])/total*100:.2f}%)")
    print(f"  Missing business_name: {null_name:,} ({null_name/total*100:.2f}%)")
    print(f"  Missing business_address: {null_addr:,} ({null_addr/total*100:.2f}%)")
    print(f"  Missing country: {null_country:,} ({null_country/total*100:.2f}%)")
    
    # Country distribution
    cdist = con.execute(f"""
    SELECT country, count(*) as count, round(count(*)*100.0/{total}, 2) as pct
    FROM read_csv('{p_sql}', delim='\t', header=true, ignore_errors=true)
    GROUP BY country
    ORDER BY count DESC;
    """).df().to_dict(orient='records')
    print(f"  Country Distribution: {cdist}")
    
    # Length statistics
    len_stats = con.execute(f"""
    SELECT 
        min(length(business_name)) as min_name_len,
        max(length(business_name)) as max_name_len,
        round(avg(length(business_name)), 2) as avg_name_len,
        median(length(business_name)) as med_name_len,
        quantile_cont(length(business_name), 0.95) as p95_name_len,
        min(length(coalesce(business_address, ''))) as min_addr_len,
        max(length(coalesce(business_address, ''))) as max_addr_len,
        round(avg(length(coalesce(business_address, ''))), 2) as avg_addr_len,
        median(length(coalesce(business_address, ''))) as med_addr_len,
        quantile_cont(length(coalesce(business_address, '')), 0.95) as p95_addr_len
    FROM read_csv('{p_sql}', delim='\t', header=true, ignore_errors=true);
    """).df().to_dict(orient='records')[0]
    print(f"  Name Lengths: min={len_stats['min_name_len']}, max={len_stats['max_name_len']}, avg={len_stats['avg_name_len']}, med={len_stats['med_name_len']}, p95={len_stats['p95_name_len']}")
    print(f"  Address Lengths: min={len_stats['min_addr_len']}, max={len_stats['max_addr_len']}, avg={len_stats['avg_addr_len']}, med={len_stats['med_addr_len']}, p95={len_stats['p95_addr_len']}")

out_json = os.path.join(os.path.dirname(__file__), "structure_and_sources_summary.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSummary saved to: {out_json}")
