import os
import sys
import time
import random
import duckdb
import pandas as pd
import numpy as np

t0 = time.time()
gt_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv'
s1_path = '6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv'

con = duckdb.connect(':memory:')
con.execute("SET threads=8;")
con.execute("SET memory_limit='6GB';")

print("Checking S1 count and sample...")
s1_count = con.execute(f"SELECT count(*) FROM read_csv_auto('{s1_path}', delim='\t', header=true, quote='')").fetchone()[0]
print(f"Total S1: {s1_count:,}")

# Sample 150,000 S1 entity IDs using reservoir/sample
print("Sampling 150k S1 entities...")
t_samp = time.time()
s1_sample_df = con.execute(f"""
SELECT entity_id, country, business_name, business_address
FROM read_csv_auto('{s1_path}', delim='\t', header=true, quote='')
USING SAMPLE 150000 (reservoir, 42);
""").fetchdf()
print(f"Sampled {len(s1_sample_df):,} in {time.time()-t_samp:.2f}s")
print(s1_sample_df['country'].value_counts())
