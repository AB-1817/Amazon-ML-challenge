import polars as pl
import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

sys.path.insert(0, r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\code\business_entity_resolution")
from src.preprocess import clean_business_name, clean_address

base_dir = r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

print("Loading 20,000 S1 entities and ground truth...")
gt = pl.read_csv(f"{base_dir}/train_ground_truth.tsv", separator="\t")
sample_gt = gt.filter(pl.col("matched_entity_ids").is_not_null()).slice(0, 20000)

gt_map = {}
for r in sample_gt.iter_rows(named=True):
    gt_map[r["source1_entity_id"]] = set(r["matched_entity_ids"].split(","))

s1_ids = set(gt_map.keys())
s1_df = pl.read_csv(f"{base_dir}/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s1_ids)))

# Also load target IDs that are actually in ground truth plus some background targets
all_target_ids = set()
for t_ids in gt_map.values():
    all_target_ids.update(t_ids)

print(f"Total true targets to find: {len(all_target_ids):,}")

# Read target files (filter for sample + random slice to test scaling)
s2_df = pl.read_csv(f"{base_dir}/train_source2.tsv", separator="\t")
s3_df = pl.read_csv(f"{base_dir}/train_source3.tsv", separator="\t")

target_df = pl.concat([s2_df, s3_df])
print(f"Total target pool size: {len(target_df):,}")

# Test country partition: let's test on India first
print("\n--- Testing on India subset ---")
s1_india = s1_df.filter(pl.col("country") == "India")
target_india = target_df.filter(pl.col("country") == "India")
print(f"S1 India count: {len(s1_india):,}, Target India pool: {len(target_india):,}")

# Preprocess names
print("Preprocessing names...")
s1_names = [clean_business_name(x)[1] for x in s1_india["business_name"]]
target_names = [clean_business_name(x)[1] for x in target_india["business_name"]]

# Build Prefix Inverted Index on first 1 and first 2 tokens
print("Building Prefix Inverted Index...")
prefix_index = defaultdict(list)
for idx, name in enumerate(target_names):
    tokens = name.split()
    if tokens:
        prefix_index[tokens[0]].append(idx)
        if len(tokens) > 1:
            prefix_index[tokens[0] + " " + tokens[1]].append(idx)

# Evaluate recall using Prefix Index
hits = 0
total_gt_matches = 0
candidate_counts = []

for s1_row, s1_name in zip(s1_india.iter_rows(named=True), s1_names):
    s1_id = s1_row["entity_id"]
    true_targets = gt_map[s1_id]
    total_gt_matches += len(true_targets)
    
    candidates = set()
    tokens = s1_name.split()
    if tokens:
        # Check 2-token prefix first
        if len(tokens) > 1:
            candidates.update(prefix_index.get(tokens[0] + " " + tokens[1], []))
        # If too few candidates, also add single token prefix
        if len(candidates) < 30:
            candidates.update(prefix_index.get(tokens[0], [])[:50])
            
    candidate_target_ids = {target_india["entity_id"][c_idx] for c_idx in candidates}
    hits += len(true_targets.intersection(candidate_target_ids))
    candidate_counts.append(len(candidates))

recall = hits / total_gt_matches if total_gt_matches > 0 else 0
print(f"Prefix Index Recall: {recall*100:.2f}% ({hits}/{total_gt_matches})")
print(f"Avg candidates per entity: {np.mean(candidate_counts):.1f}")
