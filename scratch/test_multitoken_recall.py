import polars as pl
import os
import sys
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\code\business_entity_resolution")
from src.preprocess import clean_business_name, clean_address

base_dir = r"c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

print("Testing Multi-Token & Number Inverted Index on India sample...")
gt = pl.read_csv(f"{base_dir}/train_ground_truth.tsv", separator="\t")
sample_gt = gt.filter(pl.col("matched_entity_ids").is_not_null()).slice(0, 10000)

gt_map = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) for r in sample_gt.iter_rows(named=True)}
s1_ids = set(gt_map.keys())

s1_df = pl.read_csv(f"{base_dir}/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s1_ids)))
s1_india = s1_df.filter(pl.col("country") == "India")

s2_df = pl.read_csv(f"{base_dir}/train_source2.tsv", separator="\t")
s3_df = pl.read_csv(f"{base_dir}/train_source3.tsv", separator="\t")
target_df = pl.concat([s2_df, s3_df])
target_india = target_df.filter(pl.col("country") == "India")

print(f"S1 count: {len(s1_india):,}, Target count: {len(target_india):,}")

# Preprocess target names and addresses
target_names = [clean_business_name(x)[1] for x in target_india["business_name"]]
target_addr_nums = [clean_address(x)[1] for x in target_india["business_address"]]

# Frequency count of tokens to identify stop/common tokens
token_counter = Counter()
for name in target_names:
    for t in set(name.split()):
        if len(t) > 2:
            token_counter[t] += 1

print(f"Top 10 most common tokens: {token_counter.most_common(10)}")

# Inverted index: ignore tokens that appear in more than 2% of targets (e.g. >80,000 times)
STOP_THRESHOLD = 50000
inverted_index = defaultdict(list)
for idx, (name, nums) in enumerate(zip(target_names, target_addr_nums)):
    tokens = set(name.split())
    for t in tokens:
        if len(t) > 2 and token_counter[t] < STOP_THRESHOLD:
            # store index
            if len(inverted_index[t]) < 1000:  # cap postings per token to prevent explosive lists
                inverted_index[t].append(idx)

# Also index on 6-digit PIN codes (Indian postal codes are 6 digits)
pin_index = defaultdict(list)
for idx, nums in enumerate(target_addr_nums):
    for num in nums:
        if len(num) == 6: # PIN code
            if len(pin_index[num]) < 500:
                pin_index[num].append(idx)

print("Inverted index built. Evaluating recall...")
s1_names = [clean_business_name(x)[1] for x in s1_india["business_name"]]
s1_addr_nums = [clean_address(x)[1] for x in s1_india["business_address"]]

hits = 0
total_gt_matches = 0
cand_sizes = []

for s1_row, s1_name, s1_nums in zip(s1_india.iter_rows(named=True), s1_names, s1_addr_nums):
    s1_id = s1_row["entity_id"]
    true_targets = gt_map[s1_id]
    total_gt_matches += len(true_targets)
    
    # Candidate scoring: count token overlaps
    candidate_scores = Counter()
    tokens = [t for t in set(s1_name.split()) if len(t) > 2 and token_counter.get(t, 0) < STOP_THRESHOLD]
    for t in tokens:
        postings = inverted_index.get(t, [])
        # weight inversely by frequency
        weight = 1.0 / (np.log1p(token_counter.get(t, 1)))
        for target_idx in postings:
            candidate_scores[target_idx] += weight
            
    # Also boost if PIN code matches
    for num in s1_nums:
        if len(num) == 6:
            for target_idx in pin_index.get(num, []):
                candidate_scores[target_idx] += 0.5
                
    # Pick top 25 candidates
    top_candidates = [idx for idx, _ in candidate_scores.most_common(25)]
    candidate_target_ids = {target_india["entity_id"][c_idx] for c_idx in top_candidates}
    
    hits += len(true_targets.intersection(candidate_target_ids))
    cand_sizes.append(len(top_candidates))

recall = hits / total_gt_matches if total_gt_matches > 0 else 0
print(f"Top-25 Multi-Token Inverted Index Recall: {recall*100:.2f}% ({hits}/{total_gt_matches})")
print(f"Avg candidates per entity: {np.mean(cand_sizes):.1f}")
