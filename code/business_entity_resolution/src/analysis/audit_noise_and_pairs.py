import os
import sys
import json
import duckdb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, distance
import re

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

print("=== 4 & 5. NOISE & POSITIVE-PAIR ANALYSIS ===", flush=True)

# Sample 50,000 ground truth links
con.execute(f"""
CREATE TABLE sample_links AS
SELECT 
    source1_entity_id as s1_id,
    target_id
FROM (
    SELECT 
        source1_entity_id,
        UNNEST(string_split(matched_entity_ids, ',')) as target_id
    FROM read_csv('{gt_path}', delim='\t', header=true, ignore_errors=true)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
) USING SAMPLE 50000;
""")

print("Loaded 50,000 sampled ground truth links. Joining metadata...", flush=True)

con.execute(f"""
CREATE TABLE s1_data AS
SELECT entity_id, country, business_name, business_address
FROM read_csv('{s1_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT s1_id FROM sample_links);

CREATE TABLE target_data AS
SELECT entity_id, country, business_name, business_address
FROM read_csv('{s2_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT target_id FROM sample_links)
UNION ALL
SELECT entity_id, country, business_name, business_address
FROM read_csv('{s3_path}', delim='\t', header=true, ignore_errors=true)
WHERE entity_id IN (SELECT target_id FROM sample_links);
""")

con.execute("""
CREATE TABLE pairs AS
SELECT 
    l.s1_id,
    l.target_id,
    s1.country as s1_country,
    t.country as t_country,
    s1.business_name as s1_name,
    t.business_name as t_name,
    coalesce(s1.business_address, '') as s1_addr,
    coalesce(t.business_address, '') as t_addr
FROM sample_links l
JOIN s1_data s1 ON l.s1_id = s1.entity_id
JOIN target_data t ON l.target_id = t.entity_id;
""")

df_pairs = con.execute("SELECT * FROM pairs").df()
total_pairs = len(df_pairs)
print(f"Retrieved {total_pairs:,} positive pairs for deep analysis.\n", flush=True)

# Helper for basic text normalization
RE_PUNCT = re.compile(r'[^\w\s]')
RE_SPACE = re.compile(r'\s+')
RE_DIGITS = re.compile(r'\b\d+\b')

def normalize_simple(s):
    if not s:
        return ""
    s = s.lower()
    s = RE_PUNCT.sub(' ', s)
    return RE_SPACE.sub(' ', s).strip()

def get_numbers(s):
    if not s:
        return set()
    return set(RE_DIGITS.findall(s))

exact_raw_name = 0
exact_norm_name = 0
exact_raw_addr = 0
exact_norm_addr = 0
country_agreed = 0

jw_scores = []
token_sort_name_scores = []
token_set_name_scores = []
lev_name_scores = []

token_sort_addr_scores = []
token_set_addr_scores = []
num_jaccard_scores = []

weak_name_strong_addr = 0  # Name sim < 0.50, Addr sim >= 0.80
strong_name_weak_addr = 0  # Name sim >= 0.80, Addr sim < 0.50
both_strong = 0            # Name sim >= 0.80, Addr sim >= 0.80
both_weak = 0              # Name sim < 0.50, Addr sim < 0.50

# Collect qualitative examples for Noise Analysis
noise_examples = {
    "name_abbreviations": [],
    "legal_suffix_variations": [],
    "typos_character_corruptions": [],
    "punctuation_differences": [],
    "word_order_changes": [],
    "transliterations_non_latin": [],
    "address_abbreviations": [],
    "missing_address_components": [],
    "number_formats": [],
    "landmark_addresses": [],
    "domain_as_name": []
}

for _, r in df_pairs.iterrows():
    s1_n = r["s1_name"]
    t_n = r["t_name"]
    s1_a = r["s1_addr"]
    t_a = r["t_addr"]
    
    if r["s1_country"] == r["t_country"]:
        country_agreed += 1
        
    if s1_n == t_n:
        exact_raw_name += 1
    norm_s1_n = normalize_simple(s1_n)
    norm_t_n = normalize_simple(t_n)
    if norm_s1_n == norm_t_n and norm_s1_n != "":
        exact_norm_name += 1
        
    if s1_a == t_a and s1_a != "":
        exact_raw_addr += 1
    norm_s1_a = normalize_simple(s1_a)
    norm_t_a = normalize_simple(t_a)
    if norm_s1_a == norm_t_a and norm_s1_a != "":
        exact_norm_addr += 1
        
    # Similarity metrics
    jw = distance.JaroWinkler.similarity(norm_s1_n, norm_t_n)
    lev_name = 1.0 - (distance.Levenshtein.distance(norm_s1_n, norm_t_n) / max(len(norm_s1_n), len(norm_t_n), 1))
    tsort_name = fuzz.token_sort_ratio(norm_s1_n, norm_t_n) / 100.0
    tset_name = fuzz.token_set_ratio(norm_s1_n, norm_t_n) / 100.0
    
    tsort_addr = fuzz.token_sort_ratio(norm_s1_a, norm_t_a) / 100.0 if (norm_s1_a and norm_t_a) else 0.0
    tset_addr = fuzz.token_set_ratio(norm_s1_a, norm_t_a) / 100.0 if (norm_s1_a and norm_t_a) else 0.0
    
    nums_s1 = get_numbers(s1_a)
    nums_t = get_numbers(t_a)
    if nums_s1 and nums_t:
        num_jacc = len(nums_s1 & nums_t) / len(nums_s1 | nums_t)
    elif not nums_s1 and not nums_t:
        num_jacc = 0.5
    else:
        num_jacc = 0.0
        
    jw_scores.append(jw)
    lev_name_scores.append(lev_name)
    token_sort_name_scores.append(tsort_name)
    token_set_name_scores.append(tset_name)
    token_sort_addr_scores.append(tsort_addr)
    token_set_addr_scores.append(tset_addr)
    num_jaccard_scores.append(num_jacc)
    
    # Joint strength analysis
    if tsort_name < 0.50 and tsort_addr >= 0.80:
        weak_name_strong_addr += 1
    elif tsort_name >= 0.80 and tsort_addr < 0.50:
        strong_name_weak_addr += 1
    elif tsort_name >= 0.80 and tsort_addr >= 0.80:
        both_strong += 1
    elif tsort_name < 0.50 and tsort_addr < 0.50:
        both_weak += 1
        
    # Classify noise patterns for qualitative audit (collect up to 5 each)
    # 1. Domain as name
    if ('.com' in t_n.lower() or '.in' in t_n.lower() or '.org' in t_n.lower()) and ('.com' not in s1_n.lower()) and len(noise_examples["domain_as_name"]) < 5:
        noise_examples["domain_as_name"].append({"s1_name": s1_n, "t_name": t_n, "s1_addr": s1_a, "t_addr": t_a, "country": r["s1_country"]})
        
    # 2. Non-latin script / transliterations
    has_non_ascii_s1 = any(ord(c) > 127 for c in s1_n)
    has_non_ascii_t = any(ord(c) > 127 for c in t_n)
    if (has_non_ascii_s1 or has_non_ascii_t) and len(noise_examples["transliterations_non_latin"]) < 5:
        noise_examples["transliterations_non_latin"].append({"s1_name": s1_n, "t_name": t_n, "s1_addr": s1_a, "t_addr": t_a, "country": r["s1_country"]})
        
    # 3. Word order changes
    if tsort_name > 0.85 and lev_name < 0.65 and len(noise_examples["word_order_changes"]) < 5:
        noise_examples["word_order_changes"].append({"s1_name": s1_n, "t_name": t_n, "country": r["s1_country"]})
        
    # 4. Legal suffix variations
    legal_words = {'pvt', 'private', 'ltd', 'limited', 'inc', 'corp', 'corporation', 'llc', 'llp', 'co', 'company'}
    s1_w = set(norm_s1_n.split())
    t_w = set(norm_t_n.split())
    if (s1_w - legal_words == t_w - legal_words) and (s1_w != t_w) and len(noise_examples["legal_suffix_variations"]) < 5:
        noise_examples["legal_suffix_variations"].append({"s1_name": s1_n, "t_name": t_n, "country": r["s1_country"]})
        
    # 5. Typos / slight character differences
    if 0.75 < lev_name < 0.95 and tsort_name > 0.80 and len(noise_examples["typos_character_corruptions"]) < 5:
        noise_examples["typos_character_corruptions"].append({"s1_name": s1_n, "t_name": t_n, "country": r["s1_country"]})
        
    # 6. Landmark addresses
    landmark_kws = {'near', 'opp', 'opposite', 'behind', 'beside', 'next to', 'above', 'floor'}
    if any(k in s1_a.lower() or k in t_a.lower() for k in landmark_kws) and len(noise_examples["landmark_addresses"]) < 5:
        noise_examples["landmark_addresses"].append({"s1_addr": s1_a, "t_addr": t_a, "country": r["s1_country"]})
        
    # 7. Number format variations
    if nums_s1 and nums_t and nums_s1 != nums_t and len(noise_examples["number_formats"]) < 5:
        noise_examples["number_formats"].append({"s1_addr": s1_a, "t_addr": t_a, "nums_s1": list(nums_s1), "nums_t": list(nums_t)})
        
    # 8. Address abbreviations
    addr_abbr_kws = ['st', 'street', 'rd', 'road', 'dr', 'drive', 'ave', 'avenue', 'blvd', 'boulevard']
    if any(k in s1_a.lower().split() for k in ['rd', 'st', 'dr', 'ave']) and any(k in t_a.lower().split() for k in ['road', 'street', 'drive', 'avenue']) and len(noise_examples["address_abbreviations"]) < 5:
        noise_examples["address_abbreviations"].append({"s1_addr": s1_a, "t_addr": t_a, "country": r["s1_country"]})
        
    # 9. Missing address components
    if (len(s1_a) > 50 and len(t_a) < 25) or (len(t_a) > 50 and len(s1_a) < 25) or (t_a == "") and len(noise_examples["missing_address_components"]) < 5:
        noise_examples["missing_address_components"].append({"s1_addr": s1_a, "t_addr": t_a, "country": r["s1_country"]})

print("=== QUANTITATIVE POSITIVE-PAIR METRICS ===")
print(f"Country Agreement: {country_agreed / total_pairs * 100:.2f}% ({country_agreed}/{total_pairs})")
print(f"Exact Raw Name Match: {exact_raw_name / total_pairs * 100:.2f}% ({exact_raw_name}/{total_pairs})")
print(f"Exact Normalized Name Match: {exact_norm_name / total_pairs * 100:.2f}% ({exact_norm_name}/{total_pairs})")
print(f"Exact Raw Address Match: {exact_raw_addr / total_pairs * 100:.2f}% ({exact_raw_addr}/{total_pairs})")
print(f"Exact Normalized Address Match: {exact_norm_addr / total_pairs * 100:.2f}% ({exact_norm_addr}/{total_pairs})")

def summarize_metric(name, arr):
    print(f"\n{name}:")
    print(f"  Mean:   {np.mean(arr):.4f}")
    print(f"  Median: {np.median(arr):.4f}")
    print(f"  Min:    {np.min(arr):.4f}")
    print(f"  Max:    {np.max(arr):.4f}")
    print(f"  p10:    {np.percentile(arr, 10):.4f}")
    print(f"  p25:    {np.percentile(arr, 25):.4f}")
    print(f"  p75:    {np.percentile(arr, 75):.4f}")
    print(f"  p90:    {np.percentile(arr, 90):.4f}")

summarize_metric("Name Jaro-Winkler Similarity", jw_scores)
summarize_metric("Name Token Sort Ratio", token_sort_name_scores)
summarize_metric("Name Token Set Ratio", token_set_name_scores)
summarize_metric("Name Levenshtein Ratio", lev_name_scores)

summarize_metric("Address Token Sort Ratio", token_sort_addr_scores)
summarize_metric("Address Token Set Ratio", token_set_addr_scores)
summarize_metric("Address Number Jaccard Overlap", num_jaccard_scores)

print("\n=== JOINT SIGNAL STRENGTH BREAKDOWN ===")
print(f"Both Strong (Name >= 0.80 AND Addr >= 0.80): {both_strong:,} ({both_strong/total_pairs*100:.2f}%)")
print(f"Weak Name but Strong Addr (Name < 0.50, Addr >= 0.80): {weak_name_strong_addr:,} ({weak_name_strong_addr/total_pairs*100:.2f}%)")
print(f"Strong Name but Weak Addr (Name >= 0.80, Addr < 0.50): {strong_name_weak_addr:,} ({strong_name_weak_addr/total_pairs*100:.2f}%)")
print(f"Both Weak (Name < 0.50 AND Addr < 0.50): {both_weak:,} ({both_weak/total_pairs*100:.2f}%)")
other = total_pairs - (both_strong + weak_name_strong_addr + strong_name_weak_addr + both_weak)
print(f"Intermediate / Mixed Signal: {other:,} ({other/total_pairs*100:.2f}%)")

out_metrics = {
    "total_pairs": total_pairs,
    "country_agreed_pct": country_agreed / total_pairs * 100,
    "exact_raw_name_pct": exact_raw_name / total_pairs * 100,
    "exact_norm_name_pct": exact_norm_name / total_pairs * 100,
    "exact_raw_addr_pct": exact_raw_addr / total_pairs * 100,
    "exact_norm_addr_pct": exact_norm_addr / total_pairs * 100,
    "both_strong_pct": both_strong / total_pairs * 100,
    "weak_name_strong_addr_pct": weak_name_strong_addr / total_pairs * 100,
    "strong_name_weak_addr_pct": strong_name_weak_addr / total_pairs * 100,
    "both_weak_pct": both_weak / total_pairs * 100,
    "noise_examples": noise_examples
}

out_file = os.path.join(os.path.dirname(__file__), "noise_and_pairs_summary.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(out_metrics, f, indent=2, ensure_ascii=False)
print(f"\nSaved metrics & noise examples to: {out_file}")
