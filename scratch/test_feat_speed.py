import time
import numpy as np
from rapidfuzz import fuzz, distance

def extract_features_single(s1_raw, s1_norm, s1_addr, s1_postal, s1_door,
                            t_raw, t_norm, t_addr, t_postal, t_door,
                            s1_country, t_country, rank, pass_count):
    # Name features
    exact_raw = 1.0 if s1_raw and s1_raw == t_raw else 0.0
    exact_norm = 1.0 if s1_norm and s1_norm == t_norm else 0.0
    name_sort = fuzz.token_sort_ratio(s1_norm, t_norm) / 100.0
    name_set = fuzz.token_set_ratio(s1_norm, t_norm) / 100.0
    jw = distance.JaroWinkler.similarity(s1_norm, t_norm)
    lev = distance.Levenshtein.normalized_similarity(s1_norm, t_norm)
    name_char_sim = fuzz.ratio(s1_norm, t_norm) / 100.0
    name_substr = 1.0 if (s1_norm and t_norm and (s1_norm in t_norm or t_norm in s1_norm)) else 0.0
    
    # Address features
    addr_exact = 1.0 if s1_addr and s1_addr == t_addr else 0.0
    addr_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
    addr_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
    
    s1_a_tok = set(s1_addr.split()) if s1_addr else set()
    t_a_tok = set(t_addr.split()) if t_addr else set()
    addr_jaccard = len(s1_a_tok & t_a_tok) / len(s1_a_tok | t_a_tok) if (s1_a_tok or t_a_tok) else 0.0
    addr_char_sim = fuzz.ratio(s1_addr, t_addr) / 100.0
    
    # Digit overlap
    s1_nums = {w for w in s1_a_tok if w.isdigit()}
    t_nums = {w for w in t_a_tok if w.isdigit()}
    if s1_nums and t_nums:
        num_overlap = len(s1_nums & t_nums) / len(s1_nums | t_nums)
    elif not s1_nums and not t_nums:
        num_overlap = 0.5
    else:
        num_overlap = 0.0
        
    postal_match = 1.0 if s1_postal and t_postal and s1_postal == t_postal else 0.0
    
    # Other features
    country_agree = 1.0 if s1_country == t_country else 0.0
    s1_name_miss = 1.0 if not s1_norm else 0.0
    t_name_miss = 1.0 if not t_norm else 0.0
    s1_addr_miss = 1.0 if not s1_addr else 0.0
    t_addr_miss = 1.0 if not t_addr else 0.0
    rank_feat = 1.0 / rank if rank > 0 else 0.0
    pass_cnt_feat = float(pass_count)
    name_x_addr = name_sort * addr_sort
    name_diff_addr = abs(name_sort - addr_sort)
    
    return [
        exact_raw, exact_norm, name_sort, name_set, jw, lev, name_char_sim, name_substr,
        addr_exact, addr_sort, addr_set, addr_jaccard, addr_char_sim, num_overlap, postal_match,
        country_agree, s1_name_miss, t_name_miss, s1_addr_miss, t_addr_miss,
        rank_feat, pass_cnt_feat, name_x_addr, name_diff_addr
    ]

# Benchmark 50,000 calls
N = 50000
t0 = time.time()
for _ in range(N):
    f = extract_features_single(
        "Apple Store Inc", "apple store inc", "1 infinite loop cupertino ca 95014", "95014", "1",
        "Apple Store", "apple store", "1 infinite loop cupertino 95014", "95014", "1",
        "US", "US", 1, 3
    )
el = time.time() - t0
print(f"Computed {N:,} features in {el:.2f}s ({N/el:,.0f} pairs/sec)")
