import numpy as np
from rapidfuzz import fuzz, distance
from typing import List, Dict, Any, Tuple
import re

RE_DIGITS = re.compile(r'\b\d+\b')

FEATURE_NAMES = [
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_exact_match",
    "name_core_match",
    "name_len_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_num_jaccard",
    "addr_postal_match",
    "addr_door_match",
    "name_x_addr",
    "composite_score"
]


def extract_digit_set(text: str) -> set:
    if not text:
        return set()
    return set(RE_DIGITS.findall(text))


def compute_pair_features(
    s1_name: str,
    s1_core: str,
    s1_addr: str,
    s1_postal: str,
    s1_door: str,
    t_name: str,
    t_core: str,
    t_addr: str,
    t_postal: str,
    t_door: str
) -> List[float]:
    """Compute rich similarity feature vector between an S1 entity and a target candidate."""
    s1_n = s1_name or ""
    t_n = t_name or ""
    s1_c = s1_core or ""
    t_c = t_core or ""
    s1_a = s1_addr or ""
    t_a = t_addr or ""
    
    # 1. Name Features
    jw = distance.JaroWinkler.similarity(s1_n, t_n)
    tsort = fuzz.token_sort_ratio(s1_n, t_n) / 100.0
    tset = fuzz.token_set_ratio(s1_n, t_n) / 100.0
    pratio = fuzz.partial_ratio(s1_n, t_n) / 100.0
    exact_match = 1.0 if s1_n and s1_n == t_n else 0.0
    core_match = 1.0 if s1_c and s1_c == t_c else 0.0
    len_ratio = min(len(s1_n), len(t_n)) / max(len(s1_n), len(t_n), 1)
    
    # 2. Address Features
    addr_tsort = fuzz.token_sort_ratio(s1_a, t_a) / 100.0
    addr_tset = fuzz.token_set_ratio(s1_a, t_a) / 100.0
    
    # Numbers Jaccard
    s1_nums = extract_digit_set(s1_a)
    t_nums = extract_digit_set(t_a)
    if s1_nums and t_nums:
        num_jaccard = len(s1_nums & t_nums) / len(s1_nums | t_nums)
    elif not s1_nums and not t_nums:
        num_jaccard = 0.5
    else:
        num_jaccard = 0.0
        
    postal_match = 1.0 if s1_postal and t_postal and s1_postal == t_postal else 0.0
    door_match = 1.0 if s1_door and t_door and s1_door == t_door else 0.0
    
    # 3. Cross / Combined Features
    name_x_addr = tsort * addr_tsort
    composite = 0.65 * tsort + 0.35 * addr_tsort
    
    return [
        jw,
        tsort,
        tset,
        pratio,
        exact_match,
        core_match,
        len_ratio,
        addr_tsort,
        addr_tset,
        num_jaccard,
        postal_match,
        door_match,
        name_x_addr,
        composite
    ]


def batch_compute_features(
    s1_names: List[str],
    s1_cores: List[str],
    s1_addrs: List[str],
    s1_postals: List[str],
    s1_doors: List[str],
    t_names: List[str],
    t_cores: List[str],
    t_addrs: List[str],
    t_postals: List[str],
    t_doors: List[str]
) -> np.ndarray:
    """Compute feature matrix for a batch of candidate pairs."""
    n = len(s1_names)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(n):
        X[i] = compute_pair_features(
            s1_names[i], s1_cores[i], s1_addrs[i], s1_postals[i], s1_doors[i],
            t_names[i], t_cores[i], t_addrs[i], t_postals[i], t_doors[i]
        )
    return X
