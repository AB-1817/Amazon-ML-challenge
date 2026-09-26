"""
create_colab_notebook.py
========================
Generates notebooks/train_matching_model_colab.ipynb with all 18 sections
for Google Colab execution.
"""
import json
import os

notebook_path = "notebooks/train_matching_model_colab.ipynb"
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# --- Notebook Header ---
add_md("""
# Amazon ML Challenge 2026: Business Entity Resolution
## Candidate-Pair Matching Model Training & Validation (Google Colab Ready)

This notebook implements the complete, reproducible training and validation pipeline for the **candidate-pair matching model** under the finalized **8-pass blocking architecture**.

### Pipeline Architecture:
1. **Google Drive Integration & Environment Setup**: Configurable paths with automatic file verification.
2. **S1 Entity Sampling & Train/Val Split**: Strict 80/20 S1-level split (zero candidate pair leakage).
3. **8-Pass Blocking**: Country-partitioned candidate generation with token-Jaccard pre-ranking ($K=25$).
4. **Hard-Negative Sampling**: Prioritizing hard negatives from candidate pool (name/address collisions).
5. **Feature Engineering**: 24 candidate-pair similarity features using RapidFuzz on top-$K$ pairs.
6. **Model Training & Comparison**: Logistic Regression vs. XGBoost under identical splits and features.
7. **Threshold Search & Macro $F_{0.5}$ Optimization**: Exact challenge evaluation including singletons.
8. **Error Analysis**: Systematic categorization of False Positives and False Negatives.
9. **Artifact & Results Export**: Self-contained models (`artifacts/`) and reports (`results/`).
""")

# --- Section 1: Environment Setup ---
add_md("### Section 1: Environment Setup & Reproducibility")
add_code("""
import os
import sys
import time
import json
import random
import platform
import numpy as np
import pandas as pd

# Set global random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("  SYSTEM & ENVIRONMENT INFORMATION")
print("=" * 60)
print(f"Python Version:   {platform.python_version()}")
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"CPU Count:        {os.cpu_count()}")
print(f"NumPy Version:    {np.__version__}")
print(f"Pandas Version:   {pd.__version__}")
print(f"Random Seed:      {SEED}")
print("=" * 60)
""")

# --- Section 2: Dataset Location / Configuration ---
add_md("""
### Section 2: Dataset Location / Configuration & Google Drive Support

Set `DATA_ROOT` to the directory containing the TSV files.
If running in Google Colab, you can mount Google Drive and point `DATA_ROOT` to your Drive folder.
""")
add_code("""
# Check if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    # Default Colab Drive path - change this if your files are in a different folder
    DATA_ROOT = "/content/drive/MyDrive/amazon_ml_challenge"
    if not os.path.exists(DATA_ROOT):
        # Fallback to local /content/ if copied directly
        DATA_ROOT = "/content/dataset/train"
else:
    # Local fallback path
    DATA_ROOT = os.path.abspath("../6ab10eb3b23ba_student_resource/student_resource/dataset/train")

print(f"DATA_ROOT configured as: {DATA_ROOT}")

# Expected training files
REQUIRED_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv"
]

# Verify required files exist and display sizes
all_found = True
print("\\nVerifying required dataset files:")
for fname in REQUIRED_FILES:
    fpath = os.path.join(DATA_ROOT, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  [OK] {fname:<25} ({size_mb:6.1f} MB)")
    else:
        print(f"  [MISSING] {fname:<25} at {fpath}")
        all_found = False

if not all_found:
    print("\\nWARNING: One or more files are missing. Please ensure datasets are uploaded to DATA_ROOT.")
else:
    print("\\nAll required training files successfully verified!")
""")

# --- Section 3: Install Dependencies ---
add_md("### Section 3: Install Dependencies")
add_code("""
# Install required libraries
!pip install -q duckdb>=1.1.0 rapidfuzz>=3.0.0 xgboost>=2.0.0 scikit-learn>=1.3.0 joblib>=1.3.0

import duckdb
import rapidfuzz
import xgboost as xgb
import sklearn

print("Successfully verified core package imports:")
print(f"  DuckDB:     {duckdb.__version__}")
print(f"  RapidFuzz:  {rapidfuzz.__version__}")
print(f"  XGBoost:    {xgb.__version__}")
print(f"  scikit-learn:{sklearn.__version__}")
""")

# --- Section 4: Data Loading & Validation ---
add_md("""
### Section 4: Load and Validate Data

Load the TSV files using `sep="\\t"` and inspect row counts, schema, and missing values.
""")
add_code("""
# Load head of each file with sep='\\t' to inspect schema
s1_path = os.path.join(DATA_ROOT, "train_source1.tsv").replace("\\\\", "/")
s2_path = os.path.join(DATA_ROOT, "train_source2.tsv").replace("\\\\", "/")
s3_path = os.path.join(DATA_ROOT, "train_source3.tsv").replace("\\\\", "/")
gt_path = os.path.join(DATA_ROOT, "train_ground_truth.tsv").replace("\\\\", "/")

con = duckdb.connect(":memory:")
con.execute("SET memory_limit='6GB';")
con.execute("SET threads=8;")

print("Data Validation & Schema Inspection:")
for name, p in [("Source 1", s1_path), ("Source 2", s2_path), ("Source 3", s3_path), ("Ground Truth", gt_path)]:
    sample_df = con.execute(f"SELECT * FROM read_csv_auto('{p}', delim='\\t', header=true, quote='') LIMIT 3").df()
    cnt = con.execute(f"SELECT count(*) FROM read_csv_auto('{p}', delim='\\t', header=true, quote='')").fetchone()[0]
    print(f"\\n--- {name} ({cnt:,} rows) ---")
    print(f"Columns: {list(sample_df.columns)}")
    display(sample_df)
""")

# --- Section 5: Preprocessing & SQL Normalization Macros ---
add_md("""
### Section 5: Preprocessing & SQL Normalization Macros

Register standardized, high-performance DuckDB SQL macros:
- `norm_text`: Unicode NFKD normalization, punctuation replacement, lowercasing.
- `clean_legal`: Stripping common corporate suffixes (LLC, Inc, Pvt Ltd, Sarl, etc.).
- `compact_name`: Removing all whitespace for substring & prefix indexing.
- `get_token`: Extracting $n$-th space-delimited token.
- `clean_addr`: Address abbreviation normalization and whitespace compaction.
""")
add_code("""
con.execute(\"\"\"
CREATE OR REPLACE MACRO norm_text(s) AS
    lower(trim(regexp_replace(regexp_replace(
        regexp_replace(coalesce(s,''), 'https?://(?:www\\\\.)?|www\\\\.', '', 'g'),
        '\\\\.(?:com|org|net|in|co|gov|edu|fr|io)\\\\b', ' ', 'g'),
        '[^\\\\w\\\\s]', ' ', 'g')));

CREATE OR REPLACE MACRO clean_legal(s) AS
    trim(regexp_replace(norm_text(s),
    '\\\\b(?:private\\\\s+limited|pvt\\\\s+ltd|pvt\\\\s+limited|incorporated|corporation|enterprises|enterprise|associates|industries|industry|holdings|holding|services|service|company|limited|llc|inc|corp|ltd|llp|co|sarl|sas|sa|sci|eurl|sasu)\\\\b',
    ' ', 'g'));

CREATE OR REPLACE MACRO compact_name(s) AS
    regexp_replace(norm_text(s), '\\\\s+', '', 'g');

CREATE OR REPLACE MACRO get_token(s, n) AS
    CASE WHEN array_length(string_split(trim(coalesce(s,'')), ' ')) >= n
         THEN string_split(trim(coalesce(s,'')), ' ')[n]
         ELSE '' END;

CREATE OR REPLACE MACRO clean_addr(s) AS
    lower(trim(regexp_replace(
        regexp_replace(coalesce(s,''), '[^\\\\w\\\\s]', ' ', 'g'),
        '\\\\s+', ' ', 'g')));
\"\"\")

print("DuckDB preprocessing macros successfully registered.")
""")

# --- Section 6: S1 Entity Sampling & Train/Val Split ---
add_md("""
### Section 6: S1 Entity Sampling & Train/Validation Split

**Anti-Leakage Guarantee**:
Candidate pairs belonging to a Source 1 entity remain **entirely** within either the training set or the validation set.
We split 80% S1 entities for training and 20% for validation, stratified by country and singleton status.
""")
add_code("""
# Configurable S1 limit (10000, 50000, 150000, or None for all)
TRAIN_S1_LIMIT = 150000
VALIDATION_FRACTION = 0.20
CHUNK_SIZE = 50000
TOP_K = 25
HARD_NEG_RATIO = 3.0

print(f"Sampling {TRAIN_S1_LIMIT if TRAIN_S1_LIMIT else 'ALL'} S1 entities with SEED={SEED}...")
sample_clause = f"USING SAMPLE {TRAIN_S1_LIMIT} (reservoir, {SEED})" if TRAIN_S1_LIMIT else ""

con.execute(f\"\"\"
CREATE OR REPLACE TABLE s1_sample AS
SELECT 
    entity_id,
    coalesce(country, '') as country,
    business_name as raw_name,
    business_address as raw_addr,
    norm_text(business_name) as norm_name,
    clean_legal(business_name) as core_name,
    clean_addr(business_address) as norm_addr,
    get_token(clean_legal(business_name), 1) as p1,
    get_token(clean_legal(business_name), 2) as p2,
    get_token(clean_legal(business_name), 3) as p3,
    left(compact_name(business_name), 4) as cmp4,
    left(compact_name(business_name), 5) as cmp5,
    regexp_extract(clean_addr(business_address), '\\\\b(\\\\d{{3,}})\\\\b', 1) as postal,
    regexp_extract(clean_addr(business_address), '^(\\\\d+)', 1) as door,
    get_token(clean_addr(business_address), 1) as a1,
    get_token(clean_addr(business_address), 2) as a2
FROM read_csv_auto('{s1_path}', delim='\\\\t', header=true, quote='')
{sample_clause};
\"\"\")

# Join ground truth and compute singleton indicator
con.execute(f\"\"\"
CREATE OR REPLACE TABLE s1_gt AS
SELECT 
    s.*,
    g.matched_entity_ids,
    case when g.matched_entity_ids is null or trim(g.matched_entity_ids) = '' then 0
         else length(g.matched_entity_ids) - length(replace(g.matched_entity_ids, ',', '')) + 1 end as match_count,
    case when g.matched_entity_ids is null or trim(g.matched_entity_ids) = '' then 1 else 0 end as is_singleton
FROM s1_sample s
LEFT JOIN read_csv_auto('{gt_path}', delim='\\\\t', header=true, quote='') g
ON s.entity_id = g.source1_entity_id;
\"\"\")

# Stratified S1-level split
con.execute(f\"\"\"
CREATE OR REPLACE TABLE s1_split AS
SELECT *,
    case when row_number() over (partition by country, is_singleton order by hash(entity_id)) <= count(*) over (partition by country, is_singleton) * (1.0 - {VALIDATION_FRACTION})
         then 'train' else 'val' end as split
FROM s1_gt;
\"\"\")

# Display distribution statistics
dist_df = con.execute(\"\"\"
SELECT split, country, 
       count(*) as s1_count, 
       sum(is_singleton) as singleton_count,
       round(sum(is_singleton)::float / count(*) * 100, 2) as singleton_pct,
       sum(match_count) as total_true_matches,
       round(avg(match_count), 2) as avg_matches_per_s1
FROM s1_split GROUP BY split, country ORDER BY split, country
\"\"\").df()

print("\\nTrain / Validation S1 Entity Distributions:")
display(dist_df)
""")

# --- Section 7: 8-Pass Candidate Generation ---
add_md("""
### Section 7: Final 8-Pass Blocking & Candidate Generation

Execute the finalized **8-pass blocking architecture**:
- Strictly partitioned by country.
- Passes 1–8 (2-token prefix, token+postal, token+door, compact 5-char prefix, long single token, door+postal anchor, door+street token 1, door+street token 2).
- Fast Token-Jaccard Pre-ranking: $\\text{Score} = 0.6 \\times \\text{NameJaccard} + 0.4 \\times \\text{AddrJaccard}$.
- Keep top $K=25$ candidates per Source 1 entity.
""")
add_code("""
# Build target candidate pool: true targets + background distractors
print("Building Target Pool (true targets + 300,000 background distractors)...")
t0_pool = time.time()

con.execute(\"\"\"
CREATE OR REPLACE TABLE true_target_ids AS
SELECT DISTINCT unnest(string_split(matched_entity_ids, ',')) as target_id
FROM s1_gt WHERE matched_entity_ids is not null and trim(matched_entity_ids) != '';
\"\"\")

con.execute(f\"\"\"
CREATE OR REPLACE TABLE target_pool AS
SELECT 
    entity_id, coalesce(country,'') as country,
    business_name as raw_name, business_address as raw_addr,
    norm_text(business_name) as norm_name,
    clean_legal(business_name) as core_name,
    clean_addr(business_address) as norm_addr,
    get_token(clean_legal(business_name), 1) as p1,
    get_token(clean_legal(business_name), 2) as p2,
    get_token(clean_legal(business_name), 3) as p3,
    left(compact_name(business_name), 4) as cmp4,
    left(compact_name(business_name), 5) as cmp5,
    regexp_extract(clean_addr(business_address), '\\\\b(\\\\d{{3,}})\\\\b', 1) as postal,
    regexp_extract(clean_addr(business_address), '^(\\\\d+)', 1) as door,
    get_token(clean_addr(business_address), 1) as a1,
    get_token(clean_addr(business_address), 2) as a2
FROM (
    SELECT * FROM read_csv_auto('{s2_path}', delim='\\\\t', header=true, quote='')
    WHERE entity_id IN (SELECT target_id FROM true_target_ids)
    UNION ALL
    SELECT * FROM read_csv_auto('{s3_path}', delim='\\\\t', header=true, quote='')
    WHERE entity_id IN (SELECT target_id FROM true_target_ids)
    UNION ALL
    (SELECT * FROM read_csv_auto('{s2_path}', delim='\\\\t', header=true, quote='') USING SAMPLE 150000 (reservoir, {SEED}))
    UNION ALL
    (SELECT * FROM read_csv_auto('{s3_path}', delim='\\\\t', header=true, quote='') USING SAMPLE 150000 (reservoir, {SEED}))
);
\"\"\")

target_cnt = con.execute("SELECT count(*) FROM target_pool").fetchone()[0]
print(f"Target candidate pool built: {target_cnt:,} records in {time.time()-t0_pool:.2f}s")

# Execute 8-pass blocking
print("\\nExecuting 8-pass blocking across countries...")
t_block_start = time.time()

con.execute(\"\"\"
CREATE OR REPLACE TABLE raw_candidate_pairs AS
SELECT sub.entity_id as s1_id, sub.target_id, count(*) as pass_count
FROM (
    SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.p2=t.p2 WHERE length(s.p1)>=3 AND length(s.p2)>=3
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.postal=t.postal WHERE length(s.p1)>=3 AND s.postal!=''
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 AND s.door=t.door WHERE length(s.p1)>=3 AND s.door!=''
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.cmp5=t.cmp5 WHERE length(s.cmp5)>=5
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.p1=t.p1 WHERE length(s.p1)>=5
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.postal=t.postal AND s.door=t.door WHERE s.postal!='' AND length(s.door)>=2
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.door=t.door AND s.a1=t.a1 WHERE length(s.door)>=2 AND length(s.a1)>=4
    UNION ALL
    SELECT s.entity_id, t.entity_id as target_id FROM s1_split s JOIN target_pool t ON s.country=t.country AND s.door=t.door AND s.a2=t.a2 WHERE length(s.door)>=2 AND length(s.a2)>=4
) sub
GROUP BY sub.entity_id, sub.target_id;
\"\"\")

raw_pairs = con.execute("SELECT count(*) FROM raw_candidate_pairs").fetchone()[0]
print(f"Generated {raw_pairs:,} distinct raw candidate pairs in {time.time()-t_block_start:.2f}s.")

# Pre-ranking with token Jaccard & Top-K Pruning
print(f"Ranking candidates and selecting top K={TOP_K} per S1 entity...")
con.execute(f\"\"\"
CREATE OR REPLACE TABLE ranked_candidates AS
WITH pair_scored AS (
    SELECT 
        c.s1_id, c.target_id, c.pass_count,
        s.split, s.country as s1_country, s.raw_name as s1_raw_name, s.raw_addr as s1_raw_addr, s.norm_name as s1_norm_name, s.norm_addr as s1_norm_addr, s.postal as s1_postal, s.door as s1_door,
        t.country as t_country, t.raw_name as t_raw_name, t.raw_addr as t_raw_addr, t.norm_name as t_norm_name, t.norm_addr as t_norm_addr, t.postal as t_postal, t.door as t_door,
        (0.6 * (len(list_intersect(string_split(s.norm_name, ' '), string_split(t.norm_name, ' ')))::float / 
                nullif(len(list_distinct(list_concat(string_split(s.norm_name, ' '), string_split(t.norm_name, ' ')))), 0)) +
         0.4 * (len(list_intersect(string_split(s.norm_addr, ' '), string_split(t.norm_addr, ' ')))::float / 
                nullif(len(list_distinct(list_concat(string_split(s.norm_addr, ' '), string_split(t.norm_addr, ' ')))), 0))) as jaccard_score
    FROM raw_candidate_pairs c
    JOIN s1_split s ON c.s1_id = s.entity_id
    JOIN target_pool t ON c.target_id = t.entity_id
)
SELECT *,
    row_number() over (partition by s1_id order by coalesce(jaccard_score, 0) desc, pass_count desc) as candidate_rank
FROM pair_scored;

CREATE OR REPLACE TABLE topk_candidates AS
SELECT * FROM ranked_candidates WHERE candidate_rank <= {TOP_K};
\"\"\")

topk_cnt = con.execute("SELECT count(*) FROM topk_candidates").fetchone()[0]
per_s1 = np.array([r[0] for r in con.execute("SELECT count(*) FROM topk_candidates GROUP BY s1_id").fetchall()])
print(f"\\nCandidate Generation Results:")
print(f"  Total Retained Pairs: {topk_cnt:,} across {len(per_s1):,} S1 entities")
print(f"  Avg Candidates/S1:    {np.mean(per_s1):.1f}")
print(f"  Median:               {np.median(per_s1):.0f}")
print(f"  95th Percentile:      {np.percentile(per_s1, 95):.0f}")
print(f"  Maximum:              {np.max(per_s1)}")
""")

# --- Section 8 & 9: Labels & Hard Negatives ---
add_md("""
### Section 8 & 9: Positive / Negative Labels & Hard-Negative Prioritization

- **Positives (`label = 1`)**: Candidate pairs verified present in `train_ground_truth.tsv`.
- **Negatives (`label = 0`)**: Candidate pairs produced by blocking that are NOT ground-truth matches.
- **Hard-Negative Prioritization**: Rank-1..5 candidates with high token similarity but address/business mismatch are prioritized at a configurable ratio (default $3.0:1$).
""")
add_code("""
# Label candidate pairs against ground truth
con.execute(\"\"\"
CREATE OR REPLACE TABLE labeled_topk AS
SELECT 
    c.*,
    case when g.target_id is not null then 1 else 0 end as label
FROM topk_candidates c
LEFT JOIN (
    SELECT entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) as target_id
    FROM s1_gt WHERE matched_entity_ids is not null
) g ON c.s1_id = g.s1_id AND c.target_id = g.target_id;
\"\"\")

label_dist = con.execute(\"\"\"
SELECT split, label, count(*) as pair_count
FROM labeled_topk GROUP BY split, label ORDER BY split, label
\"\"\").df()
print("Raw Candidate Label Distribution:")
display(label_dist)

# Sample balanced training set with hard negatives
train_pos_cnt = con.execute("SELECT count(*) FROM labeled_topk WHERE split='train' AND label=1").fetchone()[0]
max_train_negs = int(train_pos_cnt * HARD_NEG_RATIO)

con.execute(f\"\"\"
CREATE OR REPLACE TABLE train_pairs_selected AS
SELECT * FROM labeled_topk WHERE split='train' AND label=1
UNION ALL
(
    SELECT * FROM labeled_topk 
    WHERE split='train' AND label=0
    ORDER BY candidate_rank ASC, coalesce(jaccard_score, 0) DESC
    LIMIT {max_train_negs}
);
\"\"\")

final_train_cnt = con.execute("SELECT count(*) FROM train_pairs_selected").fetchone()[0]
final_neg_cnt = final_train_cnt - train_pos_cnt
print(f"\\nBalanced Training Set Created:")
print(f"  Positive Pairs:  {train_pos_cnt:,}")
print(f"  Hard Negatives:  {final_neg_cnt:,}")
print(f"  Ratio (Neg:Pos): {final_neg_cnt / train_pos_cnt:.2f} : 1")
print(f"  Total Pairs:     {final_train_cnt:,}")
""")

# --- Section 10: Feature Engineering ---
add_md("""
### Section 10: Feature Engineering (24 Candidate-Pair Features)

We compute 24 similarity and interaction features using **RapidFuzz** exclusively on the final top-$K$ candidate pairs:
1. `name_exact_raw`
2. `name_exact_norm`
3. `name_token_sort_ratio`
4. `name_token_set_ratio`
5. `name_jaro_winkler`
6. `name_levenshtein_ratio`
7. `name_char_sim`
8. `name_substring_containment`
9. `addr_exact_norm`
10. `addr_token_sort_ratio`
11. `addr_token_set_ratio`
12. `addr_token_jaccard`
13. `addr_char_sim`
14. `addr_num_overlap`
15. `addr_postal_match`
16. `country_agreement`
17. `s1_name_missing`
18. `t_name_missing`
19. `s1_addr_missing`
20. `t_addr_missing`
21. `candidate_rank`
22. `blocking_pass_count`
23. `name_x_addr`
24. `name_diff_addr`
""")
add_code("""
FEATURE_NAMES = [
    "name_exact_raw", "name_exact_norm", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaro_winkler", "name_levenshtein_ratio", "name_char_sim", "name_substring_containment",
    "addr_exact_norm", "addr_token_sort_ratio", "addr_token_set_ratio", "addr_token_jaccard",
    "addr_char_sim", "addr_num_overlap", "addr_postal_match",
    "country_agreement", "s1_name_missing", "t_name_missing", "s1_addr_missing", "t_addr_missing",
    "candidate_rank", "blocking_pass_count", "name_x_addr", "name_diff_addr"
]

from rapidfuzz import fuzz, distance

def compute_pair_features(
    s1_raw_name, s1_norm_name, s1_addr, s1_postal, s1_door,
    t_raw_name, t_norm_name, t_addr, t_postal, t_door,
    s1_country, t_country, rank, pass_count
):
    # Name features
    exact_raw = 1.0 if s1_raw_name and s1_raw_name == t_raw_name else 0.0
    exact_norm = 1.0 if s1_norm_name and s1_norm_name == t_norm_name else 0.0
    name_sort = fuzz.token_sort_ratio(s1_norm_name, t_norm_name) / 100.0
    name_set = fuzz.token_set_ratio(s1_norm_name, t_norm_name) / 100.0
    jw = distance.JaroWinkler.similarity(s1_norm_name, t_norm_name)
    lev = distance.Levenshtein.normalized_similarity(s1_norm_name, t_norm_name)
    name_char_sim = fuzz.ratio(s1_norm_name, t_norm_name) / 100.0
    name_substr = 1.0 if (s1_norm_name and t_norm_name and (s1_norm_name in t_norm_name or t_norm_name in s1_norm_name)) else 0.0

    # Address features
    addr_exact = 1.0 if s1_addr and s1_addr == t_addr else 0.0
    addr_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
    addr_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0

    s1_a_tok = set(s1_addr.split()) if s1_addr else set()
    t_a_tok = set(t_addr.split()) if t_addr else set()
    addr_jaccard = len(s1_a_tok & t_a_tok) / len(s1_a_tok | t_a_tok) if (s1_a_tok or t_a_tok) else 0.0
    addr_char_sim = fuzz.ratio(s1_addr, t_addr) / 100.0

    # Number overlap
    s1_nums = {w for w in s1_a_tok if w.isdigit()}
    t_nums = {w for w in t_a_tok if w.isdigit()}
    if s1_nums and t_nums:
        num_overlap = len(s1_nums & t_nums) / len(s1_nums | t_nums)
    elif not s1_nums and not t_nums:
        num_overlap = 0.5
    else:
        num_overlap = 0.0

    postal_match = 1.0 if s1_postal and t_postal and s1_postal == t_postal else 0.0

    # Interaction & auxiliary
    country_agree = 1.0 if s1_country == t_country else 0.0
    s1_name_miss = 1.0 if not s1_norm_name else 0.0
    t_name_miss = 1.0 if not t_norm_name else 0.0
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

# Extract feature matrices
print("Extracting features for training and validation pairs...")
t_feat_start = time.time()

df_train_pairs = con.execute("SELECT * FROM train_pairs_selected").df()
df_val_pairs = con.execute("SELECT * FROM labeled_topk WHERE split='val'").df()

def build_features(df):
    n = len(df)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    
    s1_raw = df["s1_raw_name"].fillna("").values
    s1_norm = df["s1_norm_name"].fillna("").values
    s1_addr = df["s1_norm_addr"].fillna("").values
    s1_postal = df["s1_postal"].fillna("").values
    s1_door = df["s1_door"].fillna("").values
    s1_country = df["s1_country"].fillna("").values

    t_raw = df["t_raw_name"].fillna("").values
    t_norm = df["t_norm_name"].fillna("").values
    t_addr = df["t_norm_addr"].fillna("").values
    t_postal = df["t_postal"].fillna("").values
    t_door = df["t_door"].fillna("").values
    t_country = df["t_country"].fillna("").values

    ranks = df["candidate_rank"].values
    pass_counts = df["pass_count"].values

    for i in range(n):
        X[i] = compute_pair_features(
            s1_raw[i], s1_norm[i], s1_addr[i], s1_postal[i], s1_door[i],
            t_raw[i], t_norm[i], t_addr[i], t_postal[i], t_door[i],
            s1_country[i], t_country[i], ranks[i], pass_counts[i]
        )
    return X, y

X_train, y_train = build_features(df_train_pairs)
X_val, y_val = build_features(df_val_pairs)
print(f"Features computed in {time.time()-t_feat_start:.2f}s!")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape:   {X_val.shape}")
""")

# --- Section 11 & 12: Train Models ---
add_md("""
### Section 11 & 12: Model Training (Logistic Regression vs. XGBoost)

Train both models using identical training and validation examples/features.
""")
add_code("""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Model A: Logistic Regression with Feature Scaling
print("Training Model A: Logistic Regression...")
t_lr_start = time.time()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr_model = LogisticRegression(C=1.0, max_iter=500, random_state=SEED, solver="lbfgs")
lr_model.fit(X_train_scaled, y_train)
lr_time = time.time() - t_lr_start
lr_val_probs = lr_model.predict_proba(X_val_scaled)[:, 1]
print(f"Logistic Regression trained in {lr_time:.2f}s.")

# Model B: XGBoost
print("\\nTraining Model B: XGBoost...")
t_xgb_start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=SEED,
    n_jobs=8
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - t_xgb_start
xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
print(f"XGBoost trained in {xgb_time:.2f}s.")

# Top Feature Importances
importances = xgb_model.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("\\nTop 10 Feature Importances (XGBoost):")
for r, idx in enumerate(top_idx, 1):
    print(f"  {r:>2}. {FEATURE_NAMES[idx]:<28} : {importances[idx]*100:5.2f}%")
""")

# --- Section 13 & 14: Threshold Search & Macro F0.5 ---
add_md("""
### Section 13 & 14: Threshold Search & Exact Macro $F_{0.5}$ Evaluation

Evaluate across threshold grid $[0.30, 0.95]$ with step size $0.05$.
The competition metric is **Macro $F_{0.5}$**:
$$F_{0.5} = \\frac{1.25 \\times \\text{Precision} \\times \\text{Recall}}{0.25 \\times \\text{Precision} + \\text{Recall}}$$
**Singleton Rules**:
- True singleton + predicted empty $\\rightarrow F_{0.5} = 1.0$
- True singleton + predicted non-empty $\\rightarrow F_{0.5} = 0.0$
- True non-singleton + predicted empty $\\rightarrow F_{0.5} = 0.0$
""")
add_code("""
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Build validation ground truth mapping
val_gt_rows = con.execute("SELECT entity_id, matched_entity_ids FROM s1_split WHERE split='val'").fetchall()
val_gt = {}
for sid, mstr in val_gt_rows:
    if mstr and str(mstr).strip():
        val_gt[sid] = {m.strip() for m in str(mstr).split(",") if m.strip()}
    else:
        val_gt[sid] = set()

def compute_entity_f05(true_matches, pred_matches):
    if not true_matches:
        return 1.0 if not pred_matches else 0.0
    if not pred_matches:
        return 0.0
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_matches)
    rec = tp / len(true_matches)
    denom = 0.25 * prec + rec
    return (1.25 * prec * rec) / denom if denom > 0 else 0.0

def evaluate_threshold_grid(val_probs, model_name):
    scores_by_s1 = defaultdict(list)
    val_s1_ids = df_val_pairs["s1_id"].values
    val_target_ids = df_val_pairs["target_id"].values
    for s1_id, tgt_id, prob in zip(val_s1_ids, val_target_ids, val_probs):
        scores_by_s1[s1_id].append((tgt_id, float(prob)))

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
    results = []
    best_row = None
    best_f05 = -1.0

    for tau in thresholds:
        f05_list = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_preds = 0

        for sid, true_set in val_gt.items():
            cands = scores_by_s1.get(sid, [])
            pred_set = {tid for tid, prob in cands if prob >= tau}
            f05_list.append(compute_entity_f05(true_set, pred_set))
            total_tp += len(true_set & pred_set)
            total_fp += len(pred_set - true_set)
            total_fn += len(true_set - pred_set)
            total_preds += len(pred_set)

        macro_f05 = float(np.mean(f05_list))
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        avg_matches = total_preds / len(val_gt)

        row = {
            "model": model_name,
            "threshold": tau,
            "macro_f05": round(macro_f05, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "avg_matches_per_s1": round(avg_matches, 3)
        }
        results.append(row)
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_row = row

    return pd.DataFrame(results), best_row

lr_df, lr_best = evaluate_threshold_grid(lr_val_probs, "LogisticRegression")
xgb_df, xgb_best = evaluate_threshold_grid(xgb_val_probs, "XGBoost")

# Calculate secondary metrics
lr_roc = roc_auc_score(y_val, lr_val_probs)
lr_pr = average_precision_score(y_val, lr_val_probs)
xgb_roc = roc_auc_score(y_val, xgb_val_probs)
xgb_pr = average_precision_score(y_val, xgb_val_probs)

lr_f1 = f1_score(y_val, (lr_val_probs >= lr_best["threshold"]).astype(int))
xgb_f1 = f1_score(y_val, (xgb_val_probs >= xgb_best["threshold"]).astype(int))

comp_summary = pd.DataFrame([
    {"Model": "Logistic Regression", "Optimal tau*": lr_best["threshold"], "Macro F0.5": lr_best["macro_f05"], "Precision": lr_best["precision"], "Recall": lr_best["recall"], "F1": round(lr_f1, 4), "ROC-AUC": round(lr_roc, 4), "PR-AUC": round(lr_pr, 4), "Train Time": f"{lr_time:.2f}s"},
    {"Model": "XGBoost", "Optimal tau*": xgb_best["threshold"], "Macro F0.5": xgb_best["macro_f05"], "Precision": xgb_best["precision"], "Recall": xgb_best["recall"], "F1": round(xgb_f1, 4), "ROC-AUC": round(xgb_roc, 4), "PR-AUC": round(xgb_pr, 4), "Train Time": f"{xgb_time:.2f}s"}
])

print("=" * 80)
print("  FINAL VALIDATION MODEL COMPARISON (Exact Macro F0.5)")
print("=" * 80)
display(comp_summary)
""")

# --- Section 15: Error Analysis ---
add_md("""
### Section 15: Error Analysis

Categorize false positives and false negatives on the validation set for the winning model.
""")
add_code("""
winning_model = "LogisticRegression" if lr_best["macro_f05"] >= xgb_best["macro_f05"] else "XGBoost"
winning_best = lr_best if winning_model == "LogisticRegression" else xgb_best
winning_probs = lr_val_probs if winning_model == "LogisticRegression" else xgb_val_probs

df_val_eval = df_val_pairs.copy()
df_val_eval["pred_prob"] = winning_probs
df_val_eval["pred_match"] = (winning_probs >= winning_best["threshold"]).astype(int)

fp_df = df_val_eval[(df_val_eval["label"] == 0) & (df_val_eval["pred_match"] == 1)]
fn_df = df_val_eval[(df_val_eval["label"] == 1) & (df_val_eval["pred_match"] == 0)]

print(f"Error Analysis for Winning Model: {winning_model} at tau* = {winning_best['threshold']:.2f}")
print(f"  Total False Positives: {len(fp_df):,}")
print(f"  Total False Negatives: {len(fn_df):,}")

# Breakdown False Positives
fp_cats = defaultdict(int)
for _, r in fp_df.iterrows():
    s1_n, t_n = r["s1_norm_name"], r["t_norm_name"]
    s1_a, t_a = r["s1_norm_addr"], r["t_norm_addr"]
    nsim = fuzz.token_sort_ratio(s1_n, t_n)
    asim = fuzz.token_sort_ratio(s1_a, t_a)
    if nsim >= 85 and asim < 40:
        fp_cats["same_name_diff_address"] += 1
    elif asim >= 80 and nsim < 40:
        fp_cats["same_address_diff_business"] += 1
    elif r["s1_door"] and r["t_door"] and r["s1_door"] != r["t_door"] and nsim >= 70:
        fp_cats["address_number_conflict"] += 1
    elif len(s1_n) <= 6 or len(t_n) <= 6:
        fp_cats["abbreviation_collision"] += 1
    else:
        fp_cats["generic_name_collision"] += 1

fp_summary = pd.DataFrame([{"Category": k, "Count": v, "Percentage": f"{v/len(fp_df)*100:.1f}%"} for k, v in sorted(fp_cats.items(), key=lambda x: -x[1])])
print("\\nFalse Positive Categories:")
display(fp_summary)
""")

# --- Section 16 & 17: Save Artifacts & Results ---
add_md("### Section 16 & 17: Save Model Artifacts and Results")
add_code("""
import joblib

artifacts_dir = "artifacts"
results_dir = "results"
os.makedirs(os.path.join(artifacts_dir, "logistic_regression"), exist_ok=True)
os.makedirs(os.path.join(artifacts_dir, "xgboost"), exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Save models
joblib.dump(lr_model, os.path.join(artifacts_dir, "logistic_regression", "model.joblib"))
joblib.dump(scaler, os.path.join(artifacts_dir, "logistic_regression", "scaler.joblib"))
xgb_model.save_model(os.path.join(artifacts_dir, "xgboost", "xgboost_model.json"))

# Save configurations
with open(os.path.join(artifacts_dir, "feature_config.json"), "w") as f:
    json.dump({"feature_names": FEATURE_NAMES, "count": len(FEATURE_NAMES)}, f, indent=2)

with open(os.path.join(artifacts_dir, "selected_model.json"), "w") as f:
    json.dump({
        "selected_model": winning_model,
        "optimal_threshold": winning_best["threshold"],
        "validation_macro_f05": winning_best["macro_f05"],
        "validation_precision": winning_best["precision"],
        "validation_recall": winning_best["recall"]
    }, f, indent=2)

# Save result tables
comp_summary.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
pd.concat([lr_df, xgb_df]).to_csv(os.path.join(results_dir, "threshold_results.csv"), index=False)
df_val_eval[["s1_id", "target_id", "label", "pred_prob", "pred_match"]].head(10000).to_csv(
    os.path.join(results_dir, "validation_predictions.csv"), index=False
)

print(f"Artifacts and Results successfully saved to `{artifacts_dir}/` and `{results_dir}/`.")
""")

# --- Section 18: Summary ---
add_md("""
### Section 18: Final Experiment Summary & Next Steps

1. **Candidate Blocking & Generation**: Finalized 8-pass blocking with Token-Jaccard pre-ranking ($K=25$) retrieved candidate pairs with zero OOM overhead.
2. **Modeling Comparison**: Logistic Regression vs. XGBoost evaluated on exact competition Macro $F_{0.5}$ with singleton handling.
3. **Threshold Selection**: Optimal threshold $\\tau^*$ selected on validation data.
4. **Reproducibility**: Artifacts are portable and reloadable for downstream full inference.
""")

# Build final notebook structure
notebook_dict = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "train_matching_model_colab.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Google Colab Notebook successfully written to: {notebook_path}")
""")
