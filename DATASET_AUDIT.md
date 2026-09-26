# Comprehensive Dataset Audit & Evidence-Based Feasibility Analysis
**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Audit Scope:** Training & Test TSV Files, Ground Truth Linkages, Noise Distributions, Hard Negatives, Empirical Blocking Benchmarks  
**Compliance Guarantee:** Zero external lookups, zero APIs, zero external databases. Ground-truth-verified facts only.

---

## 1. Dataset Structure & TSV Parsing Integrity

### 1.1 Available Files, Paths & Sizes
| File Name | Split | Path | Size (MB) | Rows | Columns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | Train | `dataset/train/train_source1.tsv` | 200.34 MB | **2,206,821** | 4 |
| `train_source2.tsv` | Train | `dataset/train/train_source2.tsv` | 466.63 MB | **5,034,616** | 4 |
| `train_source3.tsv` | Train | `dataset/train/train_source3.tsv` | 480.37 MB | **5,285,603** | 4 |
| `train_ground_truth.tsv` | Train | `dataset/train/train_ground_truth.tsv` | 121.13 MB | **2,206,821** | 2 |
| `test_source1.tsv` | Test | `dataset/test/test_source1.tsv` | 166.91 MB | **1,732,544** | 4 |
| `test_source2.tsv` | Test | `dataset/test/test_source2.tsv` | 485.86 MB | **4,887,273** | 4 |
| `test_source3.tsv` | Test | `dataset/test/test_source3.tsv` | 482.56 MB | **5,082,316** | 4 |

### 1.2 Schema and Data Types
* **Source Records (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`):**
  - `entity_id` (`VARCHAR`): Unique identifier with source prefix (`S1-`, `S2-`, `S3-`).
  - `business_name` (`VARCHAR`): Trade or legal business entity name.
  - `business_address` (`VARCHAR`): Physical address string.
  - `country` (`VARCHAR`): Country string label (`US`, `India`, `France`).
* **Ground Truth (`train_ground_truth.tsv`):**
  - `source1_entity_id` (`VARCHAR`): Source 1 reference ID.
  - `matched_entity_ids` (`VARCHAR`): Comma-separated list of matching Source 2 and/or Source 3 IDs.

### 1.3 TSV Parsing Verification
- Tab separation (`\t`) was verified on all files. Business address strings and names frequently contain commas (`,`), semicolons (`;`), quotes (`"`), slashes (`/`), and non-ASCII characters. Reading without an explicit `sep="\t"` concatenates records. 
- All files parsed cleanly without schema errors, line shifts, or truncated columns.

---

## 2. Detailed Source Analysis (Source 1 vs Source 2 vs Source 3)

### 2.1 Entity Counts & Duplication
| Metric | `train_source1` | `train_source2` | `train_source3` | `test_source1` | `test_source2` | `test_source3` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Rows** | 2,206,821 | 5,034,616 | 5,285,603 | 1,732,544 | 4,887,273 | 5,082,316 |
| **Unique IDs** | 2,206,821 (100%) | 5,034,616 (100%) | 5,285,603 (100%) | 1,732,544 (100%) | 4,887,273 (100%) | 5,082,316 (100%) |
| **Unique Names** | 1,539,229 (69.75%) | 4,402,009 (87.43%) | 4,651,609 (88.01%) | 1,238,867 (71.51%) | 4,311,041 (88.21%) | 4,521,929 (88.97%) |
| **Duplicate Names**| 667,592 (30.25%) | 632,607 (12.57%) | 633,994 (11.99%) | 493,677 (28.49%) | 576,232 (11.79%) | 560,387 (11.03%) |

> [!NOTE]
> **Key Finding:** In Source 1, **over 30% of business names are repeated strings** (common chains, clinics, franchises, generic names like "Chiropractic Group", "Meridian LLC"). Name alone cannot uniquely identify an entity. Address disambiguation is vital.

### 2.2 Country Distribution
| Country | `train_source1` | `train_source2` | `train_source3` | `test_source1` | `test_source2` | `test_source3` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **United States (US)** | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.98%) | 663,106 (38.27%) | 1,871,330 (38.29%) | 1,945,701 (38.28%) |
| **India** | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.02%) | 809,986 (46.75%) | 2,312,565 (47.32%) | 2,405,000 (47.32%) |
| **France** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | **259,452 (14.98%)** | **703,378 (14.39%)** | **731,615 (14.40%)** |

> [!WARNING]
> **Zero-Shot Country Generalization:** France accounts for **15.0% of Test Source 1 (259k entities)** and **1.43M candidate records**, but is completely absent from the training set. Models must rely on language-agnostic character features, universal string similarities, and numerical address tokens rather than hardcoded US states or Indian PIN codes.

### 2.3 Missing Value Analysis
* `entity_id`: **0.00% missing** across all files.
* `business_name`: **0.00% missing** across all files.
* `country`: **0.00% missing** across all files.
* `business_address`:
  - `train_source1`: **0 missing (0.00%)**
  - `train_source2`: **168,967 missing (3.36%)**
  - `train_source3`: **175,916 missing (3.33%)**
  - `test_source1`: **0 missing (0.00%)**
  - `test_source2`: **129,408 missing (2.65%)**
  - `test_source3`: **136,098 missing (2.68%)**

### 2.4 Name and Address Length Statistics
| File | Name Len (Min / Avg / Med / p95 / Max) | Address Len (Min / Avg / Med / p95 / Max) |
| :--- | :--- | :--- |
| `train_source1` | 3 / 24.03 / 24 / 37 / 105 | 11 / 52.07 / 41 / 103 / 256 |
| `train_source2` | 2 / 25.10 / 25 / 40 / 104 | 0 / 46.23 / 37 / 96 / 249 |
| `train_source3` | 2 / 25.20 / 25 / 42 / 123 | 0 / 46.71 / 42 / 91 / 240 |
| `test_source1` | 3 / 23.84 / 24 / 36 / 92 | 11 / 57.21 / 50 / 105 / 268 |
| `test_source2` | 2 / 25.70 / 25 / 42 / 102 | 0 / 50.41 / 43 / 99 / 269 |
| `test_source3` | 2 / 25.66 / 25 / 42 / 103 | 0 / 48.74 / 43 / 94 / 267 |

---

## 3. Ground Truth Analysis

### 3.1 Match Count Distribution
Across all 2,206,821 Source 1 entities in `train_ground_truth.tsv`:

| Matches ($K$) | Number of S1 Entities | Percentage | Cumulative % |
| :--- | :--- | :--- | :--- |
| **0 (Singletons)** | **123,247** | **5.58%** | 5.58% |
| **1 match** | **119,157** | **5.40%** | 10.98% |
| **2 matches** | **375,212** | **17.00%** | 27.98% |
| **3 matches** | **530,841** | **24.05%** | 52.03% |
| **4 matches** | **484,115** | **21.94%** | 73.97% |
| **5 matches** | **321,957** | **14.59%** | 88.56% |
| **6 matches** | **164,868** | **7.47%** | 96.03% |
| **7 matches** | **63,968** | **2.90%** | 98.93% |
| **8 matches** | **18,680** | **0.85%** | 99.78% |
| **9 matches** | **4,205** | **0.19%** | 99.97% |
| **10 matches** | **534** | **0.02%** | 99.99% |
| **11 matches** | **37** | **0.00%** | 100.00% |

### 3.2 Ground Truth Aggregate Statistics
* **Total Ground Truth Links:** **7,638,365**
* **Average Matches per S1:** **3.461**
* **Median Matches per S1:** **3.0**
* **75th Percentile:** 5.0 | **90th Percentile:** 6.0 | **95th Percentile:** 6.0 | **99th Percentile:** 8.0
* **Target Origin Distribution:**
  - Source 3 matches: **3,944,746 (51.64%)**
  - Source 2 matches: **3,693,619 (48.36%)**
* **Internal Duplicates:** An S1 entity frequently matches multiple records from Source 2 and multiple records from Source 3 (e.g. 3 from S2 and 2 from S3).

### 3.3 Data Integrity & Validation Checks
* **S1 Self-matches:** **0 (0.0%)**. No entity in ground truth matches an S1 ID.
* **Orphan / Missing Target IDs:** **0 (0.0%)**. All 3,693,619 S2 IDs exist in `train_source2.tsv`, and all 3,944,746 S3 IDs exist in `train_source3.tsv`.
* **Duplicate Targets within single S1 record:** **0 (0.0%)**. All ID lists in ground truth are strictly deduplicated.
* **Cross-Country Matches:** **0 (EXACTLY ZERO)** across all 7,638,365 true links. Country agreement is 100.0%.

---

## 4. Empirical Noise Analysis (Representative Examples)

From actual matched pairs in the dataset, 11 distinct noise patterns were identified:

### 4.1 Legal Suffix Variations
| Source 1 Name | Target Name | Country |
| :--- | :--- | :--- |
| `Murshidabad Bio Private Limited` | `Murshidabad Bio` | India |
| `Behavioral Health Center of Powell` | `BEHAVIORAL HEALTH CENTER OF POWELL LTD` | US |
| `Aviva Industries Group Private Limited` | `Aviva Industries Group` | India |
| `Saint Presbyterian Church` | `Saint Presbyterian Church Corporation` | US |

### 4.2 Typos, Character Corruptions & OCR Artifacts
| Source 1 Name | Target Name | Country | Type of Corruption |
| :--- | :--- | :--- | :--- |
| `Soledad Liquor Inc` | `Soledad Líquor lnc` | US | Accent on `i`, lowercase `l` replacing `I` in `Inc` |
| `Hall Japan LLC` | `Hall Jpn LLC` | US | Abbreviation (`Japan` $\to$ `Jpn`) |
| `Dickison, Roberts and Faulk Xsolla LLC` | `Dickison, Rdbrts and Faulk Xsolla LLC` | US | Typo (`Roberts` $\to$ `Rdbrts`) |
| `Pitambra Industries Group Private Limited`| `Pitambra Indutsrges Group Private [Limited]`| India | Transposed letters + brackets |

### 4.3 Word-Order Permutations
| Source 1 Name | Target Name | Country |
| :--- | :--- | :--- |
| `Industries Alpha Agro Private Limited` | `Private Industries Alpha Agro Private` | India |
| `Tinley Park Regional Church Holdings` | `Holdings Tinley Park Regional Church` | US |
| `Bates, Maryanna, L.C.S.W.` | `Bates, L.C.S.W. Maryanna,` | US |
| `Lotus Blue It Private Limited` | `PRIVATE LOTUS BLUE IT LIMITED` | India |

### 4.4 Non-Latin Scripts & Transliterations
| Source 1 (Latin/English) | Target (Indic Script / Non-ASCII) | Address Match |
| :--- | :--- | :--- |
| `Bombay Finance Private Limited` | `Bombay फाइनेंस प्राइवेट लिमिटेड` | `M-577 Guru Harkishan Nagar, Paschim Vihar, Delhi` |
| `Classic Technology Private Limited` | `क्लासिक टेक्नोलॉजी प्राइवेट लिमिटेड` | `Kolekal Yan Kalina Near Crystal Plaza Santacruz East, Mumbai` |
| `Aditya Business Limited` | `आदित्य बिजनेस लिमिटेड` | `South Hajipur Dhobi Tola, Khagaria, Bihar` |

### 4.5 Domain Names as Entity Names
| Source 1 Name | Target Name | Notes |
| :--- | :--- | :--- |
| `Agro Sheron Hotels Private Limited` | `agrosheronhotels.com` | Domain name with suffixes stripped |
| `Garcia & Aguilar System` | `GARCIAAGUILARSYSTEM.COM` | Concatenated domain without spaces |
| `Steimel Capital Churchill` | `steimelcapitalchurchill.com` | Concatenated domain name |

### 4.6 Address Component Inversion & Reordering
* **S1:** `TX, Mckinney, 5617 Grove Cove Drive`
* **Target:** `5617 GROVE COVE DR, MCKINNEY, TX` (City/State preceding street address vs street address preceding City/State)

### 4.7 Landmark-Based Addresses
* **S1:** `Plot No.Pap-J-111 To 120 Near Quality Ice-Cream Shop, Bhosari, Pune, Maharashtra`
* **Target:** `PLOT 130 PLOT NO.PAP-J-111 TO 120 NEAR QUALITY ICE-CREAM SHOP, BHOSARI, PUNE, Maharashtra`

### 4.8 Number Format Variations
* **S1:** `404 Locksley Court, Fort Worth, TX`
* **Target:** `404-406 LOCKSLEY COURT, FORT WORTH, TX` (Unit ranges vs single number)

---

## 5. Quantitative Positive-Pair Analysis

Evaluated across a representative sample of **50,000 true ground-truth pairs**:

### 5.1 Exact Match Frequencies
* **Country Agreement:** **100.00%** (50,000 / 50,000)
* **Exact Raw Name Match:** **4.47%** (2,235 / 50,000)
* **Exact Normalized Name Match:** **21.58%** (10,790 / 50,000)
* **Exact Raw Address Match:** **2.26%** (1,128 / 50,000)
* **Exact Normalized Address Match:** **8.34%** (4,168 / 50,000)

> [!IMPORTANT]
> Over **78.4% of true matches have different normalized names**, and over **91.6% have different normalized addresses**. Exact string matching fails on the vast majority of true pairs.

### 5.2 Similarity Distributions on True Pairs
| Metric | Mean | Median | p10 | p25 | p75 | p90 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Name Jaro-Winkler** | 0.8790 | 0.9461 | 0.6361 | 0.8622 | 0.9818 | 1.0000 |
| **Name Token Sort Ratio** | 0.8013 | 0.8947 | 0.4324 | 0.7368 | 1.0000 | 1.0000 |
| **Name Token Set Ratio** | 0.8659 | 1.0000 | 0.4800 | 0.8750 | 1.0000 | 1.0000 |
| **Name Levenshtein Ratio** | 0.7183 | 0.7895 | 0.2857 | 0.5667 | 0.9545 | 1.0000 |
| **Address Token Sort Ratio** | 0.8100 | 0.8750 | 0.5714 | 0.7733 | 0.9425 | 1.0000 |
| **Address Token Set Ratio** | 0.8739 | 0.9375 | 0.7429 | 0.8571 | 0.9843 | 1.0000 |
| **Address Number Jaccard** | 0.7025 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |

### 5.3 Joint Signal Breakdown (Name vs Address)
* **Both Strong (Name $\ge 0.80$ AND Addr $\ge 0.80$):** **48.31%** (24,154 pairs)
* **Weak Name but Strong Addr (Name $< 0.50$, Addr $\ge 0.80$):** **7.65%** (3,824 pairs)  
  *(Driven by Indic scripts, domain names, and trade names with identical street/door locations)*
* **Strong Name but Weak Addr (Name $\ge 0.80$, Addr $< 0.50$):** **5.57%** (2,784 pairs)  
  *(Driven by missing addresses or heavy address truncation where names match closely)*
* **Both Weak (Name $< 0.50$ AND Addr $< 0.50$):** **0.90%** (449 pairs)
* **Intermediate / Mixed Signal:** **37.58%** (18,789 pairs)

---

## 6. Hard Negative Analysis (False-Match Traps)

By analyzing non-matching pairs that share strong textual overlap, three primary classes of false-positive traps were uncovered:

### 6.1 Same Name, Completely Different Locations
* **S1:** `Summit Inc` (Monroe, ME, 3 Dahlia Farm Road)  
  **Target:** `Summit Inc` (GREENSBORO, NC, 19 1/2 STARDUST TRAIL) — **NON-MATCH**
* **S1:** `Solstice LLC` (Fayetteville, AR, 151 Platinum Drive)  
  **Target:** `Solstice LLC` (ROUNDUP, MT, 303 MCCORD ROAD) — **NON-MATCH**
* **S1:** `Family Specialists` (Columbus, OH, 1035 Mediterranean Avenue)  
  **Target:** `Family Specialists` (SALEM, OR, 4733 LIBERTY RODA) — **NON-MATCH**
* **S1:** `Nion` (Ferdinand, IN, 11087 St Henry Street W)  
  **Target:** `Nion` (HENRICO COUNTY, VA, 7100 BRIGHAM ROAD) — **NON-MATCH**

> [!CAUTION]
> Relying on name similarity alone without geographic/address consistency causes severe false merges on chain stores, common corporate names, and franchises. Under $F_{0.5}$, each false merge severely penalizes precision ($2\times$ weight).

### 6.2 Same Address, Different Businesses (Multi-Tenant Buildings)
* **Address:** `14046 Mill Spring Court, Bryantown, MD`
  - Entity A: `New Life Israel`
  - Entity B: `-- VANTAGEARIA` — **NON-MATCH**
* Shared business addresses occur frequently in commercial office parks, plazas, and shopping complexes (e.g. `Samrah Plaza`, `Enkay Square`, `Merlin Oxford`). Matching on address alone without name validation will cause catastrophic false merges.

### 6.3 Generic Single-Word Names
* Common strings in S1: `Chiropractic Group`, `Internal Medicine Group`, `Meridian LLC`, `Primary Care Group`, `Happy Deli`, `Pediatric Dental Group`.
* These generic terms appear repeatedly across multiple cities.

---

## 7. Experimental Blocking Feasibility Analysis

Tested on an evaluation set of **10,000 Source 1 ground truth entities (36,533 true links)** against a realistic candidate pool of **236,533 target records** (Cartesian space: **2,365,330,000 pairs**).

### 7.1 Single-Strategy vs Multi-Pass Comparison
| Strategy | Candidate Pairs | Recall (%) | Reduction Ratio (%) | Avg Cands/S1 | Med Cands | p95 Cands | Max Cands |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Exact Normalized Name** | 8,478 | 18.40% | 99.9996% | 1.64 | 1.0 | 4.0 | 17 |
| **2. 1st Significant Word (`p1`)** | 1,188,541 | 67.40% | 99.9498% | 138.38 | 61.0 | 453.0 | 1,447 |
| **3. 2-Token Prefix (`p2`)** | 157,922 | 58.76% | 99.9933% | 16.90 | 3.0 | 33.0 | 458 |
| **4. Door + Postal Code** | 253 | 0.69% | 99.9999% | 2.58 | 2.0 | 6.0 | 8 |
| **5. Door + Addr Token (`a1`)** | 34,073 | 18.04% | 99.9986% | 11.11 | 8.0 | 32.0 | 71 |
| **6. Compact Name Prefix (`cmp6`)**| 1,038,283 | 76.97% | 99.9561% | 106.06 | 11.0 | 477.0 | 1,525 |
| **7. Combined Multi-Pass** | **2,640,905** | **85.11%** | **99.8883%** | **264.57** | **164.0** | **791.0** | **3,162** |

### 7.2 Crucial Audit Insight on Candidate Budget Truncation
When candidates are truncated arbitrarily without scoring:
* Budget $K=10$: Recall collapses to **24.96%**.
* Budget $K=25$: Recall collapses to **36.06%**.

**HOWEVER, when candidates are pre-ranked by a fast similarity score ($0.6 \times \text{Name Token Sort} + 0.4 \times \text{Addr Token Sort}$):**

| Similarity-Sorted Budget ($K$) | Candidate Recall (%) | Ground Truth Hits | Total Candidate Pairs |
| :--- | :--- | :--- | :--- |
| **$K = 5$** | **78.00%** | 2,340 / 3,000 | 5,000 |
| **$K = 10$** | **82.13%** | 2,464 / 3,000 | 10,000 |
| **$K = 15$** | **82.83%** | 2,485 / 3,000 | 15,000 |
| **$K = 20$** | **83.23%** | 2,497 / 3,000 | 20,000 |
| **$K = 25$** | **83.33%** | 2,500 / 3,000 | 25,000 |
| **$K = 30$** | **83.53%** | 2,506 / 3,000 | 30,000 |
| **$K = 50$** | **84.23%** | 2,527 / 3,000 | 47,812 |
| **$K = 100$** | **84.80%** | 2,544 / 3,000 | 72,190 |
| **Unbudgeted Ceiling** | **85.47%** | 2,564 / 3,000 | 76,164 |

> [!IMPORTANT]
> **Key Empirical Discovery:**
> 1. With similarity ranking, **$K=25$ achieves 83.33% recall**, capturing **97.5% of the total reachable candidate pool** (83.33% out of 85.47%).
> 2. Expanding $K$ from 25 to 100 costs a $4\times$ explosion in pairs for a marginal gain of only $+1.47\%$ recall.
> 3. The true upper bound of simple rule-based blocking is **$\approx 85.5\%$**, NOT $\ge 96\%$. The missing $\approx 14.5\%$ consists of cross-script translations (Devanagari/Malayalam/Bengali), unanchored trade names, and extreme typos.

---

## 8. Validation Feasibility Analysis

### 8.1 Is a 50,000 Source 1 Holdout Practical?
* **Statistical Significance:** With 50,000 entities, the standard error on macro $F_{0.5}$ is $< 0.002$. This provides reliable signal to prevent overfitting to the public leaderboard.
* **Compute Feasibility:** 50,000 entities at $K=25$ candidates yield 1.25M pairs. RapidFuzz feature extraction for 1.25M pairs takes $\approx 1.5$ seconds, and XGBoost inference takes $\approx 3$ seconds.
* **Storage Footprint:** Less than 50 MB in memory.

### 8.2 Validation Stratification Requirements
The 50,000 validation split **must strictly preserve**:
1. **Country Ratio:** 60% US, 40% India.
2. **Singleton Ratio:** Exactly 5.58% singletons (entities with 0 matches).
3. **Cardinality Distribution:** 5.4% 1-match, 17.0% 2-match, 72.0% 3+ matches.
4. **Synthetic Zero-Shot Subsplit:** Hold out a specific Indian state or US state to simulate the unseen "France effect" and verify that the model does not overfit to specific location names.

---

## 9. Final Recommendations for Production Pipeline

Based strictly on the audited data:

1. **Preprocessing Strategy:**
   - Universal Unicode decomposition (`NFKD`) to handle French accents (`é`, `ç`) and byte corruption (`\ufffd`).
   - Strip web protocols (`http://`, `www.`) and domain extensions (`.com`, `.in`, `.fr`, `.org`).
   - Standardize corporate suffixes across US (`inc, llc, corp`), India (`pvt ltd, limited, llp`), and France (`sarl, sas, sa, eurl`).
   - Strip common leading particles (`m/s`, `shree`, `sri`, `the`, `smt`, `dr`).
   - Extract numerical token sets (door numbers, 5/6 digit postal codes).

2. **Blocking Strategy:**
   - **Hard country partition:** US, India, and France processed completely independently (0 cross-country matches).
   - Multi-pass equi-joins (core name 2-token prefix, compact 6-char prefix, single long word $\ge 5$, address door + postal, address door + street token).
   - Character 3-gram sparse cosine retrieval pass for non-Latin and severe typo records.
   - **Score-sorted candidate pruning:** Sort by composite fast similarity ($0.6 \times \text{Name Token Sort} + 0.4 \times \text{Addr Token Sort}$) before truncation.

3. **Candidate Budget:**
   - **$K = 25$ candidates per entity.**
   - Retains 83.33% recall while keeping total test pairs to $\approx 43\text{M}$, well within memory and compute bounds.

4. **Feature Set:**
   - `name_jaro_winkler`, `name_token_sort_ratio`, `name_token_set_ratio`, `name_levenshtein_ratio`, `name_exact_match`, `name_core_match`.
   - `addr_token_sort_ratio`, `addr_token_set_ratio`, `addr_num_jaccard`, `addr_door_match`, `addr_postal_match`.
   - Cross features: `name_x_addr`, `composite_score`.

5. **Matching Model:**
   - GBDT (`XGBoost` with `tree_method="hist"`). Highly effective on tabular similarity features, capable of non-linear combinations (e.g. high address similarity overriding low name similarity for Indic script records).

6. **Threshold-Tuning Approach:**
   - Fine-grained grid search over $\tau \in [0.60, 0.80]$ on out-of-fold validation set directly optimizing macro $F_{0.5}$.
   - High threshold ($\tau^* \approx 0.70 - 0.75$) to protect against false merges and preserve singleton 1.0 scores.
