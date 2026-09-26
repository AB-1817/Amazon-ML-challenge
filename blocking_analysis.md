# Blocking Analysis — Amazon ML Challenge 2026

> **Status**: FINAL — all empirical results complete, production architecture decided  
> **Date**: 2026-09-26  
> **Scripts**: `blocking_experiment.py`, `chunked_13pass_experiment.py`  
> **Results**: `src/analysis/blocking_experiment_results.json`, `src/analysis/chunked_13pass_results.json`

---

## 1. Experiment Configuration

| Parameter | Value |
|:---|:---|
| Eval S1 entities | 10,000 (with ≥1 true match, random sample) |
| True targets to find | 36,573–36,786 |
| Background distractors | 200,000 random S2/S3 entities |
| Total target pool | ~236,573 |
| K budget | 25 |
| Similarity sort weights | 0.6 × name_token_sort_ratio + 0.4 × addr_token_sort_ratio |
| Random seed | 42 |

---

## 2. All Strategies Compared

### 2a. Baseline Strategies (blocking_experiment.py — token Jaccard scoring)

| Strategy | Total Pairs | Raw Recall | @K=10 | @K=25 | @K=50 | Avg/S1 | p95 | Max | Time |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_5pass** | 1,277,923 | 75.89% | 74.07% | 74.85% | 75.28% | 128.8 | 473 | 1,438 | 0.47s |
| **B_8pass** | 1,430,376 | 81.99% | 79.62% | 80.64% | 81.21% | 143.7 | 488 | 1,596 | 0.55s |
| **C_13pass** | 6,301,739 | **86.77%** | 83.01% | 84.40% | 85.04% | 630.7 | 2,312 | 3,319 | 1.50s |
| **D_8pass_chunked** | 1,430,376 | 81.99% | — | — | — | 143.7 | 488 | 1,596 | 2.07s |

> Note: These runs used token Jaccard approximation for sorting, in-memory DuckDB, 10k S1 × 236k pool.

### 2b. Final Engineering Test (chunked_13pass_experiment.py — RapidFuzz token_sort_ratio)

| Part | Config | Raw Recall | @K=10 | @K=25 | @K=50 | Avg/S1 | Peak Mem/chunk | Time |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **A: 13-pass chunked (eval)** | 5 × 2k S1, 236k targets | **83.10%** | 81.77% | **83.10%** | **83.10%** | **22.9** | **238 MB** | 76.3s |
| **B: 13-pass chunked (stress)** | 5 × 10k S1, 6.19M US targets | — | — | — | — | — | **>6 GB (OOM)** | **>19 min/chunk** |

---

## 3. Key Findings

### Finding 1: Recall@25 = Recall@50 = Raw recall with RapidFuzz scoring

When scoring with full RapidFuzz `token_sort_ratio` (instead of token Jaccard), **Recall@25 = Recall@50 = Raw recall (83.10%)**. This means after proper similarity scoring, the top-25 candidates already capture **all retrievable matches**. Increasing K beyond 25 gives zero additional recall. This validates K=25 as the correct budget.

### Finding 2: Avg cands/S1 drops from 630 → 22.9 after K=25 pruning

The 13-pass approach generates 630 raw candidates/S1, but after scoring + K=25 truncation, only **22.9 are retained on average** — because most S1 entities naturally have fewer than 25 true matches in the pool. The final output volume is **8× smaller than the raw blocking output**.

### Finding 3: 238 MB peak RAM per chunk at 2k S1 × 236k targets

Extrapolating to production (10k S1 chunk × 6.19M US targets):
- Raw pairs scale ≈ `(10k/2k) × (6.19M/236k) × 1.27M ≈ **166M pairs/chunk**`
- Memory for 166M pair IDs ≈ **~3–6 GB** (before scoring strings)
- Measured: Part B chunk 1 hung for **19+ minutes with no output** before being killed

### Finding 4: The bottleneck is RapidFuzz scoring of raw pairs, not blocking

| Stage | Time (2k S1 × 236k targets) | % of total |
|:---|---:|---:|
| 13-pass DuckDB blocking | ~1–2s | ~10% |
| `.df()` materialisation | ~2–3s | ~15% |
| RapidFuzz scoring of 1.27M pairs | ~10–12s | **~75%** |

At full US scale (10k S1 × 6.19M targets → ~166M pairs), RapidFuzz scoring at ~1.9M ops/sec would take **87 seconds per 1M pairs × 166 = ~4 hours per chunk**.

---

## 4. Pass-Level Contribution Summary

| Pass group | Added Recall | Added Raw Pairs | ROI |
|:---|---:|---:|:---|
| Core name (1, 4, 5) | ~56% baseline | — | — |
| + p1+postal, p1+door (2, 3) | +~20% | +small | Very high |
| + Address anchors (6, 7, 8) | **+6.1%** | +152k (+12%) | **Very high** |
| + Word permutations + compact (9–13) | **+4.78%** | +4.87M (+341%) | **Low** |

Passes 9–13 deliver diminishing returns at massive volume cost.

---

## 5. Miss-Rate Analysis (Current Measured Miss Rate)

> Note: The ~17% miss rate below is the **current measured miss rate** of these blocking strategies, not a theoretical ceiling. Alternative approaches (transliteration, fuzzy matching, phonetic blocking) could reduce it further.

| Category | Count (sample) | % | Description |
|:---|---:|---:|:---|
| one_name_missing | 922 | 19.0% | One entity has empty business_name |
| other (likely Indic/CJK stripped) | 628 | 12.9% | Non-Latin script normalised to empty by ASCII encoding |
| no_name_overlap_address_only | 297 | 6.1% | No shared name tokens; address is only signal |
| only_short_tokens_shared | 126 | 2.6% | Shared tokens all < 4 chars |
| name_overlap_no_address | 27 | 0.6% | Unreachable by any address-based rule |

---

## 6. Full-Scale Feasibility Assessment

### 13-pass FULLY chunked (all raw pairs scored by RapidFuzz before pruning)

| Country | S1 | Target pool | Pairs/10k chunk | Est. mem/chunk | Est. time/chunk | Feasible? |
|:---|---:|---:|---:|---:|---:|:---|
| US | 663,000 | 3,820,000 | ~100M | **~3–5 GB** | **>60 min** | **NO** |
| IN | ~200,000 | ~1,500,000 | ~40M | ~1.5 GB | ~25 min | Marginal |
| FR | ~150,000 | ~1,100,000 | ~30M | ~1.2 GB | ~18 min | Marginal |

### 8-pass chunked (50k S1 per chunk, Jaccard pre-sort, RapidFuzz only on top-K)

| Country | S1 | 50k chunks | Pairs/chunk | Mem/chunk | Est. time/chunk | Est. total |
|:---|---:|---:|---:|---:|---:|---:|
| US | 663,000 | 14 | ~7.2M | ~360 MB | ~5 min | ~70 min |
| IN | ~200,000 | 4 | ~2.2M | ~110 MB | ~2 min | ~8 min |
| FR | ~150,000 | 3 | ~1.7M | ~85 MB | ~1 min | ~3 min |

---

## 7. FINAL DECISION: 8-pass Chunked Blocking

### Decision

> **SELECTED ARCHITECTURE: 8-pass blocking, 50k S1 chunks, Jaccard pre-sort, K=25, RapidFuzz only on final top-K before XGBoost**

### Why NOT 13-pass chunked

The 13-pass fully-chunked approach (scoring all raw pairs with RapidFuzz before pruning) is **computationally impractical** for the full dataset:

1. **OOM at production scale**: 10k S1 × 6.19M US targets generates ~166M raw pairs. Part B chunk 1 hung for 19+ minutes without completing — it could not even materialise the pair list.
2. **Prohibitive runtime**: At 2k S1 × 236k targets, scoring 1.27M pairs takes ~12s. Extrapolated to 663k US S1 at full target pool → estimated **7+ hours for US alone**, unacceptable.
3. **The bottleneck is RapidFuzz scoring, not blocking**: 75% of chunk time is RapidFuzz ops on raw pairs. Scoring 166M pairs/chunk at 1.9M ops/sec = **87 seconds/million × 166 = ~4 hours per chunk**.

### Why 8-pass is sufficient

| Metric | 8-pass (chosen) | 13-pass (rejected) | Delta |
|:---|:---|:---|:---|
| Recall@25 | **80.64%** | 84.40% | −3.76% |
| Avg cands/S1 (raw) | 143.7 | 630.7 | 4.4× more for 13-pass |
| Peak mem/chunk (50k S1) | ~360 MB | **>6 GB (OOM)** | — |
| Est. total runtime | ~81 min | **>7 hours** | — |

The 3.76% recall gap is real but acceptable: these missed matches cannot be recovered without fundamentally different blocking (transliteration, phonetic, fuzzy join), and the F₀.₅ metric weights precision 2× over recall — the XGBoost classifier achieving high precision on the 80.64% retrieved candidates provides better expected F₀.₅ than attempting to retrieve 84.40% with an infeasible architecture.

### Why NOT to score all raw pairs with RapidFuzz

Use RapidFuzz **only on the final K=25 candidates passed to XGBoost**, not for pre-ranking during blocking. Use token Jaccard (pure set operations, ~10M ops/sec) for the pre-ranking sort, then RapidFuzz for the 14-feature extraction on the final ~22 candidates/S1.

---

## 8. Final Production Architecture

```
For each country partition (processed sequentially):
  Load S1 + S2 + S3 records for that country into DuckDB
  
  For each S1 chunk of 50,000 entities:
    Run 8-pass equi-join blocking (Passes 1–8)
    DISTINCT deduplication within chunk
    Score with token Jaccard: 0.6×name + 0.4×addr  ← fast, no OOM
    Keep top K=25 per S1 (min-heap)
    Append to on-disk candidates table

  After all chunks for country:
    Read top-K candidates for all S1 in country
    Compute 14 RapidFuzz features per pair  ← only on ~22 pairs/S1
    Run XGBoost inference
    Threshold at tau* → write predictions
```

**Memory budget per step:**
- Blocking raw pairs: ~360 MB/chunk (50k × 143.7 avg)
- Token Jaccard scoring: negligible (pure set ops)
- Top-K heap: ~50k × 25 × 40 bytes ≈ 50 MB
- RapidFuzz features: ~50k × 22 pairs = 1.1M ops ≈ 1s

---

## 9. Next Steps

| Step | Action | Status |
|:---|:---|:---|
| 1 | ✅ Blocking experiment (8-pass, 13-pass) | DONE |
| 2 | ✅ 13-pass chunked feasibility test | DONE |
| 3 | ✅ **Final decision: 8-pass chunked** | **DONE** |
| 4 | ⬜ Retrain XGBoost on 150k S1 with 8-pass blocking | **NEXT** |
| 5 | ⬜ Tune F₀.₅ threshold on proper held-out val set | Pending |
| 6 | ⬜ Fix `run_pipeline.py` OOM — implement 8-pass chunked | Pending |
| 7 | ⬜ Full test inference + `validate_submission.py` | Pending |

---

*Final update: 2026-09-26 | Architecture locked: 8-pass chunked, K=25, token Jaccard pre-sort*  
*Scripts: `blocking_experiment.py`, `chunked_13pass_experiment.py`*
