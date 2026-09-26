"""Append Section 24: ensemble + threshold search for all v2 models."""
import json

NB = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

nb['cells'].append(md(
"### Section 24: Ensemble + Threshold Search (All v2 Models)\n\n"
"Ensemble: `alpha * xgb_prob + (1-alpha) * cb_prob` for alpha in {0.25, 0.50, 0.75}.\n"
"Threshold grid: 0.30–0.95 step 0.05. Metric: Macro F0.5 (same implementation as baseline).\n"
"All models evaluated on the same val S1 entities.\n"
))

src_24 = (
"from collections import defaultdict\n"
"import numpy as _np2\n"
"import pandas as _pd_v2\n"
"\n"
"# Build val GT map for v2 (same val_gt dict already exists from Section 13)\n"
"# val_gt is already built — reuse it\n"
"\n"
"def evaluate_threshold_grid_v2(val_probs, model_name, s1_ids, tgt_ids):\n"
"    scores_by_s1 = defaultdict(list)\n"
"    for s1_id, tgt_id, prob in zip(s1_ids, tgt_ids, val_probs):\n"
"        scores_by_s1[s1_id].append((tgt_id, float(prob)))\n"
"    thresholds = [round(t,2) for t in _np2.arange(0.30, 0.96, 0.05)]\n"
"    results, best_row, best_f05 = [], None, -1.0\n"
"    for tau in thresholds:\n"
"        f05_list = []\n"
"        total_tp = total_fp = total_fn = total_preds = 0\n"
"        for sid, true_set in val_gt.items():\n"
"            cands    = scores_by_s1.get(sid, [])\n"
"            pred_set = {tid for tid, prob in cands if prob >= tau}\n"
"            f05_list.append(compute_entity_f05(true_set, pred_set))\n"
"            total_tp    += len(true_set & pred_set)\n"
"            total_fp    += len(pred_set - true_set)\n"
"            total_fn    += len(true_set - pred_set)\n"
"            total_preds += len(pred_set)\n"
"        macro_f05 = float(_np2.mean(f05_list))\n"
"        prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.0\n"
"        rec  = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0.0\n"
"        row  = {'model':model_name,'threshold':tau,\n"
"                'macro_f05':round(macro_f05,4),\n"
"                'precision':round(prec,4),'recall':round(rec,4),\n"
"                'false_positives':total_fp,'false_negatives':total_fn,\n"
"                'avg_matches_per_s1':round(total_preds/len(val_gt),3),\n"
"                'n_val_s1_evaluated':len(val_gt)}\n"
"        results.append(row)\n"
"        if macro_f05 > best_f05:\n"
"            best_f05, best_row = macro_f05, row\n"
"    return _pd_v2.DataFrame(results), best_row\n"
"\n"
"# ── Ensemble probabilities ─────────────────────────────────────────────────\n"
"ens_25 = 0.25 * xgb_probs_v2 + 0.75 * cb_probs\n"
"ens_50 = 0.50 * xgb_probs_v2 + 0.50 * cb_probs\n"
"ens_75 = 0.75 * xgb_probs_v2 + 0.25 * cb_probs\n"
"\n"
"# ── Threshold search for all models ───────────────────────────────────────\n"
"print('Running threshold search for all v2 models...')\n"
"models_v2 = [\n"
"    ('XGBoost_v2',          xgb_probs_v2),\n"
"    ('CatBoost',            cb_probs),\n"
"    ('Ensemble_XGB25_CB75', ens_25),\n"
"    ('Ensemble_XGB50_CB50', ens_50),\n"
"    ('Ensemble_XGB75_CB25', ens_75),\n"
"]\n"
"\n"
"all_thresh_dfs_v2 = []\n"
"best_rows_v2 = {}\n"
"for mname, mprobs in models_v2:\n"
"    df_t, best_r = evaluate_threshold_grid_v2(mprobs, mname, val_s1_ids_v2, val_tgt_ids_v2)\n"
"    all_thresh_dfs_v2.append(df_t)\n"
"    best_rows_v2[mname] = best_r\n"
"    print(f'  {mname:<30}  F0.5={best_r[\"macro_f05\"]:.4f}  tau={best_r[\"threshold\"]:.2f}  P={best_r[\"precision\"]:.4f}  R={best_r[\"recall\"]:.4f}')\n"
"\n"
"thresh_df_v2 = _pd_v2.concat(all_thresh_dfs_v2, ignore_index=True)\n"
"thresh_df_v2.to_csv('/content/experiment_v2/threshold_results_v2.csv', index=False)\n"
"print('Saved: /content/experiment_v2/threshold_results_v2.csv')\n"
)

nb['cells'].append(code(src_24))

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f'Cells: {len(nb["cells"])}  Part7 written.')
