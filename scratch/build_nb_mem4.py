import json, os

nb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb'))
with open(nb_path, 'r', encoding='utf-8') as f:
    NB = json.load(f)

# ── FIX 4a: threshold search cell (24) ────────────────────────────────────────
assert NB['cells'][24]['cell_type'] == 'code'
src24 = NB['cells'][24]['source']

# Replace the two lines that reference df_val_pairs inside evaluate_threshold_grid
OLD_LOOP = (
    "    scores_by_s1 = defaultdict(list)\n"
    "    for s1_id, tgt_id, prob in zip(df_val_pairs['s1_id'].values,\n"
    "                                    df_val_pairs['target_id'].values, val_probs):\n"
    "        scores_by_s1[s1_id].append((tgt_id, float(prob)))\n"
)
NEW_LOOP = (
    "    scores_by_s1 = defaultdict(list)\n"
    "    for s1_id, tgt_id, prob in zip(val_s1_ids, val_tgt_ids, val_probs):\n"
    "        scores_by_s1[s1_id].append((tgt_id, float(prob)))\n"
)
assert OLD_LOOP in src24, 'Could not find scores_by_s1 loop in cell 24'
NB['cells'][24]['source'] = src24.replace(OLD_LOOP, NEW_LOOP, 1)

# ── FIX 4b: error analysis cell (26) ──────────────────────────────────────────
assert NB['cells'][26]['cell_type'] == 'code'

NB['cells'][26]['source'] = (
    "winning_model = 'LogisticRegression' if lr_best['macro_f05'] >= xgb_best['macro_f05'] else 'XGBoost'\n"
    "winning_best  = lr_best  if winning_model == 'LogisticRegression' else xgb_best\n"
    "winning_probs = lr_val_probs if winning_model == 'LogisticRegression' else xgb_val_probs\n"
    "print(f'Winning model: {winning_model} at tau*={winning_best[\"threshold\"]:.2f}')\n"
    "\n"
    "# Determine predicted match using minimal val arrays (no full DataFrame needed)\n"
    "tau_star = winning_best['threshold']\n"
    "pred_match = (winning_probs >= tau_star).astype(np.int32)\n"
    "\n"
    "# Identify FP/FN indices in the val arrays\n"
    "fp_mask = (y_val == 0) & (pred_match == 1)\n"
    "fn_mask = (y_val == 1) & (pred_match == 0)\n"
    "print(f'False Positives: {fp_mask.sum():,}  False Negatives: {fn_mask.sum():,}')\n"
    "\n"
    "# Re-fetch only the text columns needed for FP analysis from DuckDB (on demand)\n"
    "# This avoids keeping the full val DataFrame in memory throughout training\n"
    "fp_s1_ids  = val_s1_ids[fp_mask].tolist()\n"
    "fp_tgt_ids = val_tgt_ids[fp_mask].tolist()\n"
    "\n"
    "if fp_s1_ids:\n"
    "    import pandas as _pd\n"
    "    con.register('fp_s1_ids_df',  _pd.DataFrame({'s1_id':  fp_s1_ids}))\n"
    "    con.register('fp_tgt_ids_df', _pd.DataFrame({'target_id': fp_tgt_ids}))\n"
    "    df_fp_text = con.execute(\"\"\"\n"
    "        SELECT tk.s1_id, tk.target_id,\n"
    "               tk.s1_norm_name, tk.t_norm_name,\n"
    "               tk.s1_norm_addr, tk.t_norm_addr,\n"
    "               tk.s1_door, tk.t_door\n"
    "        FROM labeled_topk tk\n"
    "        WHERE tk.s1_id    IN (SELECT s1_id    FROM fp_s1_ids_df)\n"
    "          AND tk.target_id IN (SELECT target_id FROM fp_tgt_ids_df)\n"
    "          AND tk.split = 'val' AND tk.label = 0\n"
    "    \"\"\").df()\n"
    "else:\n"
    "    df_fp_text = __import__('pandas').DataFrame()\n"
    "\n"
    "fp_cats = defaultdict(int)\n"
    "for _, r in df_fp_text.iterrows():\n"
    "    ns  = fuzz.token_sort_ratio(str(r.get('s1_norm_name','') or ''), str(r.get('t_norm_name','') or ''))\n"
    "    as_ = fuzz.token_sort_ratio(str(r.get('s1_norm_addr','') or ''), str(r.get('t_norm_addr','') or ''))\n"
    "    sd  = str(r.get('s1_door','') or '')\n"
    "    td  = str(r.get('t_door','') or '')\n"
    "    sn  = str(r.get('s1_norm_name','') or '')\n"
    "    if ns >= 85 and as_ < 40:                 fp_cats['same_name_diff_address'] += 1\n"
    "    elif as_ >= 80 and ns < 40:               fp_cats['same_address_diff_business'] += 1\n"
    "    elif sd and td and sd != td and ns >= 70: fp_cats['address_number_conflict'] += 1\n"
    "    elif len(sn.replace(' ','')) <= 5:        fp_cats['abbreviation_collision'] += 1\n"
    "    else:                                     fp_cats['generic_name_collision'] += 1\n"
    "\n"
    "fp_total = max(len(df_fp_text), 1)\n"
    "fp_summary = pd.DataFrame([{'Category':k,'Count':v,'Pct':f'{v/fp_total*100:.1f}%'}\n"
    "                            for k,v in sorted(fp_cats.items(), key=lambda x:-x[1])])\n"
    "print('\\nFalse Positive Categories:')\n"
    "display(fp_summary)\n"
    "\n"
    "# Clean up FP text frame — not needed after this cell\n"
    "del df_fp_text\n"
    "gc.collect()\n"
)

# ── FIX 4c: artifacts cell (28) — replace df_val_eval reference ───────────────
assert NB['cells'][28]['cell_type'] == 'code'
src28 = NB['cells'][28]['source']

OLD_VAL_EVAL = (
    "df_val_eval[['s1_id','target_id','label','pred_prob','pred_match']].head(10000).to_csv(\n"
    "    os.path.join(results_dir,'validation_predictions.csv'), index=False)\n"
)
NEW_VAL_EVAL = (
    "# Build minimal val predictions frame from arrays (no full DataFrame needed)\n"
    "import pandas as _pd2\n"
    "_val_preds = _pd2.DataFrame({\n"
    "    's1_id':      val_s1_ids[:10000],\n"
    "    'target_id':  val_tgt_ids[:10000],\n"
    "    'label':      y_val[:10000],\n"
    "    'pred_prob':  winning_probs[:10000],\n"
    "    'pred_match': pred_match[:10000],\n"
    "})\n"
    "_val_preds.to_csv(os.path.join(results_dir,'validation_predictions.csv'), index=False)\n"
    "del _val_preds\n"
)
assert OLD_VAL_EVAL in src28, 'Could not find df_val_eval reference in cell 28'
NB['cells'][28]['source'] = src28.replace(OLD_VAL_EVAL, NEW_VAL_EVAL, 1)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=2, ensure_ascii=False)
print('Fix 4 applied: threshold search, error analysis, artifacts updated.')
