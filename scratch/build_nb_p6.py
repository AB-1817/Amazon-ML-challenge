import json, os

nb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'train_matching_model_colab.ipynb'))
with open(nb_path, 'r', encoding='utf-8') as f:
    NB = json.load(f)

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

# ── SEC 15: ERROR ANALYSIS ─────────────────────────────────────────────────────
NB["cells"].append(md("### Section 15: Error Analysis\n"))
NB["cells"].append(code(
    "winning_model = 'LogisticRegression' if lr_best['macro_f05'] >= xgb_best['macro_f05'] else 'XGBoost'\n"
    "winning_best  = lr_best  if winning_model == 'LogisticRegression' else xgb_best\n"
    "winning_probs = lr_val_probs if winning_model == 'LogisticRegression' else xgb_val_probs\n"
    "print(f'Winning model: {winning_model} at tau*={winning_best[\"threshold\"]:.2f}')\n"
    "\n"
    "df_val_eval = df_val_pairs.copy()\n"
    "df_val_eval['pred_prob']  = winning_probs\n"
    "df_val_eval['pred_match'] = (winning_probs >= winning_best['threshold']).astype(int)\n"
    "\n"
    "fp_df = df_val_eval[(df_val_eval['label']==0) & (df_val_eval['pred_match']==1)]\n"
    "fn_df = df_val_eval[(df_val_eval['label']==1) & (df_val_eval['pred_match']==0)]\n"
    "print(f'False Positives: {len(fp_df):,}  False Negatives: {len(fn_df):,}')\n"
    "\n"
    "fp_cats = defaultdict(int)\n"
    "for _, r in fp_df.iterrows():\n"
    "    ns = fuzz.token_sort_ratio(str(r.get('s1_norm_name','') or ''), str(r.get('t_norm_name','') or ''))\n"
    "    as_ = fuzz.token_sort_ratio(str(r.get('s1_norm_addr','') or ''), str(r.get('t_norm_addr','') or ''))\n"
    "    sd, td = str(r.get('s1_door','') or ''), str(r.get('t_door','') or '')\n"
    "    sn = str(r.get('s1_norm_name','') or '')\n"
    "    if ns >= 85 and as_ < 40:                          fp_cats['same_name_diff_address'] += 1\n"
    "    elif as_ >= 80 and ns < 40:                        fp_cats['same_address_diff_business'] += 1\n"
    "    elif sd and td and sd != td and ns >= 70:          fp_cats['address_number_conflict'] += 1\n"
    "    elif len(sn.replace(' ','')) <= 5:                 fp_cats['abbreviation_collision'] += 1\n"
    "    else:                                              fp_cats['generic_name_collision'] += 1\n"
    "\n"
    "fp_summary = pd.DataFrame([{'Category':k,'Count':v,'Pct':f'{v/max(len(fp_df),1)*100:.1f}%'}\n"
    "                            for k,v in sorted(fp_cats.items(), key=lambda x:-x[1])])\n"
    "print('\\nFalse Positive Categories:')\n"
    "display(fp_summary)\n"
))

# ── SEC 16-17: ARTIFACTS ───────────────────────────────────────────────────────
NB["cells"].append(md("### Section 16 & 17: Save Model Artifacts and Results\n"))
NB["cells"].append(code(
    "import joblib\n"
    "\n"
    "artifacts_dir = 'artifacts'\n"
    "results_dir   = 'results'\n"
    "os.makedirs(os.path.join(artifacts_dir,'logistic_regression'), exist_ok=True)\n"
    "os.makedirs(os.path.join(artifacts_dir,'xgboost'), exist_ok=True)\n"
    "os.makedirs(results_dir, exist_ok=True)\n"
    "\n"
    "joblib.dump(lr_model, os.path.join(artifacts_dir,'logistic_regression','model.joblib'))\n"
    "joblib.dump(scaler,   os.path.join(artifacts_dir,'logistic_regression','scaler.joblib'))\n"
    "xgb_model.save_model(os.path.join(artifacts_dir,'xgboost','xgboost_model.json'))\n"
    "\n"
    "with open(os.path.join(artifacts_dir,'feature_config.json'),'w') as f:\n"
    "    json.dump({'feature_names':FEATURE_NAMES,'count':len(FEATURE_NAMES)}, f, indent=2)\n"
    "\n"
    "with open(os.path.join(artifacts_dir,'selected_model.json'),'w') as f:\n"
    "    json.dump({'selected_model':winning_model,\n"
    "               'optimal_threshold':winning_best['threshold'],\n"
    "               'validation_macro_f05':winning_best['macro_f05'],\n"
    "               'validation_precision':winning_best['precision'],\n"
    "               'validation_recall':winning_best['recall']}, f, indent=2)\n"
    "\n"
    "comp_summary.to_csv(os.path.join(results_dir,'model_comparison.csv'), index=False)\n"
    "pd.concat([lr_df,xgb_df]).to_csv(os.path.join(results_dir,'threshold_results.csv'), index=False)\n"
    "df_val_eval[['s1_id','target_id','label','pred_prob','pred_match']].head(10000).to_csv(\n"
    "    os.path.join(results_dir,'validation_predictions.csv'), index=False)\n"
    "\n"
    "print(f'Artifacts saved to {artifacts_dir}/  Results saved to {results_dir}/')\n"
))

# ── SEC 18: SUMMARY ────────────────────────────────────────────────────────────
NB["cells"].append(md(
    "### Section 18: Final Experiment Summary\n\n"
    "**This was a DEVELOPMENT EXPERIMENT** (150k S1 entities, random distractors).\n\n"
    "1. **Blocking**: Finalized 8-pass architecture, K=25, zero OOM overhead.\n"
    "2. **Hard Negatives**: 6-category deliberate sampling at HARD_NEG_RATIO=3.0.\n"
    "3. **Target Pool**: Explicitly reported true-target %, S2/S3 distractor counts.\n"
    "4. **Leakage**: Explicit checks — no S1 in both splits, no cross-split candidate pairs.\n"
    "5. **F0.5**: Unit-tested implementation, all val S1s evaluated including singletons.\n"
    "6. **Feature Validation**: NaN/inf, min/max/mean, constant, redundant features checked.\n"
    "7. **Models**: LR vs XGBoost on identical splits. Threshold selected on val only.\n"
    "8. **Reproducibility**: SEED=42, TRAIN_S1_LIMIT=150000, VALIDATION_FRACTION=0.20, TOP_K=25.\n"
))

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=2, ensure_ascii=False)
print('Sections 15-18 appended.')
