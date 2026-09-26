"""Append Section 25: fair comparison table."""
import json

NB = r'c:\Users\akash\OneDrive\Desktop\Amazon ML Challenge\notebooks\train_matching_model_colab.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

nb['cells'].append(md(
"### Section 25: Fair Comparison Table\n\n"
"All models evaluated on the same 30,001 validation S1 entities with the same Macro F0.5 implementation.\n"
))

src_25 = (
"import pandas as _pd_v2\n"
"from sklearn.metrics import roc_auc_score\n"
"\n"
"# Baseline results (from Section 13/14)\n"
"baseline_row = {\n"
"    'Experiment': 'Baseline_8pass_K25',\n"
"    'Blocking': '8-pass K=25',\n"
"    'Candidate_Recall_at_K': blocking_results[0]['recall_at_25'],\n"
"    'Model': 'XGBoost',\n"
"    'Macro_F05': xgb_best['macro_f05'],\n"
"    'Precision': xgb_best['precision'],\n"
"    'Recall': xgb_best['recall'],\n"
"    'Threshold': xgb_best['threshold'],\n"
"    'Train_Time_s': xgb_time,\n"
"    'ROC_AUC': round(roc_auc_score(y_val, xgb_val_probs), 4),\n"
"}\n"
"\n"
"# Best blocking recall from comparison\n"
"best_block_recall = best_blocking['recall_at_50']\n"
"\n"
"# v2 rows\n"
"v2_rows = []\n"
"for mname, _ in models_v2:\n"
"    br = best_rows_v2[mname]\n"
"    v2_rows.append({\n"
"        'Experiment': f'v2_{BEST_STRATEGY}',\n"
"        'Blocking': f'{BEST_STRATEGY} K={K_BEST}',\n"
"        'Candidate_Recall_at_K': best_block_recall,\n"
"        'Model': mname,\n"
"        'Macro_F05': br['macro_f05'],\n"
"        'Precision': br['precision'],\n"
"        'Recall': br['recall'],\n"
"        'Threshold': br['threshold'],\n"
"        'Train_Time_s': (\n"
"            xgb_time_v2 if 'XGBoost' in mname\n"
"            else cb_time if 'CatBoost' in mname\n"
"            else xgb_time_v2 + cb_time\n"
"        ),\n"
"        'ROC_AUC': round(roc_auc_score(y_val_v2, (\n"
"            xgb_probs_v2 if 'XGBoost' in mname\n"
"            else cb_probs if 'CatBoost' in mname\n"
"            else ens_50\n"
"        )), 4),\n"
"    })\n"
"\n"
"comparison_df = _pd_v2.DataFrame([baseline_row] + v2_rows)\n"
"print('=' * 100)\n"
"print('FINAL EXPERIMENT COMPARISON TABLE')\n"
"print('=' * 100)\n"
"display(comparison_df[['Experiment','Blocking','Candidate_Recall_at_K','Model',\n"
"                        'Macro_F05','Precision','Recall','Threshold','Train_Time_s','ROC_AUC']])\n"
"\n"
"comparison_df.to_csv('/content/experiment_v2/model_comparison_v2.csv', index=False)\n"
"print('Saved: /content/experiment_v2/model_comparison_v2.csv')\n"
"\n"
"# Identify best overall model\n"
"best_overall = comparison_df.loc[comparison_df['Macro_F05'].idxmax()]\n"
"print(f'\\nBest overall: {best_overall[\"Model\"]} ({best_overall[\"Experiment\"]})')\n"
"print(f'  Macro F0.5 = {best_overall[\"Macro_F05\"]}')\n"
"print(f'  Precision  = {best_overall[\"Precision\"]}')\n"
"print(f'  Recall     = {best_overall[\"Recall\"]}')\n"
"print(f'  Threshold  = {best_overall[\"Threshold\"]}')\n"
"\n"
"# Determine best v2 model name and probs for downstream use\n"
"best_v2_name = max(best_rows_v2, key=lambda k: best_rows_v2[k]['macro_f05'])\n"
"best_v2_probs = {\n"
"    'XGBoost_v2':          xgb_probs_v2,\n"
"    'CatBoost':            cb_probs,\n"
"    'Ensemble_XGB25_CB75': ens_25,\n"
"    'Ensemble_XGB50_CB50': ens_50,\n"
"    'Ensemble_XGB75_CB25': ens_75,\n"
"}[best_v2_name]\n"
"best_v2_tau = best_rows_v2[best_v2_name]['threshold']\n"
"print(f'\\nBest v2 model: {best_v2_name}  tau*={best_v2_tau}')\n"
)

nb['cells'].append(code(src_25))

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print(f'Cells: {len(nb["cells"])}  Part8 written.')
