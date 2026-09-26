import numpy as np
from typing import Dict, Set, List, Tuple

def compute_single_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity.
    Special handling for singletons (true_matches empty):
      - correctly predicting empty yields 1.0
      - false merge on singleton yields 0.0
    """
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
    if denom == 0:
        return 0.0
    
    return (1.25 * prec * rec) / denom


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> float:
    """
    Compute macro-averaged F_0.5 across all Source 1 entities in ground truth.
    Every S1 entity must be evaluated.
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(compute_single_entity_f05(true_set, pred_set))
    return float(np.mean(scores))


def find_optimal_threshold(
    ground_truth: Dict[str, Set[str]],
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    thresholds: List[float] = None
) -> Tuple[float, float]:
    """
    Find optimal score threshold tau that maximizes validation macro F_0.5.
    candidate_scores: {s1_id: [(target_id, prob), ...]}
    """
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.40, 0.90, 0.05)]
    
    best_thresh = 0.65
    best_f05 = -1.0
    
    for tau in thresholds:
        preds = {}
        for s1_id in ground_truth.keys():
            cands = candidate_scores.get(s1_id, [])
            matched = {t_id for t_id, prob in cands if prob >= tau}
            preds[s1_id] = matched
            
        score = evaluate_macro_f05(ground_truth, preds)
        print(f"Threshold tau = {tau:.2f} -> Macro F_0.5: {score:.4f}")
        if score > best_f05:
            best_f05 = score
            best_thresh = tau
            
    print(f"Optimal Threshold: tau* = {best_thresh:.2f} with Macro F_0.5 = {best_f05:.4f}")
    return best_thresh, best_f05
