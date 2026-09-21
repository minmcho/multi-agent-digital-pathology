"""
DeLong test for comparing AUCs from two classifiers on the same test set.
Implements the covariance estimator from DeLong et al. (1988).

Reference: DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under
two or more correlated receiver operating characteristic curves: a nonparametric
approach. Biometrics. 1988;44(3):837-845.
"""
import numpy as np
from scipy import stats

def _compute_midrank(x):
    """Compute midranks of elements in x."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # midrank (1-indexed)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _structural_components(gt, pred):
    """
    Compute DeLong structural components V_10 and V_01 for a single classifier.

    V_10(i) = per-positive contribution to AUC = (midrank_pos_i - (n_pos+1)/2) / n_neg
    V_01(j) = per-negative contribution to AUC = (midrank_neg_j - (n_neg+1)/2) / n_pos

    AUC = mean(V_10) = mean(V_01) + (n_neg - n_pos)/(2*n_neg)  [approximately]

    Returns:
        v10: (n_pos,) structural components for positive samples
        v01: (n_neg,) structural components for negative samples
        auc: AUC value
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    n_pos = np.sum(gt == 1)
    n_neg = np.sum(gt == 0)

    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([]), 0.5

    pos_mask = gt == 1
    neg_mask = gt == 0

    # Midranks on the combined set
    m = _compute_midrank(pred)
    tx = m[pos_mask]  # midranks for positive samples
    ty = m[neg_mask]  # midranks for negative samples

    # AUC
    auc = (tx.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # Structural components (divided by opposite class size)
    v10 = (tx - (n_pos + 1) / 2.0) / n_neg  # for positive samples
    v01 = (ty - (n_neg + 1) / 2.0) / n_pos  # for negative samples

    return v10, v01, auc

def delong_roc_variance(ground_truth, predictions):
    """
    Compute AUC and variance for a single classifier using DeLong's method.

    Args:
        ground_truth: (N,) array of binary labels (0 or 1)
        predictions: (N,) array of predicted probabilities

    Returns:
        auc: Area under ROC curve
        var: Variance of AUC estimate
    """
    gt = np.asarray(ground_truth)
    pred = np.asarray(predictions)
    n_pos = np.sum(gt == 1)
    n_neg = np.sum(gt == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    v10, v01, auc = _structural_components(gt, pred)

    # var(AUC) = var(V_10)/n_pos + var(V_01)/n_neg
    var = np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg

    return auc, var

def delong_roc_test(ground_truth, predictions_1, predictions_2):
    """
    DeLong test comparing AUCs of two classifiers on the same test set.

    Args:
        ground_truth: (N,) array of binary labels
        predictions_1: (N,) array of predictions from classifier 1
        predictions_2: (N,) array of predictions from classifier 2

    Returns:
        auc_1: AUC of classifier 1
        auc_2: AUC of classifier 2
        z_score: Z-score of the difference
        p_value: Two-sided p-value
    """
    gt = np.asarray(ground_truth)
    p1 = np.asarray(predictions_1)
    p2 = np.asarray(predictions_2)

    n_pos = np.sum(gt == 1)
    n_neg = np.sum(gt == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5, 0.0, 1.0

    # Structural components for each classifier
    v10_1, v01_1, auc1 = _structural_components(gt, p1)
    v10_2, v01_2, auc2 = _structural_components(gt, p2)

    # Variances: var(AUC) = var(V_10)/n_pos + var(V_01)/n_neg
    var_1 = np.var(v10_1, ddof=1) / n_pos + np.var(v01_1, ddof=1) / n_neg
    var_2 = np.var(v10_2, ddof=1) / n_pos + np.var(v01_2, ddof=1) / n_neg

    # Covariance: cov(AUC_1, AUC_2) = cov(V_10_1, V_10_2)/n_pos + cov(V_01_1, V_01_2)/n_neg
    cov = np.cov(v10_1, v10_2, ddof=1)[0, 1] / n_pos + np.cov(v01_1, v01_2, ddof=1)[0, 1] / n_neg

    # Z-score
    diff = auc1 - auc2
    se = np.sqrt(var_1 + var_2 - 2 * cov)

    if se == 0 or np.isnan(se):
        z = 0.0
        p = 1.0
    else:
        z = diff / se
        p = 2 * stats.norm.sf(np.abs(z))

    return auc1, auc2, z, p

def delong_paired_comparison(ground_truth, model_predictions, model_names=None):
    """
    Perform all pairwise DeLong tests between multiple classifiers.

    Args:
        ground_truth: (N,) array of binary labels
        model_predictions: dict {name: (N,) predictions} or list of arrays
        model_names: list of names (if model_predictions is a list)

    Returns:
        results: list of dicts with {model_1, model_2, auc_1, auc_2, z, p_value}
    """
    if isinstance(model_predictions, dict):
        names = list(model_predictions.keys())
        preds = [model_predictions[n] for n in names]
    else:
        names = model_names or [f"Model_{i}" for i in range(len(model_predictions))]
        preds = model_predictions

    n_models = len(preds)
    results = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            auc1, auc2, z, p = delong_roc_test(ground_truth, preds[i], preds[j])
            results.append({
                'model_1': names[i],
                'model_2': names[j],
                'auc_1': float(auc1),
                'auc_2': float(auc2),
                'z_score': float(z),
                'p_value': float(p),
            })

    return results

def bonferroni_correct(results, alpha=0.05):
    """Apply Bonferroni correction to p-values."""
    n_tests = len(results)
    corrected_alpha = alpha / n_tests
    for r in results:
        r['significant'] = r['p_value'] < corrected_alpha
        r['bonferroni_alpha'] = corrected_alpha
    return results
