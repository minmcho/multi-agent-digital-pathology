"""
Clinical metrics: sensitivity at fixed specificity, temperature scaling,
pooled DeLong tests, decision curve analysis.
"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats as scipy_stats
import torch
import torch.nn as nn
import torch.optim as optim


def sensitivity_at_specificity(probs, labels, target_specificity):
    """Compute sensitivity at a fixed specificity level."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    specificity = 1 - fpr
    # Find the threshold where specificity >= target
    valid = specificity >= target_specificity
    if valid.sum() == 0:
        return 0.0, None
    # Among valid thresholds, pick the one with highest sensitivity
    best_idx = np.argmax(tpr[valid])
    valid_indices = np.where(valid)[0]
    best_threshold_idx = valid_indices[best_idx]
    return float(tpr[best_threshold_idx]), float(thresholds[best_threshold_idx])


def compute_all_clinical_metrics(probs, labels):
    """Compute all clinical metrics from probability predictions."""
    auc = roc_auc_score(labels, probs)
    sens95, thresh95 = sensitivity_at_specificity(probs, labels, 0.95)
    sens99, thresh99 = sensitivity_at_specificity(probs, labels, 0.99)

    # Also compute specificity at fixed sensitivity (90%, 95%)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    specificity = 1 - fpr

    # Specificity at 90% sensitivity
    valid_90 = tpr >= 0.90
    spec90 = float(np.max(specificity[valid_90])) if valid_90.sum() > 0 else 0.0

    # Specificity at 95% sensitivity
    valid_95 = tpr >= 0.95
    spec95 = float(np.max(specificity[valid_95])) if valid_95.sum() > 0 else 0.0

    return {
        'auc': float(auc),
        'sensitivity_at_95_spec': sens95,
        'threshold_at_95_spec': thresh95,
        'sensitivity_at_99_spec': sens99,
        'threshold_at_99_spec': thresh99,
        'specificity_at_90_sens': spec90,
        'specificity_at_95_sens': spec95,
    }


# ============================================================
# Temperature Scaling
# ============================================================

class TemperatureScaler(nn.Module):
    """Learn a single temperature parameter for calibration."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def compute_logits_from_probs(probs):
    """Convert probabilities to logits."""
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return np.log(probs / (1 - probs))


def fit_temperature(val_logits, val_labels, max_iter=1000, lr=0.01):
    """Fit temperature parameter on validation set."""
    scaler = TemperatureScaler()
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    val_logits_t = torch.from_numpy(val_logits).float()
    val_labels_t = torch.from_numpy(val_labels).float()
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(val_logits_t)
        loss = criterion(scaled.squeeze(), val_labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature.item()


def apply_temperature_scaling(val_probs, val_labels, test_probs):
    """Fit temperature on validation, apply to test, return calibrated probs and ECE."""
    val_logits = compute_logits_from_probs(val_probs)
    test_logits = compute_logits_from_probs(test_probs)

    temp = fit_temperature(val_logits, val_labels)
    calibrated_test_logits = test_logits / temp
    calibrated_test_probs = 1 / (1 + np.exp(-calibrated_test_logits))

    return calibrated_test_probs, temp


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() > 0:
            conf = probs[mask].mean()
            acc = ((probs[mask] >= 0.5).astype(int) == labels[mask]).mean()
            ece += np.abs(conf - acc) * mask.sum() / len(labels)
    return float(ece)


# ============================================================
# Pooled DeLong Test (across seeds)
# ============================================================

def pooled_delong_test(labels_per_seed, probs1_per_seed, probs2_per_seed):
    """
    Pool DeLong test across seeds by averaging the test statistic.
    Returns mean p-value and fraction of seeds with p < alpha.
    """
    sys_path = '/mnt/shared-workspace/shared'
    if sys_path not in __import__('sys').path:
        __import__('sys').path.insert(0, sys_path)
    from delong_test import delong_paired_comparison

    p_values = []
    for seed_idx in range(len(labels_per_seed)):
        labels = labels_per_seed[seed_idx]
        p1 = probs1_per_seed[seed_idx]
        p2 = probs2_per_seed[seed_idx]
        try:
            result = delong_paired_comparison(labels, [p1, p2], ['model1', 'model2'])
            if 'model1_vs_model2' in result:
                p_values.append(result['model1_vs_model2']['p_value'])
        except Exception:
            continue

    if not p_values:
        return {'mean_p': 1.0, 'median_p': 1.0, 'frac_significant': 0.0, 'n_seeds': 0}

    return {
        'mean_p': float(np.mean(p_values)),
        'median_p': float(np.median(p_values)),
        'min_p': float(np.min(p_values)),
        'max_p': float(np.max(p_values)),
        'frac_significant': float(np.mean([p < 0.005 for p in p_values])),  # Bonferroni-corrected
        'frac_uncorrected_sig': float(np.mean([p < 0.05 for p in p_values])),
        'n_seeds': len(p_values),
        'all_p_values': [float(p) for p in p_values],
    }


# ============================================================
# Decision Curve Analysis
# ============================================================

def decision_curve_analysis(probs, labels, threshold_probs=None):
    """
    Compute net benefit at different threshold probabilities.
    Returns net benefit for the model, treat-all, and treat-none strategies.
    """
    if threshold_probs is None:
        threshold_probs = np.arange(0.01, 0.99, 0.01)

    n = len(labels)
    results = []

    for tp in threshold_probs:
        # Model: predict positive if prob >= threshold
        predictions = (probs >= tp).astype(int)
        tp_count = ((predictions == 1) & (labels == 1)).sum()
        fp_count = ((predictions == 1) & (labels == 0)).sum()

        # Net benefit for model
        nb_model = tp_count / n - fp_count / n * (tp / (1 - tp))

        # Net benefit for treat-all
        prevalence = labels.mean()
        nb_all = prevalence - (1 - prevalence) * (tp / (1 - tp))

        # Net benefit for treat-none
        nb_none = 0.0

        results.append({
            'threshold': float(tp),
            'nb_model': float(nb_model),
            'nb_all': float(nb_all),
            'nb_none': float(nb_none),
        })

    return results
