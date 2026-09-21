"""
Experiment runner v2: Multi-seed, multi-backbone evaluation.
Trains 5 models × 10 seeds × 3 backbones, evaluates Q1-Q6 with statistics.

Usage:
  python run_experiments_v2.py --backbone vit_tiny
  python run_experiments_v2.py --backbone vit_b
  python run_experiments_v2.py --backbone resnet18
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import json, time, os, copy, argparse, warnings, collections, glob
from collections import defaultdict
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/mnt/shared-workspace/shared')
from pathology_models import (
    BaselineClassifier, DANNModel, SSLModel, DisentanglementModel,
    MultiAgentSystem, nt_xent_loss, grl
)
from delong_test import delong_paired_comparison, bonferroni_correct

# ============================================================
# Configuration
# ============================================================
SHARED_DIR = "/mnt/shared-workspace/shared"
RESULTS_DIR = "/mnt/shared-workspace/shared/results_v2"

BACKBONE_DIMS = {
    'vit_tiny': 192,
    'vit_b': 768,
    'resnet18': 512,
}

EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_DOMAINS = 4
MC_SAMPLES = 30
N_SEEDS = 10

# ============================================================
# Data loading
# ============================================================

def load_features(backbone, name):
    """Load cached features from npz. Handles both single-file and per-domain formats."""
    path = os.path.join(SHARED_DIR, "features_v2", backbone, f"{name}.npz")
    
    # For train splits, check if per-domain files exist (ViT-B robust extraction)
    if 'train' in name and not os.path.exists(path):
        domain_files = sorted(glob.glob(os.path.join(
            SHARED_DIR, "features_v2", backbone, f"{name}_domain*.npz")))
        if domain_files:
            all_feats, all_labels, all_domains = [], [], []
            for df in domain_files:
                data = np.load(df)
                all_feats.append(data['features'])
                all_labels.append(data['labels'])
                all_domains.append(data['domains'])
            features = np.concatenate(all_feats)
            labels = np.concatenate(all_labels)
            domains = np.concatenate(all_domains)
            if features.ndim > 2:
                features = features.squeeze()
            return features.astype(np.float32), labels.astype(np.int64), domains.astype(np.int64)
    
    data = np.load(path)
    features = data['features']
    if features.ndim > 2:
        features = features.squeeze()
    labels = data['labels']
    domains = data['domains']
    return features.astype(np.float32), labels.astype(np.int64), domains.astype(np.int64)

def make_loader(features, labels, domains=None, batch_size=BATCH_SIZE, shuffle=True):
    feats = torch.from_numpy(features)
    lbls = torch.from_numpy(labels).float()
    if domains is not None:
        doms = torch.from_numpy(domains).long()
        ds = TensorDataset(feats, lbls, doms)
    else:
        ds = TensorDataset(feats, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Training functions
# ============================================================

def quick_auc(model, feats, labels, mtype='baseline'):
    """Quick AUC evaluation during training."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats)
        if mtype == 'baseline':
            probs = torch.sigmoid(model(x).squeeze()).numpy()
        elif mtype == 'dann':
            probs = torch.sigmoid(model(x)[0].squeeze()).numpy()
        elif mtype == 'ssl':
            probs = torch.sigmoid(model(x).squeeze()).numpy()
        elif mtype == 'disent':
            probs = torch.sigmoid(model(x, lambda_=0.0)[0].squeeze()).numpy()
        elif mtype == 'multi_agent':
            probs = model.predict(x).numpy()
    return roc_auc_score(labels, probs)

def train_baseline(model, train_feats, train_labels, val_feats, val_labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(train_feats, train_labels, shuffle=True)
    best_val_auc, best_state = 0, None
    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(), y)
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_feats, val_labels, 'baseline')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_val_auc

def train_dann(model, train_feats, train_labels, train_domains, val_feats, val_labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_feats, train_labels, train_domains, shuffle=True)
    best_val_auc, best_state = 0, None
    for epoch in range(EPOCHS):
        model.train()
        p = epoch / EPOCHS
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        for x, y, d in loader:
            optimizer.zero_grad()
            tumor_logit, domain_logit = model(x, lambda_=lambda_)
            loss = criterion(tumor_logit.squeeze(), y) + 0.5 * dom_criterion(domain_logit, d)
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_feats, val_labels, 'dann')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_val_auc

def train_ssl(model, train_feats, train_labels, train_domains, val_feats, val_labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = make_loader(train_feats, train_labels, train_domains, shuffle=True)
    # Phase 1: Contrastive pre-training (20 epochs)
    for epoch in range(20):
        model.train()
        for x, y, d in loader:
            optimizer.zero_grad()
            noise = torch.randn_like(x) * 0.1
            z_i = model.project(x)
            z_j = model.project(x + noise)
            loss = nt_xent_loss(z_i, z_j, temperature=0.5)
            loss.backward()
            optimizer.step()
    # Phase 2: Fine-tune (30 epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc, best_state = 0, None
    for epoch in range(30):
        model.train()
        for x, y, d in loader:
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(), y)
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_feats, val_labels, 'ssl')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_val_auc

def train_disentanglement(model, train_feats, train_labels, train_domains, val_feats, val_labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_feats, train_labels, train_domains, shuffle=True)
    best_val_auc, best_state = 0, None
    for epoch in range(EPOCHS):
        model.train()
        p = epoch / EPOCHS
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        for x, y, d in loader:
            optimizer.zero_grad()
            tumor_logit, domain_logit, z_bio, z_dom = model(x, lambda_=lambda_, return_latents=True)
            tumor_loss = criterion(tumor_logit.squeeze(), y)
            domain_loss = dom_criterion(domain_logit, d)
            ortho_loss = model.orthogonality_loss(z_bio, z_dom)
            noise = torch.randn_like(x) * 0.1
            z_bio_aug, _ = model.encode(x + noise, lambda_)
            z1 = F.normalize(model.projector(z_bio.detach()), dim=1)
            z2 = F.normalize(model.projector(z_bio_aug), dim=1)
            ssl_loss = 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()
            loss = tumor_loss + 0.3 * domain_loss + 0.1 * ortho_loss + 0.1 * ssl_loss
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_feats, val_labels, 'disent')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_val_auc

def train_multi_agent(model, train_feats, train_labels, train_domains, val_feats, val_labels):
    agent_params = model.get_agent_parameters()
    optimizer = torch.optim.Adam([
        {'params': agent_params[k], 'lr': LR if k != 'orchestrator' else LR * 0.5}
        for k in agent_params
    ], weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_feats, train_labels, train_domains, shuffle=True)
    best_val_auc, best_state = 0, None

    for epoch in range(EPOCHS):
        model.train()
        p = epoch / EPOCHS
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        for x, y, d in loader:
            optimizer.zero_grad()
            # Agent 1: Quality
            quality_target = (d == 0).float()
            q_score, _ = model.quality_agent(x)
            quality_loss = F.binary_cross_entropy(q_score.clamp(1e-7, 1-1e-7), quality_target)
            # Agent 2: Tissue
            tissue_logit, _ = model.tissue_agent(x)
            tissue_loss = criterion(tissue_logit.squeeze(), y)
            # Agent 3: Domain
            domain_logit, z_bio = model.domain_agent(x, lambda_=lambda_)
            domain_loss = dom_criterion(domain_logit, d)
            # Agent 4: Tumor
            tumor_logit, _ = model.tumor_agent(x)
            tumor_loss = criterion(tumor_logit.squeeze(), y)
            # Agent 5: Biomarker
            biomarker_score, _ = model.biomarker_agent(x)
            biomarker_loss = F.mse_loss(biomarker_score, y.float())
            # Agent 6: Uncertainty
            unc_pred, _ = model.uncertainty_agent(x)
            unc_loss = F.binary_cross_entropy(unc_pred.clamp(1e-7, 1-1e-7), y.float())
            # Orchestrator
            agent_outputs = [q_score.squeeze(), torch.sigmoid(tissue_logit).squeeze(),
                             torch.sigmoid(tumor_logit).squeeze(), biomarker_score.squeeze(),
                             unc_pred.squeeze(), torch.sigmoid(tumor_logit).squeeze()]
            agent_outputs = [ao.squeeze() if ao.dim() > 1 else ao for ao in agent_outputs]
            fused_logit, _ = model.orchestrator(x, agent_outputs)
            fused_loss = criterion(fused_logit, y)
            # Contrastive on Z_bio
            noise = torch.randn_like(x) * 0.1
            _, z_bio_aug = model.domain_agent(x + noise, lambda_=lambda_)
            ssl_loss = 1.0 - F.cosine_similarity(
                F.normalize(z_bio, dim=1), F.normalize(z_bio_aug, dim=1), dim=1).mean()

            loss = (fused_loss + 0.3 * tumor_loss + 0.2 * tissue_loss + 0.2 * domain_loss +
                    0.1 * quality_loss + 0.1 * biomarker_loss + 0.1 * unc_loss + 0.1 * ssl_loss)
            loss.backward()
            optimizer.step()

        val_auc = quick_auc(model, val_feats, val_labels, 'multi_agent')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_val_auc

# ============================================================
# Evaluation functions
# ============================================================

def get_predictions(model, feats, mtype, eval_mode=True):
    """Get probability predictions from a model."""
    if eval_mode:
        model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats)
        if mtype == 'baseline':
            return torch.sigmoid(model(x).squeeze()).numpy()
        elif mtype == 'dann':
            return torch.sigmoid(model(x)[0].squeeze()).numpy()
        elif mtype == 'ssl':
            return torch.sigmoid(model(x).squeeze()).numpy()
        elif mtype == 'disent':
            return torch.sigmoid(model(x, lambda_=0.0)[0].squeeze()).numpy()
        elif mtype == 'multi_agent':
            return model.predict(x).numpy()

def evaluate_model(model, feats, labels, mtype):
    """Full evaluation: AUC, accuracy, sensitivity, specificity, F1, Brier."""
    probs = get_predictions(model, feats, mtype)
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(labels, preds)
    brier = brier_score_loss(labels, probs)
    return {
        'auc': float(auc), 'accuracy': float(acc), 'sensitivity': float(sens),
        'specificity': float(spec), 'f1': float(f1), 'brier': float(brier),
    }, probs

def evaluate_uncertainty(model, feats, labels, mtype):
    """Evaluate uncertainty metrics: ECE, entropy, epistemic."""
    model.train()  # Enable dropout for MC sampling
    with torch.no_grad():
        x = torch.from_numpy(feats)
        outs = torch.stack([torch.tensor(get_predictions(model, x.numpy(), mtype, eval_mode=False))
                            for _ in range(MC_SAMPLES)])
    model.eval()
    mean_probs = outs.mean(0).numpy()
    epistemic = outs.std(0).numpy()
    eps = 1e-7
    entropy = -(mean_probs * np.log(mean_probs + eps) +
                (1 - mean_probs) * np.log(1 - mean_probs + eps))
    # ECE (Expected Calibration Error)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (mean_probs >= bin_boundaries[i]) & (mean_probs < bin_boundaries[i + 1])
        if mask.sum() > 0:
            conf = mean_probs[mask].mean()
            acc = (preds := (mean_probs[mask] >= 0.5).astype(int) == labels[mask]).mean()
            ece += np.abs(conf - acc) * mask.sum() / len(labels)
    return {
        'ece': float(ece), 'mean_entropy': float(entropy.mean()),
        'mean_epistemic_std': float(epistemic.mean()),
    }

def evaluate_disentanglement(model, feats, labels, domains, mtype):
    """Evaluate disentanglement via probing."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats)
        if mtype == 'multi_agent':
            _, z_bio = model.domain_agent(x)
            z_dom = None
        elif mtype == 'disent':
            z_bio, z_dom = model.encode(x, lambda_=0.0)
        elif mtype == 'dann':
            _, _, z_bio = model(x, return_features=True)
            z_dom = None
        elif mtype == 'baseline':
            _, z_bio = model(x, return_features=True)
            z_dom = None
        else:
            z_bio = model.encoder(x)
            z_dom = None

        z_bio_np = z_bio.numpy()

    # Probe: predict domain from Z_bio (should be low if disentangled)
    if len(np.unique(domains)) > 1:
        # Shuffle before splitting — data is ordered by domain, so a sequential
        # split would train on some domains and test on unseen ones (giving 0.0 acc)
        n = len(domains)
        shuffle_idx = np.random.RandomState(42).permutation(n)
        z_bio_shuf = z_bio_np[shuffle_idx]
        dom_shuf = domains[shuffle_idx]
        lbl_shuf = labels[shuffle_idx]
        split = n // 2
        probe_dom = LogisticRegression(max_iter=500, C=1.0)
        probe_dom.fit(z_bio_shuf[:split], dom_shuf[:split])
        bio_dom_acc = probe_dom.score(z_bio_shuf[split:], dom_shuf[split:])
    else:
        bio_dom_acc = 0.0

    # Probe: predict label from Z_bio (should be high)
    if len(np.unique(domains)) > 1:
        probe_lbl = LogisticRegression(max_iter=500, C=1.0)
        probe_lbl.fit(z_bio_shuf[:split], lbl_shuf[:split])
        bio_lbl_acc = probe_lbl.score(z_bio_shuf[split:], lbl_shuf[split:])
    else:
        probe_lbl = LogisticRegression(max_iter=500, C=1.0)
        probe_lbl.fit(z_bio_np[:len(labels)//2], labels[:len(labels)//2])
        bio_lbl_acc = probe_lbl.score(z_bio_np[len(labels)//2:], labels[len(labels)//2:])

    # Normalized cross-covariance between Z_bio and domain one-hot
    n_classes = len(np.unique(domains))
    if n_classes > 1:
        dom_onehot = np.eye(n_classes)[domains]
        cross_cov = np.mean(np.abs(np.corrcoef(z_bio_np.T, dom_onehot.T)[:z_bio_np.shape[1], z_bio_np.shape[1]:]))
    else:
        cross_cov = 0.0

    return {
        'bio_domain_probe_acc': float(bio_dom_acc),
        'bio_label_probe_acc': float(bio_lbl_acc),
        'normalized_cross_covariance': float(cross_cov) if not np.isnan(cross_cov) else 0.0,
    }

def evaluate_perturbation_robustness(model, pert_dir, labels, mtype):
    """Evaluate robustness using pre-extracted perturbation features."""
    results = {'perturbations': defaultdict(list), 'base_auc': 0.0}

    # Baseline
    baseline_data = np.load(os.path.join(pert_dir, "baseline.npz"))
    base_probs = get_predictions(model, baseline_data['features'].astype(np.float32), mtype)
    base_auc = roc_auc_score(labels, base_probs)
    results['base_auc'] = float(base_auc)

    # Each perturbation condition
    perturbation_types = {
        'gaussian_noise': ['mild', 'moderate', 'severe'],
        'stain_shift': ['mild', 'moderate', 'severe'],
        'gaussian_blur': ['mild', 'moderate', 'severe'],
        'jpeg_compression': ['mild', 'moderate', 'severe'],
        'brightness_contrast': ['mild', 'moderate', 'severe'],
    }

    all_deltas = []
    for ptype, sev_labels in perturbation_types.items():
        for sev_label in sev_labels:
            cond_name = f"{ptype}_{sev_label}"
            fpath = os.path.join(pert_dir, f"{cond_name}.npz")
            if not os.path.exists(fpath):
                continue
            data = np.load(fpath)
            probs = get_predictions(model, data['features'].astype(np.float32), mtype)
            auc = roc_auc_score(labels, probs)
            delta = auc - base_auc
            results['perturbations'][ptype].append({
                'severity': sev_label, 'auc': float(auc), 'delta_auc': float(delta),
                'abs_delta_auc': float(abs(delta)),
            })
            all_deltas.append(abs(delta))

    results['mean_abs_delta_auc'] = float(np.mean(all_deltas)) if all_deltas else 0.0
    return results

def optimize_threshold(model, val_feats, val_labels):
    """Sweep disagreement threshold on validation set to optimize flag rate."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(val_feats)
        out = model(x)

    disagreement = out['disagreement'].numpy()
    flag_rates = []
    thresholds = np.arange(0.05, 0.55, 0.025)

    for thresh in thresholds:
        flags = disagreement > thresh
        flag_rate = flags.mean()
        # AUC on non-flagged cases
        non_flagged = ~flags
        if non_flagged.sum() > 10 and len(np.unique(val_labels[non_flagged])) > 1:
            probs = out['consensus'].numpy()
            auc_non_flagged = roc_auc_score(val_labels[non_flagged], probs[non_flagged])
        else:
            auc_non_flagged = 0.5
        flag_rates.append({
            'threshold': float(thresh),
            'flag_rate': float(flag_rate),
            'non_flagged_auc': float(auc_non_flagged),
            'n_flagged': int(flags.sum()),
            'n_total': int(len(flags)),
        })

    # Select threshold achieving ~20% flag rate
    target_flag_rate = 0.20
    best_idx = np.argmin([abs(fr['flag_rate'] - target_flag_rate) for fr in flag_rates])
    optimal = flag_rates[best_idx]

    return {'sweep': flag_rates, 'optimal': optimal}

# ============================================================
# Statistics
# ============================================================

def compute_ci(values, confidence=0.95):
    """Compute 95% CI using t-distribution."""
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_half = t_crit * std / np.sqrt(n)
    return float(mean), float(ci_half)

def aggregate_results(per_seed_results):
    """Aggregate results across seeds with mean ± 95% CI."""
    aggregated = {}
    metric_keys = ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'brier']

    for q_name, q_results in per_seed_results.items():
        if q_name in ('training_info', 'q5_agent_weights'):
            aggregated[q_name] = q_results
            continue

        # Skip empty results (e.g., Q2 skipped when no perturbation features)
        if not q_results or (isinstance(q_results, dict) and len(q_results) == 0):
            print(f"  [SKIP] {q_name}: no results (empty)")
            continue

        # q_results is {seed: {model_name: {metric: value}}}
        # Collect all model names across all seeds
        all_model_names = set()
        for seed_data in q_results.values():
            all_model_names.update(seed_data.keys())
        all_model_names = sorted(all_model_names)

        # Detect result type from first seed's first model
        first_seed = q_results[list(q_results.keys())[0]]
        first_model = first_seed[list(first_seed.keys())[0]]

        if 'perturbations' in first_model:
            # Q2 perturbation results — structure: {seed: {model: {perturbations: {...}, base_auc, mean_abs_delta_auc}}}
            agg_q = {}
            for model_name in all_model_names:
                model_agg = {'perturbations': {}, 'base_auc': {}, 'mean_abs_delta_auc': {}}
                base_aucs = [q_results[s][model_name]['base_auc'] for s in q_results if model_name in q_results[s]]
                mean_deltas = [q_results[s][model_name]['mean_abs_delta_auc'] for s in q_results if model_name in q_results[s]]
                if base_aucs:
                    bm, bci = compute_ci(base_aucs)
                    dm, dci = compute_ci(mean_deltas)
                    model_agg['base_auc'] = {'mean': bm, 'ci': bci}
                    model_agg['mean_abs_delta_auc'] = {'mean': dm, 'ci': dci}

                ptypes = set()
                for s in q_results:
                    if model_name in q_results[s]:
                        ptypes.update(q_results[s][model_name]['perturbations'].keys())
                for ptype in sorted(ptypes):
                    model_agg['perturbations'][ptype] = []
                    for sev_idx in range(3):
                        aucs, deltas = [], []
                        for s in q_results:
                            if model_name in q_results[s] and ptype in q_results[s][model_name]['perturbations']:
                                perts = q_results[s][model_name]['perturbations'][ptype]
                                if sev_idx < len(perts):
                                    aucs.append(perts[sev_idx]['auc'])
                                    deltas.append(perts[sev_idx]['delta_auc'])
                        if aucs:
                            am, aci = compute_ci(aucs)
                            dm, dci = compute_ci(deltas)
                            model_agg['perturbations'][ptype].append({
                                'auc': {'mean': am, 'ci': aci},
                                'delta_auc': {'mean': dm, 'ci': dci},
                            })
                agg_q[model_name] = model_agg
            aggregated[q_name] = agg_q
            continue

        if 'ece' in first_model:
            # Q4 uncertainty results
            agg_q = {}
            metric_keys_q4 = ['ece', 'mean_entropy', 'mean_epistemic_std']
            for model_name in all_model_names:
                agg_q[model_name] = {}
                for mk in metric_keys_q4:
                    vals = [q_results[s][model_name][mk] for s in q_results if model_name in q_results[s] and mk in q_results[s][model_name]]
                    if vals:
                        m, ci = compute_ci(vals)
                        agg_q[model_name][mk] = {'mean': m, 'ci': ci}
            aggregated[q_name] = agg_q
            continue

        if 'bio_domain_probe_acc' in first_model:
            # Q5 disentanglement
            agg_q = {}
            probe_keys = ['bio_domain_probe_acc', 'bio_label_probe_acc', 'normalized_cross_covariance']
            for model_name in all_model_names:
                agg_q[model_name] = {}
                for pk in probe_keys:
                    vals = [q_results[s][model_name][pk] for s in q_results if model_name in q_results[s] and pk in q_results[s][model_name]]
                    if vals:
                        m, ci = compute_ci(vals)
                        agg_q[model_name][pk] = {'mean': m, 'ci': ci}
            aggregated[q_name] = agg_q
            continue

        # Standard Q1/Q3/Q5-ablation/Q6 results
        agg_q = {}
        for model_name in all_model_names:
            agg_q[model_name] = {}
            for mk in metric_keys:
                vals = [q_results[s][model_name][mk] for s in q_results if model_name in q_results[s] and mk in q_results[s][model_name]]
                if vals:
                    m, ci = compute_ci(vals)
                    agg_q[model_name][mk] = {'mean': m, 'ci': ci}
            # Q6 specific metrics
            for mk in ['inference_time_ms', 'flag_rate', 'n_flagged_for_review']:
                vals = [q_results[s][model_name][mk] for s in q_results if model_name in q_results[s] and mk in q_results[s][model_name]]
                if vals:
                    m, ci = compute_ci(vals)
                    agg_q[model_name][mk] = {'mean': m, 'ci': ci}
        aggregated[q_name] = agg_q

    return aggregated

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['vit_tiny', 'vit_b', 'resnet18'], required=True)
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    args = parser.parse_args()

    backbone = args.backbone
    feature_dim = BACKBONE_DIMS[backbone]
    n_seeds = args.seeds
    pert_dir = os.path.join(SHARED_DIR, "features_v2", backbone, "perturbations")
    out_dir = os.path.join(RESULTS_DIR, backbone)
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    print(f"=== Experiment Runner v2: {backbone} (dim={feature_dim}, seeds={n_seeds}) ===")

    # Load all features
    print("\nLoading features...")
    id_train_f, id_train_l, id_train_d = load_features(backbone, "camelyon17_id_train")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features(backbone, "pcam_train")
    id_val_f, id_val_l, id_val_d = load_features(backbone, "camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features(backbone, "camelyon17_ood_test")
    pcam_test_f, pcam_test_l, pcam_test_d = load_features(backbone, "pcam_test")
    pcam_val_f, pcam_val_l, _ = load_features(backbone, "pcam_validation")

    # Combine training data: Camelyon17 (domains 0-2) + PCam (domain 3)
    train_f = np.concatenate([id_train_f, pcam_train_f])
    train_l = np.concatenate([id_train_l, pcam_train_l])
    train_d = np.concatenate([id_train_d, pcam_train_d])
    print(f"  Train: {train_f.shape}, domains={collections.Counter(train_d.tolist())}")
    print(f"  ID val: {id_val_f.shape}, OOD test: {ood_test_f.shape}")
    print(f"  PCam test: {pcam_test_f.shape}, PCam val: {pcam_val_f.shape}")

    # Per-seed results storage
    all_per_seed = {
        'q1': {}, 'q2': {}, 'q3': {}, 'q3_pcam': {},
        'q4': {}, 'q4_id': {}, 'q5_disent': {}, 'q5_ablation': {},
        'q6': {}, 'predictions': {},
    }

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed}/{n_seeds-1}")
        print(f"{'='*60}")
        set_seeds(seed)
        t_seed = time.time()

        # Create models
        model1 = BaselineClassifier(input_dim=feature_dim)
        model2 = DANNModel(input_dim=feature_dim, n_domains=N_DOMAINS)
        model3 = SSLModel(input_dim=feature_dim)
        model4 = DisentanglementModel(input_dim=feature_dim, n_domains=N_DOMAINS)
        model5 = MultiAgentSystem(input_dim=feature_dim, n_domains=N_DOMAINS)

        # Train
        print("  Training baseline...", end=' ')
        m1, auc1 = train_baseline(model1, train_f, train_l, id_val_f, id_val_l)
        print(f"val AUC={auc1:.4f}")

        print("  Training DANN...", end=' ')
        m2, auc2 = train_dann(model2, train_f, train_l, train_d, id_val_f, id_val_l)
        print(f"val AUC={auc2:.4f}")

        print("  Training SSL...", end=' ')
        m3, auc3 = train_ssl(model3, train_f, train_l, train_d, id_val_f, id_val_l)
        print(f"val AUC={auc3:.4f}")

        print("  Training Disentanglement...", end=' ')
        m4, auc4 = train_disentanglement(model4, train_f, train_l, train_d, id_val_f, id_val_l)
        print(f"val AUC={auc4:.4f}")

        print("  Training Multi-Agent...", end=' ')
        m5, auc5 = train_multi_agent(model5, train_f, train_l, train_d, id_val_f, id_val_l)
        print(f"val AUC={auc5:.4f}")

        models = {
            'baseline': (m1, 'baseline'),
            'dann': (m2, 'dann'),
            'ssl': (m3, 'ssl'),
            'disentanglement': (m4, 'disent'),
            'multi_agent': (m5, 'multi_agent'),
        }

        # Q1: Diagnostic Validity
        print("  Q1: Diagnostic Validity...")
        q1 = {}
        for name, (model, mtype) in models.items():
            r, probs = evaluate_model(model, id_val_f, id_val_l, mtype)
            q1[name] = r
            all_per_seed['predictions'].setdefault(name, {})['id_val'] = probs
        all_per_seed['q1'][seed] = q1

        # Q3: External Generalization
        print("  Q3: OOD Test...")
        q3 = {}
        for name, (model, mtype) in models.items():
            r, probs = evaluate_model(model, ood_test_f, ood_test_l, mtype)
            q3[name] = r
            all_per_seed['predictions'].setdefault(name, {})['ood_test'] = probs
        all_per_seed['q3'][seed] = q3

        print("  Q3: PCam Test...")
        q3_pcam = {}
        for name, (model, mtype) in models.items():
            r, probs = evaluate_model(model, pcam_test_f, pcam_test_l, mtype)
            q3_pcam[name] = r
            all_per_seed['predictions'].setdefault(name, {})['pcam_test'] = probs
        all_per_seed['q3_pcam'][seed] = q3_pcam

        # Q2: Perturbation Robustness
        if os.path.exists(pert_dir):
            print("  Q2: Perturbation Robustness...")
            q2 = {}
            for name, (model, mtype) in models.items():
                q2[name] = evaluate_perturbation_robustness(model, pert_dir, pcam_test_l, mtype)
            all_per_seed['q2'][seed] = q2
        else:
            print("  Q2: SKIPPED (no perturbation features found)")

        # Q4: Uncertainty
        print("  Q4: Uncertainty (OOD)...")
        q4 = {}
        for name, (model, mtype) in models.items():
            q4[name] = evaluate_uncertainty(model, ood_test_f, ood_test_l, mtype)
        all_per_seed['q4'][seed] = q4

        print("  Q4: Uncertainty (ID)...")
        q4_id = {}
        for name, (model, mtype) in models.items():
            q4_id[name] = evaluate_uncertainty(model, id_val_f, id_val_l, mtype)
        all_per_seed['q4_id'][seed] = q4_id

        # Q5: Disentanglement (use training data for domain probe — has 4 domains)
        print("  Q5: Disentanglement...")
        q5_disent = {}
        for name, (model, mtype) in [('baseline', (m1, 'baseline')), ('dann', (m2, 'dann')),
                                      ('disentanglement', (m4, 'disent')), ('multi_agent', (m5, 'multi_agent'))]:
            q5_disent[name] = evaluate_disentanglement(model, train_f, train_l, train_d, mtype)
        all_per_seed['q5_disent'][seed] = q5_disent

        # Q5: Ablation
        print("  Q5: Agent Ablation...")
        ablation = {}
        r_full, _ = evaluate_model(m5, ood_test_f, ood_test_l, 'multi_agent')
        ablation['full_system'] = r_full
        for agent_name, agent_module in [('tumor_only', m5.tumor_agent), ('tissue_only', m5.tissue_agent),
                                          ('uncertainty_only', m5.uncertainty_agent)]:
            agent_module.eval()
            with torch.no_grad():
                x = torch.from_numpy(ood_test_f)
                if agent_name in ('tumor_only', 'tissue_only'):
                    logit, _ = agent_module(x)
                    probs = torch.sigmoid(logit.squeeze()).numpy()
                else:
                    probs, _ = agent_module(x)
                    probs = probs.numpy()
            auc = roc_auc_score(ood_test_l, probs)
            acc = accuracy_score(ood_test_l, (probs >= 0.5).astype(int))
            ablation[agent_name] = {'auc': float(auc), 'accuracy': float(acc)}
        all_per_seed['q5_ablation'][seed] = ablation

        # Q6: Efficiency
        print("  Q6: Efficiency...")
        q6 = {}
        for name, (model, mtype) in models.items():
            model.eval()
            x = torch.from_numpy(ood_test_f[:1000])
            t0 = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
            elapsed = (time.time() - t0) / 10
            q6[name] = {'inference_time_ms': float(elapsed * 1000), 'n_samples': 1000}

        # Multi-agent flag rate
        m5.eval()
        with torch.no_grad():
            x = torch.from_numpy(ood_test_f)
            out = m5(x)
        n_flagged = out['flag_for_review'].sum().item()
        q6['multi_agent']['n_flagged_for_review'] = int(n_flagged)
        q6['multi_agent']['flag_rate'] = float(n_flagged / len(ood_test_l))
        all_per_seed['q6'][seed] = q6

        print(f"  Seed {seed} completed in {time.time()-t_seed:.1f}s")

        # Checkpoint: save per-seed results after each seed
        checkpoint_path = os.path.join("/workspace", f"checkpoint_{backbone}_seed{seed}.json")
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump({k: v for k, v in all_per_seed.items() if k != 'predictions'}, f, indent=2, default=str)
        except Exception as e:
            print(f"  [WARNING] Checkpoint save failed: {e}")

    # ===== Threshold Optimization (using seed 0 multi-agent model) =====
    print("\nThreshold Optimization...")
    # Re-train multi-agent with seed 0 for threshold optimization
    set_seeds(0)
    m5_thresh = MultiAgentSystem(input_dim=feature_dim, n_domains=N_DOMAINS)
    m5_thresh, _ = train_multi_agent(m5_thresh, train_f, train_l, train_d, id_val_f, id_val_l)
    thresh_results = optimize_threshold(m5_thresh, id_val_f, id_val_l)
    # Also evaluate on OOD test with optimal threshold
    m5_thresh.eval()
    with torch.no_grad():
        x = torch.from_numpy(ood_test_f)
        out = m5_thresh(x)
    disagreement = out['disagreement'].numpy()
    flags = disagreement > thresh_results['optimal']['threshold']
    non_flagged = ~flags
    if non_flagged.sum() > 10 and len(np.unique(ood_test_l[non_flagged])) > 1:
        probs = out['consensus'].numpy()
        ood_auc_non_flagged = roc_auc_score(ood_test_l[non_flagged], probs[non_flagged])
    else:
        ood_auc_non_flagged = 0.5
    thresh_results['ood_with_optimal_threshold'] = {
        'flag_rate': float(flags.mean()),
        'n_flagged': int(flags.sum()),
        'n_total': int(len(flags)),
        'non_flagged_auc': float(ood_auc_non_flagged),
    }
    print(f"  Optimal threshold: {thresh_results['optimal']['threshold']:.3f}")
    print(f"  ID val flag rate: {thresh_results['optimal']['flag_rate']:.3f}")
    print(f"  OOD flag rate: {thresh_results['ood_with_optimal_threshold']['flag_rate']:.3f}")

    # ===== DeLong Tests (using seed 0 predictions) =====
    print("\nDeLong Tests...")
    seed0_preds = {name: all_per_seed['predictions'][name] for name in all_per_seed['predictions']}

    delong_id = delong_paired_comparison(id_val_l,
        [seed0_preds[n]['id_val'] for n in ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']],
        ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent'])
    delong_id = bonferroni_correct(delong_id, alpha=0.05)

    delong_ood = delong_paired_comparison(ood_test_l,
        [seed0_preds[n]['ood_test'] for n in ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']],
        ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent'])
    delong_ood = bonferroni_correct(delong_ood, alpha=0.05)

    delong_pcam = delong_paired_comparison(pcam_test_l,
        [seed0_preds[n]['pcam_test'] for n in ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']],
        ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent'])
    delong_pcam = bonferroni_correct(delong_pcam, alpha=0.05)

    # ===== Agent weights (seed 0) =====
    agent_weights = m5_thresh.orchestrator.agent_weights.detach().numpy()
    agent_names = ['Quality', 'Tissue', 'Tumor', 'Biomarker', 'Uncertainty', 'Tumor_dup']

    # ===== Aggregate across seeds =====
    print("\nAggregating results across seeds...")
    aggregated = aggregate_results({
        'q1_diagnostic_validity': all_per_seed['q1'],
        'q3_external_generalization': all_per_seed['q3'],
        'q3_pcam_external': all_per_seed['q3_pcam'],
        'q2_robustness': all_per_seed['q2'],
        'q4_uncertainty': all_per_seed['q4'],
        'q4_uncertainty_id': all_per_seed['q4_id'],
        'q5_disentanglement': all_per_seed['q5_disent'],
        'q5_ablation': all_per_seed['q5_ablation'],
        'q6_efficiency': all_per_seed['q6'],
    })

    # Add non-aggregated results
    aggregated['delong_tests'] = {
        'id_val': delong_id,
        'ood_test': delong_ood,
        'pcam_test': delong_pcam,
    }
    aggregated['threshold_optimization'] = thresh_results
    aggregated['q5_agent_weights'] = {
        'agents': agent_names,
        'weights': agent_weights.tolist(),
    }
    aggregated['training_info'] = {
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
        'weight_decay': WEIGHT_DECAY, 'n_domains': N_DOMAINS,
        'mc_samples': MC_SAMPLES, 'n_seeds': n_seeds,
        'feature_dim': feature_dim, 'backbone': backbone,
        'id_train_size': int(len(id_train_l)),
        'pcam_train_size': int(len(pcam_train_l)),
        'total_train_size': int(len(train_l)),
        'id_val_size': int(len(id_val_l)),
        'ood_test_size': int(len(ood_test_l)),
        'pcam_test_size': int(len(pcam_test_l)),
        'pcam_val_size': int(len(pcam_val_l)),
    }

    # Save (to local disk first, then cp to S3 mount for FUSE compatibility)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "all_results.json")
    tmp_results = os.path.join("/workspace", f"results_{backbone}_tmp.json")
    with open(tmp_results, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    import subprocess as sp
    sp.run(['cp', tmp_results, results_path], check=True)
    os.remove(tmp_results)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for q_name in ['q1_diagnostic_validity', 'q3_external_generalization', 'q3_pcam_external']:
        if q_name in aggregated:
            print(f"\n{q_name}:")
            for model_name, metrics in aggregated[q_name].items():
                if 'auc' in metrics:
                    print(f"  {model_name:20s}: AUC={metrics['auc']['mean']:.4f} ± {metrics['auc']['ci']:.4f}")

    return aggregated

if __name__ == "__main__":
    main()
