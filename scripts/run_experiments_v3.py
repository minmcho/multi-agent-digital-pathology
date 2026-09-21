"""
Experiment runner v3: prediction saving + new model variants.
Supports: original 5 models, orchestrator variants, MixStyle, FACT,
disentanglement sweep, ViT-Tiny subsample.

Usage:
  python run_experiments_v3.py --backbone vit_tiny --experiment all
  python run_experiments_v3.py --backbone vit_tiny --experiment orchestrator
  python run_experiments_v3.py --backbone vit_tiny --experiment disentanglement_sweep
  python run_experiments_v3.py --backbone vit_tiny --experiment mixstyle_fact
  python run_experiments_v3.py --backbone vit_tiny --experiment subsample_40k
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, brier_score_loss)
from sklearn.linear_model import LogisticRegression
import json, time, os, copy, argparse, warnings, collections, glob, shutil
from collections import defaultdict
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/mnt/shared-workspace/shared')
sys.path.insert(0, '/workspace')
from pathology_models import (
    BaselineClassifier, DANNModel, SSLModel, DisentanglementModel,
    MultiAgentSystem, nt_xent_loss, grl
)
from pathology_models_v3 import (
    MultiAgentSystemV3, MixStyleClassifier, FACTClassifier,
    DisentanglementModelV3
)
from delong_test import delong_paired_comparison, bonferroni_correct

# ============================================================
# Configuration
# ============================================================
SHARED_DIR = "/mnt/shared-workspace/shared"
RESULTS_DIR = "/mnt/shared-workspace/shared/results_v3"
PRED_DIR = "/mnt/shared-workspace/shared/predictions_v3"

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
# Data loading (same as v2)
# ============================================================

def load_features(backbone, name):
    path = os.path.join(SHARED_DIR, "features_v2", backbone, f"{name}.npz")
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Training functions (same as v2, with prediction saving)
# ============================================================

def quick_auc(model, feats, labels, mtype='baseline'):
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
        elif mtype == 'mixstyle':
            probs = torch.sigmoid(model(x).squeeze()).numpy()
        elif mtype == 'fact':
            probs = torch.sigmoid(model(x).squeeze()).numpy()
        elif mtype.startswith('multi_agent_'):
            probs = model.predict(x).numpy()
    return roc_auc_score(labels, probs)

def train_baseline(model, train_f, train_l, val_f, val_l):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(train_f, train_l, shuffle=True)
    best_auc, best_state = 0, None
    for epoch in range(EPOCHS):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(), y)
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_f, val_l, 'baseline')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc

def train_dann(model, train_f, train_l, train_d, val_f, val_l):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_f, train_l, train_d, shuffle=True)
    best_auc, best_state = 0, None
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
        val_auc = quick_auc(model, val_f, val_l, 'dann')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc

def train_ssl(model, train_f, train_l, train_d, val_f, val_l):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = make_loader(train_f, train_l, train_d, shuffle=True)
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
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None
    for epoch in range(30):
        model.train()
        for x, y, d in loader:
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(), y)
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_f, val_l, 'ssl')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc

def train_disentanglement(model, train_f, train_l, train_d, val_f, val_l, disent_lambda=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_f, train_l, train_d, shuffle=True)
    best_auc, best_state = 0, None
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
            loss = tumor_loss + disent_lambda * 0.3 * domain_loss + 0.1 * ortho_loss + 0.1 * ssl_loss
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_f, val_l, 'disent')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc

def train_multi_agent(model, train_f, train_l, train_d, val_f, val_l, ood_val_f=None, ood_val_l=None):
    agent_params = model.get_agent_parameters()
    optimizer = torch.optim.Adam([
        {'params': agent_params[k], 'lr': LR if k != 'orchestrator' else LR * 0.5}
        for k in agent_params
    ], weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_f, train_l, train_d, shuffle=True)
    best_auc, best_state = 0, None
    # For OOD-validated variant, use OOD val for model selection
    selection_f = ood_val_f if (ood_val_f is not None and hasattr(model, 'variant') and model.variant == 'ood_validated') else val_f
    selection_l = ood_val_l if (ood_val_l is not None and hasattr(model, 'variant') and model.variant == 'ood_validated') else val_l
    mtype = 'multi_agent' if not hasattr(model, 'variant') or model.variant == 'original' else f'multi_agent_{model.variant}'

    for epoch in range(EPOCHS):
        model.train()
        p = epoch / EPOCHS
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        for x, y, d in loader:
            optimizer.zero_grad()
            q_score, _ = model.quality_agent(x)
            quality_loss = F.binary_cross_entropy(q_score.clamp(1e-7, 1-1e-7), (d == 0).float())
            tissue_logit, _ = model.tissue_agent(x)
            tissue_loss = criterion(tissue_logit.squeeze(), y)
            domain_logit, z_bio = model.domain_agent(x, lambda_=lambda_)
            domain_loss = dom_criterion(domain_logit, d)
            tumor_logit, _ = model.tumor_agent(x)
            tumor_loss = criterion(tumor_logit.squeeze(), y)
            biomarker_score, _ = model.biomarker_agent(x)
            biomarker_loss = F.mse_loss(biomarker_score, y.float())
            unc_pred, _ = model.uncertainty_agent(x)
            unc_loss = F.binary_cross_entropy(unc_pred.clamp(1e-7, 1-1e-7), y.float())
            agent_outputs = [q_score.squeeze(), torch.sigmoid(tissue_logit).squeeze(),
                             torch.sigmoid(tumor_logit).squeeze(), biomarker_score.squeeze(),
                             unc_pred.squeeze(), torch.sigmoid(tumor_logit).squeeze()]
            agent_outputs = [ao.squeeze() if ao.dim() > 1 else ao for ao in agent_outputs]

            if hasattr(model, 'variant') and model.variant == 'uncertainty_weighted':
                fused_logit, _ = model.orchestrator(x, agent_outputs,
                    agent_uncertainties=[torch.zeros_like(ao) for ao in agent_outputs])
            else:
                fused_logit, _ = model.orchestrator(x, agent_outputs)
            fused_loss = criterion(fused_logit, y)
            noise = torch.randn_like(x) * 0.1
            _, z_bio_aug = model.domain_agent(x + noise, lambda_=lambda_)
            ssl_loss = 1.0 - F.cosine_similarity(
                F.normalize(z_bio, dim=1), F.normalize(z_bio_aug, dim=1), dim=1).mean()
            loss = (fused_loss + 0.3 * tumor_loss + 0.2 * tissue_loss + 0.2 * domain_loss +
                    0.1 * quality_loss + 0.1 * biomarker_loss + 0.1 * unc_loss + 0.1 * ssl_loss)
            loss.backward()
            optimizer.step()

        val_auc = quick_auc(model, selection_f, selection_l, mtype)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc

def train_mixstyle(model, train_f, train_l, train_d, val_f, val_l):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(train_f, train_l, train_d, shuffle=True)
    best_auc, best_state = 0, None
    for epoch in range(EPOCHS):
        model.train()
        for x, y, d in loader:
            optimizer.zero_grad()
            logits = model(x, domains=d)
            loss = criterion(logits.squeeze(), y)
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_f, val_l, 'mixstyle')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc

def train_fact(model, train_f, train_l, train_d, val_f, val_l):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(train_f, train_l, train_d, shuffle=True)
    best_auc, best_state = 0, None
    for epoch in range(EPOCHS):
        model.train()
        for x, y, d in loader:
            optimizer.zero_grad()
            logits = model(x, domains=d)
            loss = criterion(logits.squeeze(), y)
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_f, val_l, 'fact')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc

# ============================================================
# Prediction extraction
# ============================================================

def get_predictions(model, feats, mtype):
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
        elif mtype == 'mixstyle':
            return torch.sigmoid(model(x).squeeze()).numpy()
        elif mtype == 'fact':
            return torch.sigmoid(model(x).squeeze()).numpy()
        elif mtype.startswith('multi_agent_'):
            return model.predict(x).numpy()

def evaluate_model(model, feats, labels, mtype):
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

def compute_ci(values, confidence=0.95):
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_half = t_crit * std / np.sqrt(n)
    return float(mean), float(ci_half)

def save_predictions(pred_dir, backbone, model_name, seed, dataset_name, probs, labels):
    """Save raw predictions to disk. Write to /workspace first then copy (S3 mount can't do ZIP random-access)."""
    os.makedirs(os.path.join(pred_dir, backbone), exist_ok=True)
    path = os.path.join(pred_dir, backbone, f"{model_name}_seed{seed}_{dataset_name}.npz")
    tmp = os.path.join("/workspace", "pred_tmp", f"{model_name}_seed{seed}_{dataset_name}.npz")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    np.savez(tmp, probs=probs, labels=labels)
    shutil.copy(tmp, path)

# ============================================================
# Main experiment functions
# ============================================================

def run_all_models(backbone, n_seeds, subsample=False):
    """Run all 5 original models + save predictions."""
    feature_dim = BACKBONE_DIMS[backbone]
    out_dir = os.path.join(RESULTS_DIR, backbone)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== All Models: {backbone} (dim={feature_dim}, seeds={n_seeds}) ===")

    # Load features
    id_train_f, id_train_l, id_train_d = load_features(backbone, "camelyon17_id_train")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features(backbone, "pcam_train")
    id_val_f, id_val_l, id_val_d = load_features(backbone, "camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features(backbone, "camelyon17_ood_test")
    pcam_test_f, pcam_test_l, pcam_test_d = load_features(backbone, "pcam_test")
    pcam_val_f, pcam_val_l, _ = load_features(backbone, "pcam_validation")

    # Optionally subsample to 40K (10K per domain)
    if subsample:
        print("  Subsampling to 10K per domain (40K total)...")
        np.random.seed(42)
        idx_train = []
        for dom in range(N_DOMAINS - 1):
            dom_idx = np.where(id_train_d == dom)[0]
            idx_train.append(np.random.choice(dom_idx, min(10000, len(dom_idx)), replace=False))
        idx_pcam = np.random.choice(len(pcam_train_l), min(10000, len(pcam_train_l)), replace=False)
        id_train_f = id_train_f[np.concatenate(idx_train)]
        id_train_l = id_train_l[np.concatenate(idx_train)]
        id_train_d = id_train_d[np.concatenate(idx_train)]
        pcam_train_f = pcam_train_f[idx_pcam]
        pcam_train_l = pcam_train_l[idx_pcam]
        pcam_train_d = pcam_train_d[idx_pcam]

    train_f = np.concatenate([id_train_f, pcam_train_f])
    train_l = np.concatenate([id_train_l, pcam_train_l])
    train_d = np.concatenate([id_train_d, pcam_train_d])
    print(f"  Train: {train_f.shape}, domains={collections.Counter(train_d.tolist())}")

    all_results = {'q1': {}, 'q3': {}, 'q3_pcam': {}, 'q4': {}, 'q5_disent': {}, 'q5_ablation': {}, 'q6': {}}

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed}/{n_seeds-1} ---")
        set_seeds(seed)
        t0 = time.time()

        models = {}
        # Baseline
        m = BaselineClassifier(input_dim=feature_dim)
        m, auc = train_baseline(m, train_f, train_l, id_val_f, id_val_l)
        models['baseline'] = (m, 'baseline')
        print(f"  baseline: val AUC={auc:.4f}")

        # DANN
        m = DANNModel(input_dim=feature_dim, n_domains=N_DOMAINS)
        m, auc = train_dann(m, train_f, train_l, train_d, id_val_f, id_val_l)
        models['dann'] = (m, 'dann')
        print(f"  dann: val AUC={auc:.4f}")

        # SSL
        m = SSLModel(input_dim=feature_dim)
        m, auc = train_ssl(m, train_f, train_l, train_d, id_val_f, id_val_l)
        models['ssl'] = (m, 'ssl')
        print(f"  ssl: val AUC={auc:.4f}")

        # Disentanglement
        m = DisentanglementModel(input_dim=feature_dim, n_domains=N_DOMAINS)
        m, auc = train_disentanglement(m, train_f, train_l, train_d, id_val_f, id_val_l)
        models['disentanglement'] = (m, 'disent')
        print(f"  disent: val AUC={auc:.4f}")

        # Multi-Agent
        m = MultiAgentSystem(input_dim=feature_dim, n_domains=N_DOMAINS)
        m, auc = train_multi_agent(m, train_f, train_l, train_d, id_val_f, id_val_l)
        models['multi_agent'] = (m, 'multi_agent')
        print(f"  multi_agent: val AUC={auc:.4f}")

        # Evaluate + save predictions
        for dataset_name, (feats, labels) in [('id_val', (id_val_f, id_val_l)),
                                                ('ood_test', (ood_test_f, ood_test_l)),
                                                ('pcam_test', (pcam_test_f, pcam_test_l))]:
            q_key = 'q1' if dataset_name == 'id_val' else ('q3' if dataset_name == 'ood_test' else 'q3_pcam')
            all_results[q_key].setdefault(seed, {})
            for name, (model, mtype) in models.items():
                r, probs = evaluate_model(model, feats, labels, mtype)
                all_results[q_key][seed][name] = r
                save_predictions(PRED_DIR, backbone, name, seed, dataset_name, probs, labels)

        # Q5 ablation
        all_results['q5_ablation'].setdefault(seed, {})
        m5 = models['multi_agent'][0]
        r_full, _ = evaluate_model(m5, ood_test_f, ood_test_l, 'multi_agent')
        all_results['q5_ablation'][seed]['full_system'] = r_full
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
            all_results['q5_ablation'][seed][agent_name] = {'auc': float(auc), 'accuracy': float(accuracy_score(ood_test_l, (probs >= 0.5).astype(int)))}

        print(f"  Seed {seed} done in {time.time()-t0:.1f}s")

        # Checkpoint
        ckpt = os.path.join("/workspace", f"ckpt_{backbone}_{'sub' if subsample else 'full'}_seed{seed}.json")
        with open(ckpt, 'w') as f:
            json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=str)

    # Aggregate
    aggregated = {}
    for q_name, q_results in all_results.items():
        if not q_results:
            continue
        all_model_names = set()
        for seed_data in q_results.values():
            all_model_names.update(seed_data.keys())
        all_model_names = sorted(all_model_names)

        agg_q = {}
        for model_name in all_model_names:
            agg_q[model_name] = {}
            for mk in ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'brier']:
                vals = [q_results[s][model_name][mk] for s in q_results if model_name in q_results[s] and mk in q_results[s][model_name]]
                if vals:
                    m, ci = compute_ci(vals)
                    agg_q[model_name][mk] = {'mean': m, 'ci': ci}
        aggregated[q_name] = agg_q

    suffix = '_subsample40k' if subsample else ''
    results_path = os.path.join(out_dir, f"all_results_v3{suffix}.json")
    tmp = os.path.join("/workspace", f"results_v3_{backbone}{suffix}_tmp.json")
    with open(tmp, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    import subprocess as sp
    shutil.copy(tmp, results_path)
    os.remove(tmp)
    print(f"\nResults saved to {results_path}")
    return aggregated


def run_orchestrator_variants(backbone, n_seeds):
    """Run 3 orchestrator variants."""
    feature_dim = BACKBONE_DIMS[backbone]
    out_dir = os.path.join(RESULTS_DIR, backbone)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Orchestrator Variants: {backbone} ===")

    id_train_f, id_train_l, id_train_d = load_features(backbone, "camelyon17_id_train")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features(backbone, "pcam_train")
    id_val_f, id_val_l, id_val_d = load_features(backbone, "camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features(backbone, "camelyon17_ood_test")
    pcam_test_f, pcam_test_l, pcam_test_d = load_features(backbone, "pcam_test")
    pcam_val_f, pcam_val_l, _ = load_features(backbone, "pcam_validation")

    # Load OOD val for OOD-validated variant
    ood_val_path = os.path.join(SHARED_DIR, "features_v2", backbone, "camelyon17_ood_val.npz")
    if os.path.exists(ood_val_path):
        ood_val_f, ood_val_l, _ = load_features(backbone, "camelyon17_ood_val")
    else:
        # Use pcam_val as proxy OOD val
        ood_val_f, ood_val_l = pcam_val_f, pcam_val_l

    train_f = np.concatenate([id_train_f, pcam_train_f])
    train_l = np.concatenate([id_train_l, pcam_train_l])
    train_d = np.concatenate([id_train_d, pcam_train_d])

    variants = ['uncertainty_weighted', 'ood_validated']
    all_results = {}

    for variant in variants:
        print(f"\n--- Variant: {variant} ---")
        all_results[variant] = {'q1': {}, 'q3': {}, 'q3_pcam': {}, 'q5_ablation': {}}

        for seed in range(n_seeds):
            print(f"  Seed {seed}/{n_seeds-1}...", end=' ')
            set_seeds(seed)
            t0 = time.time()

            m = MultiAgentSystemV3(input_dim=feature_dim, n_domains=N_DOMAINS, variant=variant)
            mtype = f'multi_agent_{variant}'

            if variant == 'ood_validated':
                m, auc = train_multi_agent(m, train_f, train_l, train_d, id_val_f, id_val_l,
                                           ood_val_f=ood_val_f, ood_val_l=ood_val_l)
            else:
                m, auc = train_multi_agent(m, train_f, train_l, train_d, id_val_f, id_val_l)

            # Evaluate
            for dataset_name, (feats, labels) in [('id_val', (id_val_f, id_val_l)),
                                                    ('ood_test', (ood_test_f, ood_test_l)),
                                                    ('pcam_test', (pcam_test_f, pcam_test_l))]:
                q_key = 'q1' if dataset_name == 'id_val' else ('q3' if dataset_name == 'ood_test' else 'q3_pcam')
                r, probs = evaluate_model(m, feats, labels, mtype)
                all_results[variant][q_key].setdefault(seed, {})[f'multi_agent_{variant}'] = r
                save_predictions(PRED_DIR, backbone, f'multi_agent_{variant}', seed, dataset_name, probs, labels)

            # Ablation
            all_results[variant]['q5_ablation'].setdefault(seed, {})
            r_full, _ = evaluate_model(m, ood_test_f, ood_test_l, mtype)
            all_results[variant]['q5_ablation'][seed][f'full_system_{variant}'] = r_full
            for agent_name, agent_module in [('tumor_only', m.tumor_agent), ('tissue_only', m.tissue_agent),
                                              ('uncertainty_only', m.uncertainty_agent)]:
                agent_module.eval()
                with torch.no_grad():
                    x = torch.from_numpy(ood_test_f)
                    if agent_name in ('tumor_only', 'tissue_only'):
                        logit, _ = agent_module(x)
                        probs = torch.sigmoid(logit.squeeze()).numpy()
                    else:
                        probs, _ = agent_module(x)
                        probs = probs.numpy()
                auc_val = roc_auc_score(ood_test_l, probs)
                all_results[variant]['q5_ablation'][seed][f'{agent_name}_{variant}'] = {'auc': float(auc_val)}

            print(f"val AUC={auc:.4f}, {time.time()-t0:.1f}s")

    # Aggregate and save
    aggregated = {}
    for variant in variants:
        aggregated[variant] = {}
        for q_name in ['q1', 'q3', 'q3_pcam']:
            q_results = all_results[variant][q_name]
            if not q_results:
                continue
            model_name = f'multi_agent_{variant}'
            agg = {}
            for mk in ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'brier']:
                vals = [q_results[s][model_name][mk] for s in q_results if model_name in q_results.get(s, {})]
                if vals:
                    m, ci = compute_ci(vals)
                    agg[mk] = {'mean': m, 'ci': ci}
            aggregated[variant][q_name] = {model_name: agg}

    results_path = os.path.join(out_dir, "orchestrator_variants.json")
    tmp = os.path.join("/workspace", f"orch_variants_{backbone}_tmp.json")
    with open(tmp, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    shutil.copy(tmp, results_path)
    os.remove(tmp)
    print(f"\nOrchestrator variants saved to {results_path}")
    return aggregated


def run_disentanglement_sweep(backbone, n_seeds):
    """Sweep disentanglement loss weight."""
    feature_dim = BACKBONE_DIMS[backbone]
    out_dir = os.path.join(RESULTS_DIR, backbone)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Disentanglement Sweep: {backbone} ===")

    id_train_f, id_train_l, id_train_d = load_features(backbone, "camelyon17_id_train")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features(backbone, "pcam_train")
    id_val_f, id_val_l, id_val_d = load_features(backbone, "camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features(backbone, "camelyon17_ood_test")

    train_f = np.concatenate([id_train_f, pcam_train_f])
    train_l = np.concatenate([id_train_l, pcam_train_l])
    train_d = np.concatenate([id_train_d, pcam_train_d])

    lambdas = [0.1, 0.3, 0.5, 1.0, 2.0]
    all_results = {}

    for lam in lambdas:
        print(f"\n--- Lambda = {lam} ---")
        all_results[f'lambda_{lam}'] = {'q1': {}, 'q3': {}, 'q5_disent': {}}

        for seed in range(n_seeds):
            print(f"  Seed {seed}...", end=' ')
            set_seeds(seed)
            t0 = time.time()

            m = DisentanglementModel(input_dim=feature_dim, n_domains=N_DOMAINS)
            m, auc = train_disentanglement(m, train_f, train_l, train_d, id_val_f, id_val_l, disent_lambda=lam)

            # Evaluate
            r_id, _ = evaluate_model(m, id_val_f, id_val_l, 'disent')
            r_ood, _ = evaluate_model(m, ood_test_f, ood_test_l, 'disent')
            all_results[f'lambda_{lam}']['q1'].setdefault(seed, {})[f'disent_lam{lam}'] = r_id
            all_results[f'lambda_{lam}']['q3'].setdefault(seed, {})[f'disent_lam{lam}'] = r_ood

            # Domain/label probe
            m.eval()
            with torch.no_grad():
                x = torch.from_numpy(train_f)
                z_bio, _ = m.encode(x, lambda_=0.0)
                z_bio_np = z_bio.numpy()

            n = len(train_d)
            shuffle_idx = np.random.RandomState(42).permutation(n)
            z_shuf = z_bio_np[shuffle_idx]
            d_shuf = train_d[shuffle_idx]
            l_shuf = train_l[shuffle_idx]
            split = n // 2
            probe_d = LogisticRegression(max_iter=500, C=1.0)
            probe_d.fit(z_shuf[:split], d_shuf[:split])
            dom_acc = probe_d.score(z_shuf[split:], d_shuf[split:])
            probe_l = LogisticRegression(max_iter=500, C=1.0)
            probe_l.fit(z_shuf[:split], l_shuf[:split])
            lbl_acc = probe_l.score(z_shuf[split:], l_shuf[split:])

            all_results[f'lambda_{lam}']['q5_disent'].setdefault(seed, {})[f'disent_lam{lam}'] = {
                'bio_domain_probe_acc': float(dom_acc),
                'bio_label_probe_acc': float(lbl_acc),
            }

            print(f"val={auc:.4f}, dom_probe={dom_acc:.3f}, lbl_probe={lbl_acc:.3f}, {time.time()-t0:.1f}s")

    # Aggregate
    aggregated = {}
    for lam_key, lam_results in all_results.items():
        aggregated[lam_key] = {}
        for q_name in ['q1', 'q3', 'q5_disent']:
            q_results = lam_results[q_name]
            if not q_results:
                continue
            model_name = list(q_results[list(q_results.keys())[0]].keys())[0]
            agg = {}
            for mk in q_results[list(q_results.keys())[0]][model_name].keys():
                vals = [q_results[s][model_name][mk] for s in q_results if model_name in q_results.get(s, {})]
                if vals:
                    m, ci = compute_ci(vals)
                    agg[mk] = {'mean': m, 'ci': ci}
            aggregated[lam_key][q_name] = {model_name: agg}

    results_path = os.path.join(out_dir, "disentanglement_sweep.json")
    tmp = os.path.join("/workspace", f"disent_sweep_{backbone}_tmp.json")
    with open(tmp, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    shutil.copy(tmp, results_path)
    os.remove(tmp)
    print(f"\nDisentanglement sweep saved to {results_path}")
    return aggregated


def run_mixstyle_fact(backbone, n_seeds):
    """Run MixStyle and FACT baselines."""
    feature_dim = BACKBONE_DIMS[backbone]
    out_dir = os.path.join(RESULTS_DIR, backbone)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== MixStyle + FACT: {backbone} ===")

    id_train_f, id_train_l, id_train_d = load_features(backbone, "camelyon17_id_train")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features(backbone, "pcam_train")
    id_val_f, id_val_l, id_val_d = load_features(backbone, "camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features(backbone, "camelyon17_ood_test")
    pcam_test_f, pcam_test_l, pcam_test_d = load_features(backbone, "pcam_test")

    train_f = np.concatenate([id_train_f, pcam_train_f])
    train_l = np.concatenate([id_train_l, pcam_train_l])
    train_d = np.concatenate([id_train_d, pcam_train_d])

    all_results = {'q1': {}, 'q3': {}, 'q3_pcam': {}}

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed}/{n_seeds-1} ---")
        set_seeds(seed)
        t0 = time.time()

        # MixStyle
        m_ms = MixStyleClassifier(input_dim=feature_dim, n_domains=N_DOMAINS)
        m_ms, auc_ms = train_mixstyle(m_ms, train_f, train_l, train_d, id_val_f, id_val_l)
        print(f"  mixstyle: val AUC={auc_ms:.4f}")

        # FACT
        m_fact = FACTClassifier(input_dim=feature_dim, n_domains=N_DOMAINS)
        m_fact, auc_fact = train_fact(m_fact, train_f, train_l, train_d, id_val_f, id_val_l)
        print(f"  fact: val AUC={auc_fact:.4f}")

        models = {'mixstyle': (m_ms, 'mixstyle'), 'fact': (m_fact, 'fact')}

        for dataset_name, (feats, labels) in [('id_val', (id_val_f, id_val_l)),
                                                ('ood_test', (ood_test_f, ood_test_l)),
                                                ('pcam_test', (pcam_test_f, pcam_test_l))]:
            q_key = 'q1' if dataset_name == 'id_val' else ('q3' if dataset_name == 'ood_test' else 'q3_pcam')
            all_results[q_key].setdefault(seed, {})
            for name, (model, mtype) in models.items():
                r, probs = evaluate_model(model, feats, labels, mtype)
                all_results[q_key][seed][name] = r
                save_predictions(PRED_DIR, backbone, name, seed, dataset_name, probs, labels)

        print(f"  Done in {time.time()-t0:.1f}s")

    # Aggregate
    aggregated = {}
    for q_name, q_results in all_results.items():
        if not q_results:
            continue
        all_model_names = sorted(set().union(*[set(s.keys()) for s in q_results.values()]))
        agg_q = {}
        for model_name in all_model_names:
            agg_q[model_name] = {}
            for mk in ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'brier']:
                vals = [q_results[s][model_name][mk] for s in q_results if model_name in q_results.get(s, {})]
                if vals:
                    m, ci = compute_ci(vals)
                    agg_q[model_name][mk] = {'mean': m, 'ci': ci}
        aggregated[q_name] = agg_q

    results_path = os.path.join(out_dir, "mixstyle_fact.json")
    tmp = os.path.join("/workspace", f"mixstyle_fact_{backbone}_tmp.json")
    with open(tmp, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    shutil.copy(tmp, results_path)
    os.remove(tmp)
    print(f"\nMixStyle/FACT saved to {results_path}")
    return aggregated


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['vit_tiny', 'vit_b', 'resnet18'], required=True)
    parser.add_argument('--experiment', required=True,
                        choices=['all', 'orchestrator', 'disentanglement_sweep',
                                 'mixstyle_fact', 'subsample_40k'])
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    args = parser.parse_args()

    if args.experiment == 'all':
        run_all_models(args.backbone, args.seeds)
    elif args.experiment == 'orchestrator':
        run_orchestrator_variants(args.backbone, args.seeds)
    elif args.experiment == 'disentanglement_sweep':
        run_disentanglement_sweep(args.backbone, args.seeds)
    elif args.experiment == 'mixstyle_fact':
        run_mixstyle_fact(args.backbone, args.seeds)
    elif args.experiment == 'subsample_40k':
        run_all_models(args.backbone, args.seeds, subsample=True)

if __name__ == "__main__":
    main()
