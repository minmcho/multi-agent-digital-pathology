"""
Experiment runner for multi-agent pathology AI framework.
Trains 5 models, evaluates on Camelyon17 ID/OOD + PCam, runs perturbation
and uncertainty experiments, saves all results to JSON.
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
import json, time, os, copy
from collections import defaultdict

from pathology_models import (
    BaselineClassifier, DANNModel, SSLModel, DisentanglementModel,
    MultiAgentSystem, nt_xent_loss, grl
)

# ============================================================
# Configuration
# ============================================================
FEATURE_DIR = "/mnt/shared-workspace/shared/features"
RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)

EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_DOMAINS = 4
MC_SAMPLES = 30

# ============================================================
# Data loading
# ============================================================

def load_features(name):
    """Load cached features from npz."""
    path = os.path.join(FEATURE_DIR, f"{name}.npz")
    data = np.load(path)
    features = data['features']
    if features.ndim > 2:
        features = features.squeeze()
    labels = data['labels']
    domains = data['domains']
    return features.astype(np.float32), labels.astype(np.int64), domains.astype(np.int64)

def make_loader(features, labels, domains=None, batch_size=BATCH_SIZE, shuffle=True):
    """Create DataLoader from numpy arrays."""
    feats = torch.from_numpy(features)
    lbls = torch.from_numpy(labels).float()
    if domains is not None:
        doms = torch.from_numpy(domains).long()
        ds = TensorDataset(feats, lbls, doms)
    else:
        ds = TensorDataset(feats, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# ============================================================
# Training functions
# ============================================================

def train_baseline(model, train_feats, train_labels, val_feats, val_labels):
    """Train baseline classifier."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(train_feats, train_labels, shuffle=True)
    best_val_auc = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            x, y = batch
            optimizer.zero_grad()
            logit = model(x).squeeze()
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        val_auc = quick_auc(model, val_feats, val_labels)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_auc


def train_dann(model, train_feats, train_labels, train_domains, val_feats, val_labels):
    """Train DANN model with domain adversarial loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_feats, train_labels, train_domains, shuffle=True)
    best_val_auc = 0
    best_state = None
    n_batches = len(loader)

    for epoch in range(EPOCHS):
        model.train()
        # GRL lambda schedule: 0 -> 1 over training
        p = epoch / EPOCHS
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        for i, batch in enumerate(loader):
            x, y, d = batch
            optimizer.zero_grad()
            tumor_logit, domain_logit = model(x, lambda_=lambda_)
            tumor_loss = criterion(tumor_logit.squeeze(), y)
            domain_loss = dom_criterion(domain_logit, d)
            loss = tumor_loss + 0.5 * domain_loss
            loss.backward()
            optimizer.step()

        val_auc = quick_auc_dann(model, val_feats, val_labels)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_auc


def train_ssl(model, train_feats, train_labels, train_domains, val_feats, val_labels):
    """Train SSL model: contrastive pre-training + fine-tuning."""
    # Phase 1: Contrastive pre-training (20 epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = make_loader(train_feats, train_labels, train_domains, shuffle=True)

    for epoch in range(20):
        model.train()
        for batch in loader:
            x, y, d = batch
            optimizer.zero_grad()
            # Create augmented views by adding noise
            noise = torch.randn_like(x) * 0.1
            z_i = model.project(x)
            z_j = model.project(x + noise)
            loss = nt_xent_loss(z_i, z_j, temperature=0.5)
            loss.backward()
            optimizer.step()

    # Phase 2: Fine-tune with tumor labels (30 epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0
    best_state = None

    for epoch in range(30):
        model.train()
        for batch in loader:
            x, y, d = batch
            optimizer.zero_grad()
            logit = model(x).squeeze()
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()

        val_auc = quick_auc(model, val_feats, val_labels)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_auc


def train_disentanglement(model, train_feats, train_labels, train_domains,
                           val_feats, val_labels):
    """Train monolithic disentanglement model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_feats, train_labels, train_domains, shuffle=True)
    best_val_auc = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        p = epoch / EPOCHS
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        for batch in loader:
            x, y, d = batch
            optimizer.zero_grad()
            tumor_logit, domain_logit, z_bio, z_dom = model(x, lambda_=lambda_, return_latents=True)
            tumor_loss = criterion(tumor_logit.squeeze(), y)
            domain_loss = dom_criterion(domain_logit, d)
            ortho_loss = model.orthogonality_loss(z_bio, z_dom)
            # Contrastive loss on Z_bio
            noise = torch.randn_like(x) * 0.1
            z_bio_aug, _ = model.encode(x + noise, lambda_)
            z_bio_orig = z_bio.detach()
            z1 = F.normalize(model.projector(z_bio_orig), dim=1)
            z2 = F.normalize(model.projector(z_bio_aug), dim=1)
            # Simple contrastive: maximize agreement
            ssl_loss = 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()

            loss = tumor_loss + 0.3 * domain_loss + 0.1 * ortho_loss + 0.1 * ssl_loss
            loss.backward()
            optimizer.step()

        val_auc = quick_auc_disent(model, val_feats, val_labels)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_auc


def train_multi_agent(model, train_feats, train_labels, train_domains,
                       val_feats, val_labels):
    """Train multi-agent system with separate agent objectives."""
    # Separate optimizers for different agent groups
    agent_params = model.get_agent_parameters()

    optimizer = torch.optim.Adam([
        {'params': agent_params['quality'], 'lr': LR},
        {'params': agent_params['tissue'], 'lr': LR},
        {'params': agent_params['domain'], 'lr': LR},
        {'params': agent_params['tumor'], 'lr': LR},
        {'params': agent_params['biomarker'], 'lr': LR},
        {'params': agent_params['uncertainty'], 'lr': LR},
        {'params': agent_params['orchestrator'], 'lr': LR * 0.5},
    ], weight_decay=WEIGHT_DECAY)

    criterion = nn.BCEWithLogitsLoss()
    dom_criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_feats, train_labels, train_domains, shuffle=True)
    best_val_auc = 0
    best_state = None

    # Create synthetic quality labels: samples from domain 0 = high quality,
    # samples from augmented domains with extreme params = lower quality
    train_domains_t = torch.from_numpy(train_domains)

    for epoch in range(EPOCHS):
        model.train()
        p = epoch / EPOCHS
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        for batch in loader:
            x, y, d = batch
            optimizer.zero_grad()

            # Agent 1: Quality (synthetic labels: domain 0 = quality 1, others vary)
            quality_target = (d == 0).float()
            q_score, _ = model.quality_agent(x)
            quality_loss = F.binary_cross_entropy(q_score.clamp(1e-7, 1-1e-7), quality_target)

            # Agent 2: Tissue classification (same as tumor label as proxy)
            tissue_logit, _ = model.tissue_agent(x)
            tissue_loss = criterion(tissue_logit.squeeze(), y)

            # Agent 3: Domain adaptation
            domain_logit, z_bio = model.domain_agent(x, lambda_=lambda_)
            domain_loss = dom_criterion(domain_logit, d)

            # Agent 4: Tumor detection
            tumor_logit, _ = model.tumor_agent(x)
            tumor_loss = criterion(tumor_logit.squeeze(), y)

            # Agent 5: Biomarker (regress to tumor probability as tumor burden)
            biomarker_score, _ = model.biomarker_agent(x)
            biomarker_loss = F.mse_loss(biomarker_score, y)

            # Agent 6: Uncertainty (same target as tumor, but with higher dropout)
            unc_pred, _ = model.uncertainty_agent(x)
            unc_loss = F.binary_cross_entropy(unc_pred.clamp(1e-7, 1-1e-7), y)

            # Orchestrator: fuse agent outputs
            agent_probs = [q_score.detach(), torch.sigmoid(tissue_logit).squeeze().detach(),
                          torch.sigmoid(tumor_logit).squeeze().detach(),
                          biomarker_score.detach(), unc_pred.detach(),
                          torch.sigmoid(tumor_logit).squeeze().detach()]
            fused_logit, _ = model.orchestrator(x, agent_probs)
            fused_loss = criterion(fused_logit, y)

            # Total loss with agent-specific weights
            loss = (tumor_loss + 0.3 * domain_loss + 0.2 * tissue_loss +
                    0.1 * quality_loss + 0.1 * biomarker_loss +
                    0.1 * unc_loss + 0.3 * fused_loss)

            loss.backward()
            optimizer.step()

        val_auc = quick_auc_ma(model, val_feats, val_labels)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_auc


# ============================================================
# Quick AUC evaluation (for early stopping)
# ============================================================

def quick_auc(model, feats, labels):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats)
        logits = model(x).squeeze()
        probs = torch.sigmoid(logits).numpy()
    return roc_auc_score(labels, probs)

def quick_auc_dann(model, feats, labels):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats)
        tumor_logit, _ = model(x)
        probs = torch.sigmoid(tumor_logit.squeeze()).numpy()
    return roc_auc_score(labels, probs)

def quick_auc_disent(model, feats, labels):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats)
        tumor_logit, _ = model(x)
        probs = torch.sigmoid(tumor_logit.squeeze()).numpy()
    return roc_auc_score(labels, probs)

def quick_auc_ma(model, feats, labels):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats)
        probs = model.predict(x).numpy()
    return roc_auc_score(labels, probs)


# ============================================================
# Full evaluation
# ============================================================

def evaluate_model(model, feats, labels, model_type='baseline'):
    """Full evaluation: AUC, accuracy, sensitivity, specificity, F1, Brier."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats.astype(np.float32))
        if model_type == 'baseline':
            probs = torch.sigmoid(model(x).squeeze()).numpy()
        elif model_type == 'dann':
            tumor_logit, _ = model(x)
            probs = torch.sigmoid(tumor_logit.squeeze()).numpy()
        elif model_type == 'ssl':
            probs = torch.sigmoid(model(x).squeeze()).numpy()
        elif model_type == 'disent':
            tumor_logit, _ = model(x)
            probs = torch.sigmoid(tumor_logit.squeeze()).numpy()
        elif model_type == 'multi_agent':
            probs = model.predict(x).numpy()

    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    results = {
        'auc': float(roc_auc_score(labels, probs)),
        'accuracy': float(accuracy_score(labels, preds)),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'brier': float(brier_score_loss(labels, probs)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }
    return results, probs


def evaluate_uncertainty(model, feats, labels, model_type='baseline', n_samples=MC_SAMPLES):
    """Evaluate uncertainty: MC-Dropout, risk-coverage, calibration."""
    model.eval()
    x = torch.from_numpy(feats.astype(np.float32))

    if model_type == 'multi_agent':
        mean_probs, std_probs = model.predict_with_uncertainty(x, n_samples)
    else:
        mean_probs, std_probs = model.predict_with_uncertainty(x, n_samples)

    mean_probs = mean_probs.numpy()
    std_probs = std_probs.numpy()

    # Predictive entropy
    eps = 1e-7
    entropy = -(mean_probs * np.log(mean_probs + eps) +
                (1 - mean_probs) * np.log(1 - mean_probs + eps))

    # Risk-coverage curve
    coverage_points = np.linspace(0.05, 1.0, 20)
    risk_coverage = []
    for cov in coverage_points:
        # Keep cov fraction of most confident predictions
        n_keep = int(cov * len(std_probs))
        if n_keep < 2:
            continue
        idx = np.argsort(std_probs)[:n_keep]  # least uncertain
        kept_labels = labels[idx]
        kept_probs = mean_probs[idx]
        kept_preds = (kept_probs >= 0.5).astype(int)
        risk = 1.0 - accuracy_score(kept_labels, kept_preds)
        risk_coverage.append({'coverage': float(cov), 'risk': float(risk), 'n': n_keep})

    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (mean_probs >= lo) & (mean_probs < hi)
        if mask.sum() > 0:
            bin_acc = (mean_probs[mask] >= 0.5).astype(int) == labels[mask]
            bin_conf = np.abs(mean_probs[mask] - 0.5) * 2  # confidence = |p - 0.5| * 2
            bin_acc_mean = bin_acc.mean()
            bin_conf_mean = bin_conf.mean()
            ece += np.abs(bin_conf_mean - bin_acc_mean) * mask.sum() / len(labels)

    # Selective prediction: AUC when abstaining on top-uncertain
    selective_aucs = []
    for frac_abstain in [0.0, 0.1, 0.2, 0.3, 0.5]:
        n_keep = int((1 - frac_abstain) * len(std_probs))
        if n_keep < 10:
            continue
        idx = np.argsort(std_probs)[:n_keep]
        if len(np.unique(labels[idx])) < 2:
            selective_aucs.append({'abstain_frac': float(frac_abstain), 'auc': None})
            continue
        sel_auc = roc_auc_score(labels[idx], mean_probs[idx])
        selective_aucs.append({'abstain_frac': float(frac_abstain), 'auc': float(sel_auc)})

    return {
        'mean_entropy': float(np.mean(entropy)),
        'mean_epistemic_std': float(np.mean(std_probs)),
        'ece': float(ece),
        'risk_coverage': risk_coverage,
        'selective_aucs': selective_aucs,
        'std_probs': std_probs.tolist()[:500],  # subsample for storage
        'mean_probs': mean_probs.tolist()[:500],
    }


def evaluate_disentanglement(model, feats, labels, domains, model_type='disent'):
    """Evaluate disentanglement quality: cross-covariance, linear probe."""
    model.eval()
    x = torch.from_numpy(feats.astype(np.float32))

    with torch.no_grad():
        if model_type == 'disent':
            z_bio, z_dom = model.encode(x)
        elif model_type == 'multi_agent':
            z_bio = model.domain_agent.get_bio_features(x)
            _, z_dom = model.tumor_agent(x)
        else:
            z_bio = model.encoder(x)
            z_dom = z_bio  # No disentanglement

    z_bio_np = z_bio.numpy()
    z_dom_np = z_dom.numpy()

    # Cross-covariance based orthogonality measure (dimension-agnostic)
    z_bio_c = z_bio_np - z_bio_np.mean(0, keepdims=True)
    z_dom_c = z_dom_np - z_dom_np.mean(0, keepdims=True)
    cross_cov = z_bio_c.T @ z_dom_c / z_bio_np.shape[0]
    # Normalized Frobenius norm of cross-covariance (0 = perfectly orthogonal)
    frob = np.linalg.norm(cross_cov, 'fro')
    bio_norm = np.linalg.norm(z_bio_c, 'fro')
    dom_norm = np.linalg.norm(z_dom_c, 'fro')
    normalized_cross_cov = float(frob / (bio_norm * dom_norm + 1e-8))

    # Linear probe: can domain be predicted from Z_bio?
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(domains)) > 1:
        probe = LogisticRegression(max_iter=500, C=1.0)
        probe.fit(z_bio_np, domains)
        bio_domain_acc = float(probe.score(z_bio_np, domains))
    else:
        bio_domain_acc = None

    # Linear probe: can label be predicted from Z_dom?
    if len(np.unique(labels)) > 1:
        probe2 = LogisticRegression(max_iter=500, C=1.0)
        probe2.fit(z_dom_np, labels)
        dom_label_acc = float(probe2.score(z_dom_np, labels))
    else:
        dom_label_acc = None

    return {
        'normalized_cross_covariance': normalized_cross_cov,
        'bio_domain_probe_acc': bio_domain_acc,
        'dom_label_probe_acc': dom_label_acc,
    }


# ============================================================
# Perturbation experiments (Q2)
# ============================================================

def apply_feature_perturbation(features, perturbation_type, severity=1.0):
    """Apply perturbations to features (simulating image-level corruptions)."""
    features = features.astype(np.float32)
    if perturbation_type == 'noise':
        return (features + np.random.randn(*features.shape).astype(np.float32) * 0.05 * severity).astype(np.float32)
    elif perturbation_type == 'stain_shift':
        mean_shift = np.mean(features, axis=0) * 0.1 * severity
        return (features + mean_shift).astype(np.float32)
    elif perturbation_type == 'feature_dropout':
        mask = (np.random.rand(*features.shape) > (0.1 * severity)).astype(np.float32)
        return (features * mask).astype(np.float32)
    elif perturbation_type == 'adversarial':
        return (features + np.sign(features) * 0.02 * severity).astype(np.float32)
    elif perturbation_type == 'distribution_shift':
        scale = 1.0 + 0.2 * severity
        return (features * scale + np.random.randn(*features.shape).astype(np.float32) * 0.02 * severity).astype(np.float32)
    else:
        return features.astype(np.float32)


def evaluate_perturbation_robustness(model, feats, labels, model_type='baseline'):
    """Evaluate model robustness under various perturbations."""
    base_results, base_probs = evaluate_model(model, feats, labels, model_type)
    base_auc = base_results['auc']

    perturbation_types = ['noise', 'stain_shift', 'feature_dropout', 'adversarial', 'distribution_shift']
    severities = [0.5, 1.0, 2.0]

    results = {'base_auc': base_auc, 'perturbations': {}}

    for ptype in perturbation_types:
        results['perturbations'][ptype] = []
        for sev in severities:
            perturbed = apply_feature_perturbation(feats, ptype, sev)
            r, _ = evaluate_model(model, perturbed, labels, model_type)
            delta_auc = r['auc'] - base_auc
            results['perturbations'][ptype].append({
                'severity': sev,
                'auc': r['auc'],
                'delta_auc': delta_auc,
                'accuracy': r['accuracy'],
            })

    return results


# ============================================================
# Main experiment runner
# ============================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("MULTI-AGENT PATHOLOGY AI: EXPERIMENT RUNNER")
    print("=" * 70)

    # Load all features
    print("\nLoading features...")
    id_train_f, id_train_l, id_train_d = load_features("camelyon17_id_train")
    id_val_f, id_val_l, id_val_d = load_features("camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features("camelyon17_ood_test")
    ood_val_f, ood_val_l, ood_val_d = load_features("camelyon17_ood_val")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features("pcam_train")
    pcam_test_f, pcam_test_l, pcam_test_d = load_features("pcam_test")
    pcam_val_f, pcam_val_l, pcam_val_d = load_features("pcam_validation")

    print(f"  ID Train:  {id_train_f.shape}  labels={np.bincount(id_train_l)}  domains={np.bincount(id_train_d)}")
    print(f"  ID Val:    {id_val_f.shape}  labels={np.bincount(id_val_l)}")
    print(f"  OOD Test:  {ood_test_f.shape}  labels={np.bincount(ood_test_l)}")
    print(f"  OOD Val:   {ood_val_f.shape}  labels={np.bincount(ood_val_l)}")
    print(f"  PCam Train:{pcam_train_f.shape}  labels={np.bincount(pcam_train_l)}")
    print(f"  PCam Test: {pcam_test_f.shape}  labels={np.bincount(pcam_test_l)}")
    print(f"  PCam Val:  {pcam_val_f.shape}  labels={np.bincount(pcam_val_l)}")

    all_results = {}

    # ===== Model 1: Baseline =====
    print("\n" + "=" * 50)
    print("Training Model 1: Baseline Classifier")
    print("=" * 50)
    t0 = time.time()
    model1 = BaselineClassifier(input_dim=512, hidden=256, dropout=0.1)
    model1, val_auc = train_baseline(model1, id_train_f, id_train_l, id_val_f, id_val_l)
    print(f"  Trained in {time.time()-t0:.1f}s, best val AUC={val_auc:.4f}")

    # ===== Model 2: DANN =====
    print("\n" + "=" * 50)
    print("Training Model 2: DANN")
    print("=" * 50)
    t0 = time.time()
    model2 = DANNModel(input_dim=512, hidden=256, n_domains=N_DOMAINS, dropout=0.1)
    model2, val_auc = train_dann(model2, id_train_f, id_train_l, id_train_d, id_val_f, id_val_l)
    print(f"  Trained in {time.time()-t0:.1f}s, best val AUC={val_auc:.4f}")

    # ===== Model 3: SSL =====
    print("\n" + "=" * 50)
    print("Training Model 3: SSL (Contrastive + Fine-tune)")
    print("=" * 50)
    t0 = time.time()
    model3 = SSLModel(input_dim=512, hidden=256, proj_dim=128, dropout=0.1)
    model3, val_auc = train_ssl(model3, id_train_f, id_train_l, id_train_d, id_val_f, id_val_l)
    print(f"  Trained in {time.time()-t0:.1f}s, best val AUC={val_auc:.4f}")

    # ===== Model 4: Disentanglement (Monolithic) =====
    print("\n" + "=" * 50)
    print("Training Model 4: Disentanglement (Monolithic)")
    print("=" * 50)
    t0 = time.time()
    model4 = DisentanglementModel(input_dim=512, bio_dim=256, dom_dim=128, n_domains=N_DOMAINS, dropout=0.1)
    model4, val_auc = train_disentanglement(model4, id_train_f, id_train_l, id_train_d, id_val_f, id_val_l)
    print(f"  Trained in {time.time()-t0:.1f}s, best val AUC={val_auc:.4f}")

    # ===== Model 5: Multi-Agent System =====
    print("\n" + "=" * 50)
    print("Training Model 5: Multi-Agent System")
    print("=" * 50)
    t0 = time.time()
    model5 = MultiAgentSystem(input_dim=512, n_domains=N_DOMAINS, dropout=0.1)
    model5, val_auc = train_multi_agent(model5, id_train_f, id_train_l, id_train_d, id_val_f, id_val_l)
    print(f"  Trained in {time.time()-t0:.1f}s, best val AUC={val_auc:.4f}")

    models = {
        'baseline': (model1, 'baseline'),
        'dann': (model2, 'dann'),
        'ssl': (model3, 'ssl'),
        'disentanglement': (model4, 'disent'),
        'multi_agent': (model5, 'multi_agent'),
    }

    # Save trained models so evaluation can resume if needed
    torch.save({
        'baseline': model1.state_dict(),
        'dann': model2.state_dict(),
        'ssl': model3.state_dict(),
        'disentanglement': model4.state_dict(),
        'multi_agent': model5.state_dict(),
    }, os.path.join(RESULTS_DIR, "model_states.pt"))
    print("  [Saved trained model states]")

    # ===== Q1: Diagnostic Validity (Camelyon17 ID Val) =====
    print("\n" + "=" * 50)
    print("Q1: Diagnostic Validity (Camelyon17 ID Val)")
    print("=" * 50)
    q1_results = {}
    for name, (model, mtype) in models.items():
        r, _ = evaluate_model(model, id_val_f, id_val_l, mtype)
        q1_results[name] = r
        print(f"  {name:20s}: AUC={r['auc']:.4f} Acc={r['accuracy']:.4f} "
              f"Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f} F1={r['f1']:.4f}")
    all_results['q1_diagnostic_validity'] = q1_results

    # ===== Q3: External Generalization (Camelyon17 OOD Test) =====
    print("\n" + "=" * 50)
    print("Q3: External Generalization (Camelyon17 OOD Test)")
    print("=" * 50)
    q3_results = {}
    for name, (model, mtype) in models.items():
        r, _ = evaluate_model(model, ood_test_f, ood_test_l, mtype)
        q3_results[name] = r
        print(f"  {name:20s}: AUC={r['auc']:.4f} Acc={r['accuracy']:.4f} "
              f"Sens={r['sensitivity']:.4f} Spec={r['specificity']:.4f} F1={r['f1']:.4f}")
    all_results['q3_external_generalization'] = q3_results

    # Also evaluate on PCam (external dataset)
    print("\n  PCam (external dataset):")
    pcam_results = {}
    for name, (model, mtype) in models.items():
        r, _ = evaluate_model(model, pcam_test_f, pcam_test_l, mtype)
        pcam_results[name] = r
        print(f"  {name:20s}: AUC={r['auc']:.4f} Acc={r['accuracy']:.4f}")
    all_results['q3_pcam_external'] = pcam_results

    # ===== Q2: Technical Robustness (Perturbations on PCam) =====
    print("\n" + "=" * 50)
    print("Q2: Technical Robustness (Perturbations on PCam Test)")
    print("=" * 50)
    q2_results = {}
    for name, (model, mtype) in models.items():
        r = evaluate_perturbation_robustness(model, pcam_test_f, pcam_test_l, mtype)
        q2_results[name] = r
        # Summary: mean delta AUC across perturbations
        all_deltas = []
        for ptype, perts in r['perturbations'].items():
            for p in perts:
                all_deltas.append(abs(p['delta_auc']))
        mean_delta = np.mean(all_deltas) if all_deltas else 0
        print(f"  {name:20s}: base_AUC={r['base_auc']:.4f} mean|ΔAUC|={mean_delta:.4f}")
    all_results['q2_robustness'] = q2_results

    # ===== Q4: Uncertainty Reliability =====
    print("\n" + "=" * 50)
    print("Q4: Uncertainty Reliability (Camelyon17 OOD Test)")
    print("=" * 50)
    q4_results = {}
    for name, (model, mtype) in models.items():
        r = evaluate_uncertainty(model, ood_test_f, ood_test_l, mtype)
        q4_results[name] = r
        print(f"  {name:20s}: ECE={r['ece']:.4f} entropy={r['mean_entropy']:.4f} "
              f"epistemic={r['mean_epistemic_std']:.4f}")
    all_results['q4_uncertainty'] = q4_results

    # Also on ID val
    print("\n  ID Val uncertainty:")
    q4_id_results = {}
    for name, (model, mtype) in models.items():
        r = evaluate_uncertainty(model, id_val_f, id_val_l, mtype)
        q4_id_results[name] = r
        print(f"  {name:20s}: ECE={r['ece']:.4f} entropy={r['mean_entropy']:.4f}")
    all_results['q4_uncertainty_id'] = q4_id_results

    # ===== Q5: System-Level (Multi-Agent vs Monolithic) =====
    print("\n" + "=" * 50)
    print("Q5: System-Level Comparison (Multi-Agent vs Monolithic)")
    print("=" * 50)

    # Disentanglement analysis (use OOD test which has domain shift)
    disent_results = {}
    for name, (model, mtype) in [('baseline', (model1, 'baseline')),
                                  ('dann', (model2, 'dann')),
                                  ('disentanglement', (model4, 'disent')),
                                  ('multi_agent', (model5, 'multi_agent'))]:
        r = evaluate_disentanglement(model, ood_test_f, ood_test_l, ood_test_d, mtype)
        disent_results[name] = r
        print(f"  {name:20s}: cross_cov={r['normalized_cross_covariance']:.4f} "
              f"bio→dom_acc={r['bio_domain_probe_acc']} dom→label_acc={r['dom_label_probe_acc']}")
    all_results['q5_disentanglement'] = disent_results

    # Agent contribution analysis (ablation: remove each agent)
    print("\n  Agent ablation (multi-agent system):")
    ablation_results = {}
    model5_full, _ = evaluate_model(model5, ood_test_f, ood_test_l, 'multi_agent')
    ablation_results['full_system'] = model5_full

    # Test with individual agents
    for agent_name, agent_module in [('tumor_only', model5.tumor_agent),
                                      ('tissue_only', model5.tissue_agent),
                                      ('uncertainty_only', model5.uncertainty_agent)]:
        # Evaluate single agent
        agent_module.eval()
        with torch.no_grad():
            x = torch.from_numpy(ood_test_f)
            if agent_name == 'tumor_only':
                logit, _ = agent_module(x)
                probs = torch.sigmoid(logit.squeeze()).numpy()
            elif agent_name == 'tissue_only':
                logit, _ = agent_module(x)
                probs = torch.sigmoid(logit.squeeze()).numpy()
            elif agent_name == 'uncertainty_only':
                probs, _ = agent_module(x)
                probs = probs.numpy()

        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(ood_test_l, probs)
        acc = accuracy_score(ood_test_l, preds)
        ablation_results[agent_name] = {'auc': float(auc), 'accuracy': float(acc)}
        print(f"  {agent_name:20s}: AUC={auc:.4f} Acc={acc:.4f}")

    all_results['q5_ablation'] = ablation_results

    # Agent weights from orchestrator
    weights = model5.orchestrator.agent_weights.detach().numpy()
    agent_names = ['Quality', 'Tissue', 'Tumor', 'Biomarker', 'Uncertainty', 'Tumor_dup']
    all_results['q5_agent_weights'] = {
        'agents': agent_names,
        'weights': weights.tolist(),
    }
    print(f"\n  Orchestrator agent weights: {dict(zip(agent_names, weights))}")

    # ===== Q6: Workflow Efficiency =====
    print("\n" + "=" * 50)
    print("Q6: Workflow Efficiency")
    print("=" * 50)

    # Measure inference time
    q6_results = {}
    for name, (model, mtype) in models.items():
        model.eval()
        x = torch.from_numpy(ood_test_f[:1000])
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                if mtype == 'baseline':
                    _ = model(x)
                elif mtype == 'dann':
                    _ = model(x)
                elif mtype == 'ssl':
                    _ = model(x)
                elif mtype == 'disent':
                    _ = model(x)
                elif mtype == 'multi_agent':
                    _ = model(x)
        elapsed = (time.time() - t0) / 10
        q6_results[name] = {'inference_time_ms': float(elapsed * 1000),
                            'n_samples': 1000}
        print(f"  {name:20s}: {elapsed*1000:.1f} ms / 1000 samples")

    # Multi-agent: count how many cases flagged for review
    model5.eval()
    with torch.no_grad():
        x = torch.from_numpy(ood_test_f)
        out = model5(x)
    n_flagged = out['flag_for_review'].sum().item()
    q6_results['multi_agent']['n_flagged_for_review'] = int(n_flagged)
    q6_results['multi_agent']['flag_rate'] = float(n_flagged / len(ood_test_l))
    print(f"  Multi-agent: {n_flagged}/{len(ood_test_l)} flagged for review ({n_flagged/len(ood_test_l)*100:.1f}%)")

    all_results['q6_efficiency'] = q6_results

    # ===== Training info =====
    all_results['training_info'] = {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'n_domains': N_DOMAINS,
        'mc_samples': MC_SAMPLES,
        'feature_dim': 512,
        'backbone': 'ResNet-18 (ImageNet pretrained, frozen)',
        'id_train_size': int(len(id_train_l)),
        'id_val_size': int(len(id_val_l)),
        'ood_test_size': int(len(ood_test_l)),
        'ood_val_size': int(len(ood_val_l)),
        'pcam_train_size': int(len(pcam_train_l)),
        'pcam_test_size': int(len(pcam_test_l)),
        'pcam_val_size': int(len(pcam_val_l)),
    }

    # ===== Save results =====
    results_path = os.path.join(RESULTS_DIR, "all_experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"All results saved to {results_path}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*70}")

    # Also save model states
    torch.save({
        'baseline': model1.state_dict(),
        'dann': model2.state_dict(),
        'ssl': model3.state_dict(),
        'disentanglement': model4.state_dict(),
        'multi_agent': model5.state_dict(),
    }, os.path.join(RESULTS_DIR, "model_states.pt"))
    print(f"Model states saved to {RESULTS_DIR}/model_states.pt")

    return all_results


if __name__ == "__main__":
    results = main()
