"""
Kather2016 (NCT-CRC-HE) colorectal cancer dataset:
  1. Extract TUM/NORM images from CRC-VAL-HE-7K parquet files
  2. Extract features using ViT-Tiny (phikon-distil) and ResNet-18
  3. Train 5 models on Camelyon17+PCam, evaluate on Kather2016 as cross-cancer OOD

Usage:
  python run_kather2016.py --backbone vit_tiny --seeds 10
  python run_kather2016.py --backbone resnet18 --seeds 10
"""
import pyarrow.parquet as pq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from skimage.color import rgb2hed, hed2rgb
from PIL import Image
import io, os, time, argparse, warnings, collections, copy, json, shutil
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/mnt/shared-workspace/shared')
sys.path.insert(0, '/workspace')
from pathology_models import (
    BaselineClassifier, DANNModel, SSLModel, DisentanglementModel,
    MultiAgentSystem, nt_xent_loss, grl
)
from delong_test import delong_paired_comparison, bonferroni_correct

# ============================================================
# Configuration
# ============================================================
SHARED_DIR = "/mnt/shared-workspace/shared"
RESULTS_DIR = "/mnt/shared-workspace/shared/results_v3"
PRED_DIR = "/mnt/shared-workspace/shared/predictions_v3"
KATHER_FEATURE_DIR = os.path.join(SHARED_DIR, "features_v2")

BACKBONE_CONFIGS = {
    'vit_tiny': {
        'model_id': '1aurent/phikon-distil-vit-tiny-patch16-224-kather2016',
        'feature_dim': 192, 'batch_size': 64,
    },
    'resnet18': {
        'model_id': None,  # torchvision pretrained
        'feature_dim': 512, 'batch_size': 128,
    },
}

EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_DOMAINS = 4
N_SEEDS = 10

# Kather2016 label mapping: 6=NORM, 8=TUM
KATHER_NORM_LABEL = 6
KATHER_TUM_LABEL = 8

# ============================================================
# Feature extraction
# ============================================================

def decode_image(image_dict):
    """Decode image from parquet dict format."""
    img_bytes = image_dict['bytes']
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def extract_kather_images():
    """Extract TUM and NORM images from CRC-VAL-HE-7K parquet files."""
    parquet_dir = '/workspace/kather2016_data'
    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])

    tum_images, tum_labels = [], []
    norm_images, norm_labels = [], []

    for fname in files:
        fpath = os.path.join(parquet_dir, fname)
        t = pq.read_table(fpath, columns=['image', 'label'])
        images = t.column('image').to_pylist()
        labels = t.column('label').to_pylist()

        for img_dict, label in zip(images, labels):
            if label == KATHER_TUM_LABEL:
                tum_images.append(img_dict)
                tum_labels.append(1)
            elif label == KATHER_NORM_LABEL:
                norm_images.append(img_dict)
                norm_labels.append(0)

    print(f"  TUM images: {len(tum_images)}")
    print(f"  NORM images: {len(norm_images)}")
    return tum_images + norm_images, tum_labels + norm_labels


def extract_features_vit_tiny(images, labels):
    """Extract features using ViT-Tiny (phikon-distil)."""
    from transformers import AutoModel, AutoImageProcessor

    config = BACKBONE_CONFIGS['vit_tiny']
    model_id = config['model_id']
    batch_size = config['batch_size']

    print(f"  Loading model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    all_features = []
    all_labels = []

    n = len(images)
    for i in range(0, n, batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # Decode and preprocess
        pil_imgs = [decode_image(img_dict) for img_dict in batch_imgs]
        inputs = processor(images=pil_imgs, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token (pooler_output or last_hidden_state[:, 0])
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]

        all_features.append(feats.numpy())
        all_labels.extend(batch_labels)

        if (i // batch_size) % 10 == 0:
            print(f"    Processed {min(i+batch_size, n)}/{n} images")

    features = np.concatenate(all_features)
    labels_arr = np.array(all_labels)
    return features.astype(np.float32), labels_arr.astype(np.int64)


def extract_features_resnet18(images, labels):
    """Extract features using ResNet-18 (ImageNet pretrained)."""
    from torchvision.models import resnet18, ResNet18_Weights
    import torchvision.transforms as transforms

    config = BACKBONE_CONFIGS['resnet18']
    batch_size = config['batch_size']

    print(f"  Loading ResNet-18 (ImageNet pretrained)")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    # Remove classification head
    model = nn.Sequential(*list(model.children())[:-1])  # Remove fc layer
    model.eval()

    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_features = []
    all_labels = []

    n = len(images)
    for i in range(0, n, batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # Decode and preprocess
        pil_imgs = [decode_image(img_dict) for img_dict in batch_imgs]
        tensors = torch.stack([transform(img) for img in pil_imgs])

        with torch.no_grad():
            feats = model(tensors)
            feats = feats.squeeze(-1).squeeze(-1)  # Remove spatial dims

        all_features.append(feats.numpy())
        all_labels.extend(batch_labels)

        if (i // batch_size) % 5 == 0:
            print(f"    Processed {min(i+batch_size, n)}/{n} images")

    features = np.concatenate(all_features)
    labels_arr = np.array(all_labels)
    return features.astype(np.float32), labels_arr.astype(np.int64)


def save_kather_features(backbone):
    """Extract and save Kather2016 features for a given backbone."""
    feature_dir = os.path.join(KATHER_FEATURE_DIR, backbone)
    os.makedirs(feature_dir, exist_ok=True)

    out_path = os.path.join(feature_dir, "kather2016_test.npz")
    if os.path.exists(out_path):
        data = np.load(out_path)
        print(f"  Kather2016 features already exist: {data['features'].shape}")
        return out_path

    print(f"\n=== Extracting Kather2016 features: {backbone} ===")
    images, labels = extract_kather_images()
    print(f"  Total binary images: {len(images)}")

    if backbone == 'vit_tiny':
        features, labels = extract_features_vit_tiny(images, labels)
    elif backbone == 'resnet18':
        features, labels = extract_features_resnet18(images, labels)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Create dummy domains (all 0, since Kather2016 is a single domain)
    domains = np.zeros(len(labels), dtype=np.int64)

    # Save to /workspace first, then copy to shared
    tmp_path = os.path.join('/workspace', f'kather2016_{backbone}.npz')
    np.savez(tmp_path, features=features, labels=labels, domains=domains)
    shutil.copy(tmp_path, out_path)
    os.remove(tmp_path)

    print(f"  Saved: {out_path} (features: {features.shape})")
    return out_path


# ============================================================
# Data loading (same as v3)
# ============================================================

def load_features(backbone, name):
    path = os.path.join(SHARED_DIR, "features_v2", backbone, f"{name}.npz")
    if 'train' in name and not os.path.exists(path):
        import glob
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
# Training functions (same as v3)
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


def train_disentanglement(model, train_f, train_l, train_d, val_f, val_l):
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
            loss = tumor_loss + 0.3 * domain_loss + 0.1 * ortho_loss + 0.1 * ssl_loss
            loss.backward()
            optimizer.step()
        val_auc = quick_auc(model, val_f, val_l, 'disent')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc


def train_multi_agent(model, train_f, train_l, train_d, val_f, val_l):
    agent_params = model.get_agent_parameters()
    optimizer = torch.optim.Adam([
        {'params': agent_params[k], 'lr': LR if k != 'orchestrator' else LR * 0.5}
        for k in agent_params
    ], weight_decay=WEIGHT_DECAY)
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
        val_auc = quick_auc(model, val_f, val_l, 'multi_agent')
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc


# ============================================================
# Evaluation
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
    os.makedirs(os.path.join(pred_dir, backbone), exist_ok=True)
    path = os.path.join(pred_dir, backbone, f"{model_name}_seed{seed}_{dataset_name}.npz")
    tmp = os.path.join("/workspace", "pred_tmp", f"{model_name}_seed{seed}_{dataset_name}.npz")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    np.savez(tmp, probs=probs, labels=labels)
    shutil.copy(tmp, path)


# ============================================================
# Main experiment
# ============================================================

def run_kather2016_experiment(backbone, n_seeds):
    """Train 5 models on Camelyon17+PCam, evaluate on Kather2016 as cross-cancer OOD."""
    feature_dim = BACKBONE_CONFIGS[backbone]['feature_dim']
    out_dir = os.path.join(RESULTS_DIR, backbone)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Kather2016 Experiment: {backbone} (dim={feature_dim}, seeds={n_seeds}) ===")

    # Load Camelyon17+PCam features for training
    id_train_f, id_train_l, id_train_d = load_features(backbone, "camelyon17_id_train")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features(backbone, "pcam_train")
    id_val_f, id_val_l, id_val_d = load_features(backbone, "camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features(backbone, "camelyon17_ood_test")
    pcam_test_f, pcam_test_l, pcam_test_d = load_features(backbone, "pcam_test")

    # Load Kather2016 features
    kather_f, kather_l, kather_d = load_features(backbone, "kather2016_test")
    print(f"  Kather2016 test: {kather_f.shape}, labels={collections.Counter(kather_l.tolist())}")

    train_f = np.concatenate([id_train_f, pcam_train_f])
    train_l = np.concatenate([id_train_l, pcam_train_l])
    train_d = np.concatenate([id_train_d, pcam_train_d])
    print(f"  Train: {train_f.shape}, domains={collections.Counter(train_d.tolist())}")

    all_results = {'q1': {}, 'q3': {}, 'q3_pcam': {}, 'q3_kather': {}, 'q5_ablation': {}}

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

        # Evaluate on all datasets including Kather2016
        for dataset_name, (feats, labels) in [('id_val', (id_val_f, id_val_l)),
                                                ('ood_test', (ood_test_f, ood_test_l)),
                                                ('pcam_test', (pcam_test_f, pcam_test_l)),
                                                ('kather_test', (kather_f, kather_l))]:
            q_key = {'id_val': 'q1', 'ood_test': 'q3', 'pcam_test': 'q3_pcam',
                     'kather_test': 'q3_kather'}[dataset_name]
            all_results[q_key].setdefault(seed, {})
            for name, (model, mtype) in models.items():
                r, probs = evaluate_model(model, feats, labels, mtype)
                all_results[q_key][seed][name] = r
                save_predictions(PRED_DIR, backbone, name, seed, dataset_name, probs, labels)

        # Q5 ablation on Kather2016
        all_results['q5_ablation'].setdefault(seed, {})
        m5 = models['multi_agent'][0]
        r_full, _ = evaluate_model(m5, kather_f, kather_l, 'multi_agent')
        all_results['q5_ablation'][seed]['full_system'] = r_full
        for agent_name, agent_module in [('tumor_only', m5.tumor_agent), ('tissue_only', m5.tissue_agent),
                                          ('uncertainty_only', m5.uncertainty_agent)]:
            agent_module.eval()
            with torch.no_grad():
                x = torch.from_numpy(kather_f)
                if agent_name in ('tumor_only', 'tissue_only'):
                    logit, _ = agent_module(x)
                    probs = torch.sigmoid(logit.squeeze()).numpy()
                else:
                    probs, _ = agent_module(x)
                    probs = probs.numpy()
            auc_val = roc_auc_score(kather_l, probs)
            all_results['q5_ablation'][seed][agent_name] = {'auc': float(auc_val)}

        print(f"  Kather2016 AUCs: " + ", ".join(
            f"{n}={all_results['q3_kather'][seed][n]['auc']:.4f}" for n in ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']))
        print(f"  Seed {seed} done in {time.time()-t0:.1f}s")

        # Checkpoint
        ckpt = os.path.join("/workspace", f"ckpt_kather_{backbone}_seed{seed}.json")
        with open(ckpt, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

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
                vals = [q_results[s][model_name][mk] for s in q_results
                        if model_name in q_results[s] and mk in q_results[s][model_name]]
                if vals:
                    m, ci = compute_ci(vals)
                    agg_q[model_name][mk] = {'mean': m, 'ci': ci}
        aggregated[q_name] = agg_q

    results_path = os.path.join(out_dir, "kather2016_results.json")
    tmp = os.path.join("/workspace", f"kather2016_{backbone}_tmp.json")
    with open(tmp, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    shutil.copy(tmp, results_path)
    os.remove(tmp)
    print(f"\nKather2016 results saved to {results_path}")
    return aggregated


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['vit_tiny', 'resnet18'], required=True)
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--extract_only', action='store_true', help='Only extract features, skip training')
    args = parser.parse_args()

    # Step 1: Extract Kather2016 features
    save_kather_features(args.backbone)

    if args.extract_only:
        print("Feature extraction complete. Exiting.")
        return

    # Step 2: Run experiment
    run_kather2016_experiment(args.backbone, args.seeds)


if __name__ == "__main__":
    main()
