"""
Frozen-Feature Experiments on New Foundation Model Backbones

Runs all 5 models (baseline, DANN, disentanglement, multi_agent, mixstyle)
on UNI, Virchow, and CONCH frozen features. 10 seeds each.
Runs on CPU — no GPU needed (operates on pre-extracted features).

Usage:
  python run_frozen_features.py --backbone uni
  python run_frozen_features.py --backbone virchow
  python run_frozen_features.py --backbone conch
  python run_frozen_features.py --backbone all

Requires:
  - Features extracted by rtx5070_feat_extraction.py
  - Existing pipeline scripts in scripts/ directory
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import pearsonr

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathology_models import (
    BaselineClassifier, DANNModel, SSLModel, DisentanglementModel,
    MultiAgentSystem, nt_xent_loss, grl
)
from pathology_models_v3 import (
    MultiAgentSystemV3, MixStyleClassifier, FACTClassifier, DisentanglementModelV3
)

# ============================================================
# Config
# ============================================================
FEATURES_DIR = os.environ.get('FEATURES_DIR', './features')
RESULTS_DIR = os.environ.get('FROZEN_RESULTS_DIR', './frozen_results')

BACKBONE_DIMS = {
    'resnet18': 512, 'vit_tiny': 192, 'vit_b': 768,
    'uni': 1024, 'virchow': 1024, 'conch': 512,
}

MODELS = ['baseline', 'dann', 'disentanglement', 'multi_agent', 'mixstyle']
SEEDS = 10

# ============================================================
# Data Loading
# ============================================================
def load_features(backbone, split_name):
    """Load pre-extracted features for a backbone and split."""
    path = os.path.join(FEATURES_DIR, backbone, f"{split_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data['features'], data['labels']

def prepare_data(backbone):
    """Load and prepare train/val/test data for a backbone."""
    train_X, train_y = load_features(backbone, 'camelyon17_id_train')
    val_X, val_y = load_features(backbone, 'camelyon17_id_val')
    ood_X, ood_y = load_features(backbone, 'camelyon17_ood_test')
    
    # Standardize
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0) + 1e-6
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std
    ood_X = (ood_X - mean) / std
    
    return (train_X, train_y, val_X, val_y, ood_X, ood_y)

# ============================================================
# Model Creation
# ============================================================
def create_model(model_name, input_dim, seed):
    """Create a model instance."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if model_name == 'baseline':
        return BaselineClassifier(input_dim=input_dim, hidden=256, dropout=0.1)
    elif model_name == 'dann':
        return DANNModel(input_dim=input_dim, n_domains=4, hidden=256, dropout=0.1)
    elif model_name == 'disentanglement':
        return DisentanglementModelV3(input_dim=input_dim, bio_dim=256, dom_dim=128,
                                       n_domains=4, dropout=0.1, disentanglement_lambda=1.0)
    elif model_name == 'multi_agent':
        return MultiAgentSystemV3(input_dim=input_dim, n_domains=4, dropout=0.1,
                                   variant='uncertainty_weighted', disentanglement_lambda=1.0)
    elif model_name == 'mixstyle':
        return MixStyleClassifier(input_dim=input_dim, hidden=256, dropout=0.1,
                                   n_domains=4, mix_prob=0.5, alpha=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ============================================================
# Training
# ============================================================
def train_model(model, train_X, train_y, val_X, val_y, input_dim,
                model_name, epochs=50, lr=1e-3, batch_size=256, device='cpu'):
    """Train a model on frozen features."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_X_t = torch.from_numpy(train_X).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    val_X_t = torch.from_numpy(val_X).float().to(device)
    val_y_t = torch.from_numpy(val_y).float().to(device)
    
    n = len(train_X_t)
    best_val_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x, y = train_X_t[idx], train_y_t[idx]
            optimizer.zero_grad()
            
            if model_name in ['dann', 'disentanglement', 'multi_agent']:
                # These models need domain labels — use dummy domains for frozen features
                domains = torch.zeros(len(idx), dtype=torch.long, device=device)
                out = model(x, lambda_=1.0) if model_name == 'multi_agent' else model(x, domains)
                if model_name == 'multi_agent':
                    logits = out['fused_logit']
                elif model_name == 'dann':
                    logits = out[0] if isinstance(out, tuple) else out
                elif model_name == 'disentanglement':
                    logits = out[0] if isinstance(out, tuple) else out
            elif model_name == 'mixstyle':
                domains = torch.zeros(len(idx), dtype=torch.long, device=device)
                logits = model(x, domains)
            else:
                logits = model(x)
            
            logits = logits.squeeze()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        with torch.no_grad():
            if model_name == 'multi_agent':
                val_logits = model(val_X_t)['fused_logit']
            elif model_name in ['dann', 'disentanglement']:
                val_out = model(val_X_t, torch.zeros(len(val_X_t), dtype=torch.long, device=device))
                val_logits = val_out[0] if isinstance(val_out, tuple) else val_out
            elif model_name == 'mixstyle':
                val_logits = model(val_X_t, torch.zeros(len(val_X_t), dtype=torch.long, device=device))
            else:
                val_logits = model(val_X_t)
            
            val_preds = torch.sigmoid(val_logits.squeeze()).cpu().numpy()
            val_auc = roc_auc_score(val_y, val_preds) if len(set(val_y.tolist())) > 1 else 0
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_auc

# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model, X, y, model_name, device='cpu'):
    """Evaluate model on test data."""
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    
    with torch.no_grad():
        if model_name == 'multi_agent':
            logits = model(X_t)['fused_logit']
        elif model_name in ['dann', 'disentanglement']:
            out = model(X_t, torch.zeros(len(X_t), dtype=torch.long, device=device))
            logits = out[0] if isinstance(out, tuple) else out
        elif model_name == 'mixstyle':
            logits = model(X_t, torch.zeros(len(X_t), dtype=torch.long, device=device))
        else:
            logits = model(X_t)
        
        preds = torch.sigmoid(logits.squeeze()).cpu().numpy()
    
    auc = roc_auc_score(y, preds) if len(set(y.tolist())) > 1 else 0
    acc = accuracy_score(y, (preds > 0.5).astype(int))
    f1 = f1_score(y, (preds > 0.5).astype(int), zero_division=0)
    return {'auc': float(auc), 'accuracy': float(acc), 'f1': float(f1)}

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=list(BACKBONE_DIMS.keys()) + ['all'], default='all')
    parser.add_argument('--seeds', type=int, default=SEEDS)
    parser.add_argument('--features_dir', default=FEATURES_DIR)
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    backbones = list(BACKBONE_DIMS.keys()) if args.backbone == 'all' else [args.backbone]
    # Filter to only new backbones (existing ones already have results)
    new_backbones = [b for b in backbones if b in ['uni', 'virchow', 'conch']]
    if not new_backbones:
        print("No new backbones to process. Use --backbone uni/virchow/conch/all")
        return
    
    os.makedirs(args.results_dir, exist_ok=True)
    all_results = {}
    
    for bb_name in new_backbones:
        print(f"\n{'='*60}")
        print(f"Backbone: {bb_name.upper()} (dim={BACKBONE_DIMS[bb_name]})")
        print(f"{'='*60}")
        
        try:
            train_X, train_y, val_X, val_y, ood_X, ood_y = prepare_data(bb_name)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue
        
        input_dim = BACKBONE_DIMS[bb_name]
        
        for model_name in MODELS:
            print(f"\n  Model: {model_name}")
            seed_results = []
            
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = create_model(model_name, input_dim, seed)
                
                t0 = time.time()
                model, best_val = train_model(
                    model, train_X, train_y, val_X, val_y, input_dim, model_name, device=device)
                train_time = time.time() - t0
                
                val_metrics = evaluate_model(model, val_X, val_y, model_name, device)
                ood_metrics = evaluate_model(model, ood_X, ood_y, model_name, device)
                
                result = {
                    'seed': seed,
                    'val_auc': val_metrics['auc'],
                    'val_acc': val_metrics['accuracy'],
                    'ood_auc': ood_metrics['auc'],
                    'ood_acc': ood_metrics['accuracy'],
                    'ood_f1': ood_metrics['f1'],
                    'gen_gap': val_metrics['auc'] - ood_metrics['auc'],
                    'train_time': train_time,
                }
                seed_results.append(result)
                print(f"    Seed {seed}: val_auc={val_metrics['auc']:.3f} ood_auc={ood_metrics['auc']:.3f} "
                      f"gap={val_metrics['auc']-ood_metrics['auc']:.3f} ({train_time:.1f}s)")
                
                del model
            
            # Aggregate
            val_aucs = [r['val_auc'] for r in seed_results]
            ood_aucs = [r['ood_auc'] for r in seed_results]
            gaps = [r['gen_gap'] for r in seed_results]
            
            all_results[f"{bb_name}_{model_name}"] = {
                'val_auc_mean': float(np.mean(val_aucs)),
                'val_auc_std': float(np.std(val_aucs)),
                'val_auc_ci': float(1.96 * np.std(val_aucs) / np.sqrt(len(val_aucs))),
                'ood_auc_mean': float(np.mean(ood_aucs)),
                'ood_auc_std': float(np.std(ood_aucs)),
                'ood_auc_ci': float(1.96 * np.std(ood_aucs) / np.sqrt(len(ood_aucs))),
                'gen_gap_mean': float(np.mean(gaps)),
                'gen_gap_std': float(np.std(gaps)),
                'n_seeds': len(seed_results),
                'seed_results': seed_results,
            }
    
    # Save
    results_path = os.path.join(args.results_dir, 'frozen_feature_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
