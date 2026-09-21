"""
WSI-Level MIL Pipeline: ABMIL + CLAM on WILDS CAMELYON17

Runs on CPU (no GPU needed — operates on pre-extracted features).
Requires WILDS MIL features extracted by rtx5070_feat_extraction.py.

Usage:
  python mil_pipeline.py --backbone vit_b --model abmil --seeds 10
  python mil_pipeline.py --backbone vit_b --model clam --seeds 10
  python mil_pipeline.py --backbone all --model all --seeds 10

Task formulation:
  All 50 WILDS CAMELYON17 slides contain tumor patches, so standard binary
  MIL (tumor vs normal slide) is not possible. Instead, we use MIL attention
  to make patch-level predictions within each slide, then aggregate to a
  slide-level tumor fraction score. Evaluation:
  - Patch-level AUC (within test slides)
  - Slide-level tumor fraction correlation (Pearson, Spearman)
  - OOD generalization gap (train vs test center)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

FEATURES_DIR = os.environ.get('MIL_FEATURES_DIR', './features')
RESULTS_DIR = os.environ.get('MIL_RESULTS_DIR', './mil_results')

BACKBONES = ['resnet18', 'vit_tiny', 'vit_b', 'uni', 'virchow', 'conch']
MODELS = ['abmil', 'clam']

# Slide-level splits (WILDS protocol: center-based OOD)
TRAIN_CENTERS = [0, 1, 2]
VAL_CENTER = 4
TEST_CENTER = 3

# Training hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 1  # One slide at a time (variable bag size)


# ============================================================
# MIL Dataset
# ============================================================

class MILDataset(Dataset):
    """Dataset of slide-level MIL bags from pre-extracted features."""
    
    def __init__(self, features_dir, backbone, centers, max_patches=None):
        self.features_dir = os.path.join(features_dir, backbone, 'mil')
        self.centers = centers
        self.max_patches = max_patches
        self.slides = []
        
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"MIL features not found at {self.features_dir}")
        
        # Find all slide files matching the centers
        for fname in sorted(os.listdir(self.features_dir)):
            if not fname.startswith('slide_') or not fname.endswith('.npz'):
                continue
            fpath = os.path.join(self.features_dir, fname)
            data = np.load(fpath, allow_pickle=True)
            center = int(data['center'])
            if center in centers:
                self.slides.append({
                    'path': fpath,
                    'slide_id': int(data['slide_id']),
                    'center': center,
                    'tumor_fraction': float(data['tumor_fraction']),
                    'n_patches': len(data['patch_labels']),
                })
        
        print(f"  MILDataset: {len(self.slides)} slides from centers {centers}")
    
    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, idx):
        info = self.slides[idx]
        data = np.load(info['path'], allow_pickle=True)
        features = data['features'].astype(np.float32)
        patch_labels = data['patch_labels'].astype(np.float32)
        
        # Subsample if max_patches is set
        if self.max_patches and len(features) > self.max_patches:
            rng = np.random.RandomState(idx)
            sel = rng.choice(len(features), self.max_patches, replace=False)
            features = features[sel]
            patch_labels = patch_labels[sel]
        
        return {
            'features': torch.from_numpy(features),
            'patch_labels': torch.from_numpy(patch_labels),
            'slide_id': info['slide_id'],
            'center': info['center'],
            'tumor_fraction': info['tumor_fraction'],
        }


# ============================================================
# MIL Models
# ============================================================

class ABMIL(nn.Module):
    """Attention-Based Multiple Instance Learning.
    
    Gated attention mechanism learns to weight patches within a slide.
    """
    
    def __init__(self, input_dim=512, hidden=256, dropout=0.1):
        super().__init__()
        self.name = 'ABMIL'
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Gated attention
        self.attention_V = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden, 1)
        
        # Patch-level classifier
        self.patch_classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        
        # Slide-level classifier
        self.slide_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (N, D) patch features for one slide
        Returns:
            patch_logits: (N, 1) per-patch predictions
            slide_logit: scalar slide-level prediction
            attention_weights: (N, 1) attention scores
        """
        # Encode features
        h = self.encoder(x)  # (N, hidden)
        
        # Gated attention
        a_V = self.attention_V(h)  # (N, hidden)
        a_U = self.attention_U(h)  # (N, hidden)
        a = a_V * a_U  # (N, hidden)
        attention_logits = self.attention_weights(a)  # (N, 1)
        attention_weights = F.softmax(attention_logits, dim=0)  # (N, 1)
        
        # Slide representation (attention-weighted sum of original features)
        slide_repr = (attention_weights * x).sum(dim=0, keepdim=True)  # (1, D)
        slide_logit = self.slide_classifier(slide_repr).squeeze()  # scalar
        
        # Patch-level predictions
        patch_logits = self.patch_classifier(h).squeeze(-1)  # (N,)
        
        return patch_logits, slide_logit, attention_weights.squeeze()


class CLAM(nn.Module):
    """Clustering-constrained Attention MIL (CLAM).
    
    Multi-head attention with instance-level clustering constraint.
    Based on Lu et al., IEEE TMI 2021.
    """
    
    def __init__(self, input_dim=512, hidden=256, n_heads=2, dropout=0.1,
                 top_k=8, instance_loss_weight=0.7):
        super().__init__()
        self.name = 'CLAM'
        self.n_heads = n_heads
        self.top_k = top_k
        self.instance_loss_weight = instance_loss_weight
        
        # Instance classifier (patch-level)
        self.instance_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            ) for _ in range(n_heads)
        ])
        
        # Multi-head gated attention
        self.attention_V = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh())
            for _ in range(n_heads)
        ])
        self.attention_U = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden), nn.Sigmoid())
            for _ in range(n_heads)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(n_heads)
        ])
        
        # Slide-level classifiers (one per head)
        self.slide_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            ) for _ in range(n_heads)
        ])
    
    def forward(self, x, patch_labels=None):
        """
        Args:
            x: (N, D) patch features
            patch_labels: (N,) ground truth patch labels (for instance loss)
        Returns:
            patch_logits: (N,) per-patch predictions (from head 0)
            slide_logit: scalar
            attention_weights: (N,) from head 0
            instance_loss: scalar (clustering constraint)
        """
        slide_logits = []
        all_attention = []
        instance_loss = 0.0
        
        for h_idx in range(self.n_heads):
            # Attention
            a_V = self.attention_V[h_idx](x)
            a_U = self.attention_U[h_idx](x)
            a = a_V * a_U
            att_logits = self.attention_weights[h_idx](a)  # (N, 1)
            att_weights = F.softmax(att_logits, dim=0)  # (N, 1)
            
            # Slide representation
            slide_repr = (att_weights * x).sum(dim=0, keepdim=True)  # (1, D)
            slide_logit = self.slide_classifiers[h_idx](slide_repr).squeeze()
            slide_logits.append(slide_logit)
            all_attention.append(att_weights.squeeze())
            
            # Instance-level loss (clustering constraint)
            if patch_labels is not None and self.training:
                patch_logits_h = self.instance_classifiers[h_idx](x).squeeze(-1)  # (N,)
                
                # Top-k positive and negative instances
                k = min(self.top_k, len(patch_logits_h) // 2)
                if k > 0:
                    pos_idx = torch.topk(patch_logits_h, k).indices
                    neg_idx = torch.topk(-patch_logits_h, k).indices
                    
                    pos_labels = torch.ones(k, device=x.device)
                    neg_labels = torch.zeros(k, device=x.device)
                    
                    instance_loss += F.binary_cross_entropy_with_logits(
                        patch_logits_h[pos_idx], pos_labels)
                    instance_loss += F.binary_cross_entropy_with_logits(
                        patch_logits_h[neg_idx], neg_labels)
        
        # Aggregate slide logits across heads
        slide_logit = torch.stack(slide_logits).mean()
        
        # Use head 0 for patch predictions and attention
        patch_logits = self.instance_classifiers[0](x).squeeze(-1)
        attention_weights = all_attention[0]
        
        return patch_logits, slide_logit, attention_weights, instance_loss


# ============================================================
# Training
# ============================================================

def train_mil_model(model, train_loader, val_loader, device, model_name='abmil',
                    max_epochs=50, patience=10, lr=1e-4, weight_decay=1e-4):
    """Train a MIL model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                            factor=0.5, patience=5)
    
    best_val_corr = -1
    best_state = None
    no_improve = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_train = 0
        
        for batch in train_loader:
            x = batch['features'].to(device)
            patch_labels = batch['patch_labels'].to(device)
            tumor_fraction = batch['tumor_fraction'].to(device)
            
            optimizer.zero_grad()
            
            if model_name == 'abmil':
                patch_logits, slide_logit, att = model(x)
                # Patch-level loss
                patch_loss = F.binary_cross_entropy_with_logits(
                    patch_logits, patch_labels)
                # Slide-level loss (predict tumor fraction)
                slide_loss = F.mse_loss(torch.sigmoid(slide_logit), tumor_fraction)
                loss = patch_loss + 0.5 * slide_loss
            else:  # clam
                patch_logits, slide_logit, att, instance_loss = model(x, patch_labels)
                patch_loss = F.binary_cross_entropy_with_logits(
                    patch_logits, patch_labels)
                slide_loss = F.mse_loss(torch.sigmoid(slide_logit), tumor_fraction)
                loss = patch_loss + 0.5 * slide_loss + model.instance_loss_weight * instance_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_train += 1
        
        # Validation
        model.eval()
        val_predictions = []
        val_fractions = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['features'].to(device)
                if model_name == 'abmil':
                    patch_logits, slide_logit, att = model(x)
                else:
                    patch_logits, slide_logit, att, _ = model(x)
                
                slide_pred = torch.sigmoid(slide_logit).item()
                val_predictions.append(slide_pred)
                val_fractions.append(batch['tumor_fraction'].item())
        
        val_corr = pearsonr(val_predictions, val_fractions)[0] if len(val_predictions) > 2 else 0
        
        scheduler.step(val_corr)
        
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{max_epochs}: train_loss={train_loss/n_train:.4f}, "
                  f"val_corr={val_corr:.3f}, best={best_val_corr:.3f}")
        
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_corr


# ============================================================
# Evaluation
# ============================================================

def evaluate_mil_model(model, test_loader, device, model_name='abmil'):
    """Evaluate MIL model on test slides."""
    model.eval()
    
    all_patch_preds = []
    all_patch_labels = []
    slide_predictions = []
    slide_fractions = []
    slide_centers = []
    slide_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(device)
            patch_labels = batch['patch_labels'].to(device)
            
            if model_name == 'abmil':
                patch_logits, slide_logit, att = model(x)
            else:
                patch_logits, slide_logit, att, _ = model(x)
            
            # Patch-level predictions
            patch_preds = torch.sigmoid(patch_logits).cpu().numpy()
            all_patch_preds.extend(patch_preds.tolist())
            all_patch_labels.extend(patch_labels.cpu().numpy().tolist())
            
            # Slide-level prediction
            slide_pred = torch.sigmoid(slide_logit).item()
            slide_predictions.append(slide_pred)
            slide_fractions.append(batch['tumor_fraction'].item())
            slide_centers.append(batch['center'].item())
            slide_ids.append(batch['slide_id'].item())
    
    # Patch-level metrics
    patch_auc = roc_auc_score(all_patch_labels, all_patch_preds) if len(set(all_patch_labels)) > 1 else 0
    patch_preds_binary = (np.array(all_patch_preds) > 0.5).astype(int)
    patch_acc = accuracy_score(all_patch_labels, patch_preds_binary)
    patch_f1 = f1_score(all_patch_labels, patch_preds_binary, zero_division=0)
    
    # Slide-level metrics
    if len(slide_predictions) > 2:
        slide_pearson = pearsonr(slide_predictions, slide_fractions)[0]
        slide_spearman = spearmanr(slide_predictions, slide_fractions)[0]
    else:
        slide_pearson = 0
        slide_spearman = 0
    
    slide_mse = np.mean((np.array(slide_predictions) - np.array(slide_fractions))**2)
    slide_mae = np.mean(np.abs(np.array(slide_predictions) - np.array(slide_fractions)))
    
    return {
        'patch_auc': float(patch_auc),
        'patch_accuracy': float(patch_acc),
        'patch_f1': float(patch_f1),
        'slide_pearson': float(slide_pearson),
        'slide_spearman': float(slide_spearman),
        'slide_mse': float(slide_mse),
        'slide_mae': float(slide_mae),
        'n_test_slides': len(slide_predictions),
        'n_test_patches': len(all_patch_preds),
        'slide_predictions': slide_predictions,
        'slide_fractions': slide_fractions,
        'slide_ids': slide_ids,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='WSI-Level MIL Pipeline')
    parser.add_argument('--backbone', choices=BACKBONES + ['all'], default='all')
    parser.add_argument('--model', choices=MODELS + ['all'], default='all')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--features_dir', default=FEATURES_DIR)
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    backbones = BACKBONES if args.backbone == 'all' else [args.backbone]
    models = MODELS if args.model == 'all' else [args.model]
    
    os.makedirs(args.results_dir, exist_ok=True)
    all_results = {}
    
    for bb_name in backbones:
        print(f"\n{'='*60}")
        print(f"Backbone: {bb_name}")
        print(f"{'='*60}")
        
        # Load datasets
        try:
            train_ds = MILDataset(args.features_dir, bb_name, TRAIN_CENTERS)
            val_ds = MILDataset(args.features_dir, bb_name, [VAL_CENTER])
            test_ds = MILDataset(args.features_dir, bb_name, [TEST_CENTER])
        except FileNotFoundError as e:
            print(f"  Skipping {bb_name}: {e}")
            continue
        
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        
        # Get feature dimension from first sample
        sample = train_ds[0]
        feat_dim = sample['features'].shape[1]
        print(f"  Feature dim: {feat_dim}")
        
        for model_name in models:
            print(f"\n  Model: {model_name.upper()}")
            
            seed_results = []
            for seed in range(args.seeds):
                print(f"\n    Seed {seed+1}/{args.seeds}")
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Create model
                if model_name == 'abmil':
                    model = ABMIL(input_dim=feat_dim, hidden=256, dropout=0.1)
                else:
                    model = CLAM(input_dim=feat_dim, hidden=256, n_heads=2,
                                 dropout=0.1, top_k=8)
                model = model.to(device)
                
                # Train
                t0 = time.time()
                model, best_val_corr = train_mil_model(
                    model, train_loader, val_loader, device, model_name,
                    max_epochs=args.max_epochs, patience=args.patience)
                train_time = time.time() - t0
                
                # Evaluate
                results = evaluate_mil_model(model, test_loader, device, model_name)
                results['train_time'] = train_time
                results['best_val_corr'] = best_val_corr
                results['seed'] = seed
                
                print(f"    Results: patch_AUC={results['patch_auc']:.3f}, "
                      f"slide_pearson={results['slide_pearson']:.3f}, "
                      f"slide_spearman={results['slide_spearman']:.3f}, "
                      f"time={train_time:.1f}s")
                
                seed_results.append(results)
                
                # Clear memory
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Aggregate across seeds
            patch_aucs = [r['patch_auc'] for r in seed_results]
            pearsons = [r['slide_pearson'] for r in seed_results]
            spearmans = [r['slide_spearman'] for r in seed_results]
            
            agg = {
                'patch_auc_mean': float(np.mean(patch_aucs)),
                'patch_auc_std': float(np.std(patch_aucs)),
                'patch_auc_ci': float(1.96 * np.std(patch_aucs) / np.sqrt(len(patch_aucs))),
                'slide_pearson_mean': float(np.mean(pearsons)),
                'slide_pearson_std': float(np.std(pearsons)),
                'slide_pearson_ci': float(1.96 * np.std(pearsons) / np.sqrt(len(pearsons))),
                'slide_spearman_mean': float(np.mean(spearmans)),
                'slide_spearman_std': float(np.std(spearmans)),
                'n_seeds': len(seed_results),
                'seed_results': seed_results,
            }
            
            key = f"{bb_name}_{model_name}"
            all_results[key] = agg
            
            print(f"\n  {model_name.upper()} Summary ({bb_name}):")
            print(f"    Patch AUC: {agg['patch_auc_mean']:.3f} ± {agg['patch_auc_ci']:.3f}")
            print(f"    Slide Pearson: {agg['slide_pearson_mean']:.3f} ± {agg['slide_pearson_ci']:.3f}")
            print(f"    Slide Spearman: {agg['slide_spearman_mean']:.3f} ± {agg['slide_spearman_std']:.3f}")
    
    # Save all results
    results_path = os.path.join(args.results_dir, 'all_mil_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: WSI-Level MIL Results")
    print(f"{'='*70}")
    print(f"{'Backbone':<15} {'Model':<8} {'Patch AUC':<15} {'Slide Pearson':<15} {'Slide Spearman':<15}")
    print("-" * 70)
    for key, r in sorted(all_results.items()):
        bb, mm = key.rsplit('_', 1)
        print(f"{bb:<15} {mm:<8} {r['patch_auc_mean']:.3f}±{r['patch_auc_ci']:.3f}   "
              f"{r['slide_pearson_mean']:.3f}±{r['slide_pearson_ci']:.3f}   "
              f"{r['slide_spearman_mean']:.3f}±{r['slide_spearman_std']:.3f}")


if __name__ == '__main__':
    main()
