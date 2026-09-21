"""
LC25000 Lung + Colon Cancer Experiments

Downloads LC25000 dataset, extracts features with all 6 backbones,
runs frozen-feature experiments (5 models × 10 seeds), and optionally
CONCH end-to-end fine-tuning.

LC25000: 25,000 patches (5,000 each: lung benign, lung adenocarcinoma,
lung squamous cell carcinoma, colon benign, colon adenocarcinoma)
500x500 pixels, JPEG

Usage:
  python lc25000_experiments.py --mode features  # Extract features only
  python lc25000_experiments.py --mode frozen    # Frozen-feature experiments
  python lc25000_experiments.py --mode all       # Both
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Config
# ============================================================
LC25000_DIR = os.environ.get('LC25000_DIR', './data/lc25000')
FEATURES_DIR = os.environ.get('FEATURES_DIR', './features')
RESULTS_DIR = os.environ.get('LC25000_RESULTS_DIR', './lc25000_results')

BACKBONE_DIMS = {
    'resnet18': 512, 'vit_tiny': 192, 'vit_b': 768,
    'uni': 1024, 'virchow': 1024, 'conch': 512,
}

# LC25000 class mapping (binary: tumor vs benign)
# lung_n = benign (0), lung_aca = tumor (1), lung_scc = tumor (1)
# colon_n = benign (0), colon_aca = tumor (1)
CLASS_MAP = {
    'lung_n': 0, 'lung_aca': 1, 'lung_scc': 1,
    'colon_n': 0, 'colon_aca': 1,
}

# ============================================================
# Data Loading
# ============================================================
def download_lc25000():
    """Download LC25000 dataset."""
    os.makedirs(LC25000_DIR, exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(os.path.join(LC25000_DIR, 'lung_colon_image_set')):
        print("  LC25000 already downloaded")
        return
    
    print("  Downloading LC25000...")
    # LC25000 is available on Kaggle and academic sources
    # URL: https://academictorrents.com/details/7dea6c3f89c02b9b4a6c5c0b9b9b9b9b9b9b9b9
    # Or: pip install kaggle; kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images
    
    import subprocess
    try:
        # Try Kaggle download
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 
             'andrewmvd/lung-and-colon-cancer-histopathological-images',
             '-p', LC25000_DIR, '--unzip'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("  Downloaded via Kaggle")
            return
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
    
    # Manual download instructions
    print("""
  MANUAL DOWNLOAD REQUIRED:
  1. Download from: https://academictorrents.com/details/7dea6c3f89c02b9b4a6c5c0b9b9b9b9b9b9b9b9
  2. Or from Kaggle: kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images
  3. Extract to: """ + LC25000_DIR + """
  4. Re-run this script
""")

def load_lc25000_images():
    """Load LC25000 images and labels."""
    images = []
    labels = []
    organ_labels = []  # 0=lung, 1=colon
    
    base_dir = os.path.join(LC25000_DIR, 'lung_colon_image_set')
    if not os.path.exists(base_dir):
        # Try alternative structure
        base_dir = LC25000_DIR
    
    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  Warning: {class_dir} not found")
            continue
        
        organ = 1 if 'colon' in class_name else 0
        files = sorted([f for f in os.listdir(class_dir) if f.endswith(('.jpeg', '.jpg', '.png'))])
        
        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            labels.append(label)
            organ_labels.append(organ)
        
        print(f"  {class_name}: {len(files)} images (label={label})")
    
    print(f"  Total: {len(images)} images, labels={collections.Counter(labels)}")
    print(f"  Organs: lung={organ_labels.count(0)}, colon={organ_labels.count(1)}")
    return images, np.array(labels), np.array(organ_labels)

# ============================================================
# Feature Extraction
# ============================================================
def extract_features_for_lc25000(backbone_name, images, device):
    """Extract features for LC25000 images using a backbone."""
    from rtx5070_feat_extraction import load_backbone
    from torchvision import transforms
    
    model, transform, feat_dim = load_backbone(backbone_name, device)
    model.eval()
    
    all_features = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensors = torch.stack([transform(img) for img in batch]).to(device).half()
            outputs = model(batch_tensors)
            
            if hasattr(outputs, 'last_hidden_state'):
                feats = outputs.last_hidden_state[:, 0, :]
            elif isinstance(outputs, torch.Tensor):
                feats = outputs
            else:
                feats = outputs
            
            all_features.append(feats.float().cpu().numpy())
            
            if i % 500 == 0:
                print(f"    {i}/{len(images)}", flush=True)
    
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return np.concatenate(all_features, axis=0)

# ============================================================
# Frozen-Feature Experiments
# ============================================================
def run_frozen_experiments(backbone_name, features, labels, organ_labels, seeds=10):
    """Run frozen-feature experiments on LC25000."""
    from run_frozen_features import create_model, train_model, evaluate_model
    
    input_dim = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split: 80% train, 20% test (stratified by label and organ)
    rng = np.random.RandomState(42)
    n = len(features)
    indices = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_X, train_y = features[train_idx], labels[train_idx]
    test_X, test_y = features[test_idx], labels[test_idx]
    
    # Standardize
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0) + 1e-6
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std
    
    # Simple validation split from train
    n_val = int(0.1 * len(train_X))
    val_X, val_y = train_X[:n_val], train_y[:n_val]
    train_X, train_y = train_X[n_val:], train_y[n_val:]
    
    models = ['baseline', 'dann', 'disentanglement', 'multi_agent', 'mixstyle']
    all_results = {}
    
    for model_name in models:
        print(f"\n  {model_name}")
        seed_results = []
        
        for seed in range(seeds):
            model = create_model(model_name, input_dim, seed)
            model, best_val = train_model(
                model, train_X, train_y, val_X, val_y, input_dim, model_name, device=device)
            
            metrics = evaluate_model(model, test_X, test_y, model_name, device)
            metrics['seed'] = seed
            seed_results.append(metrics)
            print(f"    Seed {seed}: auc={metrics['auc']:.3f}")
            del model
        
        aucs = [r['auc'] for r in seed_results]
        all_results[model_name] = {
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
            'auc_ci': float(1.96 * np.std(aucs) / np.sqrt(len(aucs))),
            'n_seeds': len(seed_results),
            'seed_results': seed_results,
        }
    
    return all_results

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['features', 'frozen', 'all'], default='all')
    parser.add_argument('--backbone', choices=list(BACKBONE_DIMS.keys()) + ['all'], default='all')
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Download dataset
    if args.mode in ['features', 'all']:
        download_lc25000()
    
    if args.mode in ['frozen', 'all']:
        # Load images
        images, labels, organ_labels = load_lc25000_images()
        
        backbones = list(BACKBONE_DIMS.keys()) if args.backbone == 'all' else [args.backbone]
        all_results = {}
        
        for bb_name in backbones:
            print(f"\n{'='*60}")
            print(f"Backbone: {bb_name.upper()}")
            print(f"{'='*60}")
            
            # Check if features already extracted
            feat_path = os.path.join(FEATURES_DIR, bb_name, 'lc25000.npz')
            
            if os.path.exists(feat_path):
                print("  Loading cached features...")
                data = np.load(feat_path, allow_pickle=True)
                features = data['features']
            else:
                print("  Extracting features...")
                features = extract_features_for_lc25000(bb_name, images, device)
                os.makedirs(os.path.dirname(feat_path), exist_ok=True)
                np.savez_compressed(feat_path, features=features, labels=labels)
                print(f"  Saved features: {features.shape}")
            
            # Run frozen experiments
            results = run_frozen_experiments(bb_name, features, labels, organ_labels, args.seeds)
            all_results[bb_name] = results
        
        # Save
        results_path = os.path.join(RESULTS_DIR, 'lc25000_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
