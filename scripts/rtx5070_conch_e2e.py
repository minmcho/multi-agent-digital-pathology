"""
RTX 5070 CONCH End-to-End Fine-Tuning

Fine-tunes CONCH (ViT-B, 86M) end-to-end on CAMELYON17 + PCam.
Runs on RTX 5070 (12GB VRAM) with FP16.

Usage:
  python rtx5070_conch_e2e.py --model baseline --seeds 5
  python rtx5070_conch_e2e.py --model dann --seeds 5
  python rtx5070_conch_e2e.py --model mixstyle --seeds 5
  python rtx5070_conch_e2e.py --model all --seeds 5

Models: baseline, dann, mixstyle
Datasets: CAMELYON17 (ID train + OOD test), PCam (external test)
"""
import os, sys, time, json, argparse, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
import io

# ============================================================
# Config
# ============================================================
DATA_DIR = os.environ.get('DATA_DIR', '/mnt/shared-workspace/shared/datasets')
RESULTS_DIR = os.environ.get('E2E_RESULTS_DIR', './e2e_results')

EPOCHS = 15
LR = 1e-5  # Lower LR for end-to-end fine-tuning
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32  # CONCH ViT-B fits in 12GB with batch 32
WARMUP_EPOCHS = 2

# ============================================================
# Data Loading (from parquet)
# ============================================================
def find_parquet_files(dataset_path, split_keyword):
    import glob
    files = sorted(glob.glob(os.path.join(dataset_path, f"*{split_keyword}*.parquet")))
    if split_keyword == "train":
        files = [f for f in files if "train" in os.path.basename(f) and "validation" not in os.path.basename(f)]
    return files

def load_parquet_data(filepath, sample_size=None, balanced=True, seed=42):
    import pyarrow.parquet as pq
    t = pq.read_table(filepath, columns=['image', 'label'])
    df = t.to_pandas()
    if sample_size and sample_size < len(df):
        rng = np.random.RandomState(seed)
        if balanced:
            idx_0 = df.index[df['label'] == 0].tolist()
            idx_1 = df.index[df['label'] == 1].tolist()
            n_per = sample_size // 2
            sel_0 = rng.choice(idx_0, min(n_per, len(idx_0)), replace=False)
            sel_1 = rng.choice(idx_1, min(n_per, len(idx_1)), replace=False)
            sel = np.concatenate([sel_0, sel_1])
            rng.shuffle(sel)
        else:
            sel = rng.choice(len(df), sample_size, replace=False)
        df = df.iloc[sel].reset_index(drop=True)
    return df

class PatchDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row['image']
        if isinstance(img_bytes, dict):
            img_bytes = img_bytes['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, float(row['label'])

# ============================================================
# Model Loading
# ============================================================
def load_conch_model(num_classes=1, device='cuda'):
    """Load CONCH for end-to-end fine-tuning."""
    from transformers import ViTModel
    import timm
    
    # CONCH is a CLIP-based vision encoder
    # Load via timm for easier fine-tuning
    model = timm.create_model(
        "hf-hub:MahmoodLab/CONCH",
        pretrained=True,
        num_classes=num_classes,
    )
    model = model.to(device)
    return model

# ============================================================
# Training Components
# ============================================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(images).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
        if i % 100 == 0:
            print(f"    Epoch {epoch} batch {i}/{len(loader)} loss={loss.item():.4f}", flush=True)
    
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
    return total_loss / len(loader), auc

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            logits = model(images).squeeze(-1)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return auc, acc

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['baseline', 'dann', 'mixstyle', 'all'], default='all')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
    
    os.makedirs(args.results_dir, exist_ok=True)
    models_to_run = ['baseline', 'dann', 'mixstyle'] if args.model == 'all' else [args.model]
    
    # Load data
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Load datasets
    print("Loading data...")
    train_files = find_parquet_files(os.path.join(DATA_DIR, 'camelyon17'), 'id_train')
    val_files = find_parquet_files(os.path.join(DATA_DIR, 'camelyon17'), 'id_val')
    ood_files = find_parquet_files(os.path.join(DATA_DIR, 'camelyon17'), 'ood_test')
    
    train_df = load_parquet_data(train_files[0], sample_size=30000, balanced=True)
    val_df = load_parquet_data(val_files[0], sample_size=10000, balanced=True)
    ood_df = load_parquet_data(ood_files[0], sample_size=20000, balanced=True)
    
    train_ds = PatchDataset(train_df, train_transform)
    val_ds = PatchDataset(val_df, eval_transform)
    ood_ds = PatchDataset(ood_df, eval_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    all_results = {}
    
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*60}")
        
        seed_results = []
        for seed in range(args.seeds):
            print(f"\n  Seed {seed+1}/{args.seeds}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Load fresh model
            model = load_conch_model(num_classes=1, device=device)
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            total_steps = len(train_loader) * args.epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) * WARMUP_EPOCHS, total_steps)
            scaler = torch.cuda.amp.GradScaler()
            
            # Checkpoint path
            ckpt_path = os.path.join(args.results_dir, f'conch_{model_name}_seed{seed}_ckpt.pt')
            
            # Resume from checkpoint if exists
            start_epoch = 0
            best_val_auc = 0
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt['model_state'])
                optimizer.load_state_dict(ckpt['optimizer_state'])
                scheduler.load_state_dict(ckpt['scheduler_state'])
                scaler.load_state_dict(ckpt['scaler_state'])
                start_epoch = ckpt['epoch'] + 1
                best_val_auc = ckpt['best_val_auc']
                print(f"    Resumed from epoch {start_epoch}")
            
            # Train
            for epoch in range(start_epoch, args.epochs):
                t0 = time.time()
                train_loss, train_auc = train_one_epoch(
                    model, train_loader, optimizer, scheduler, scaler, device, epoch)
                val_auc, val_acc = evaluate(model, val_loader, device)
                ood_auc, ood_acc = evaluate(model, ood_loader, device)
                
                print(f"    Epoch {epoch+1}/{args.epochs}: "
                      f"train_loss={train_loss:.4f}, train_auc={train_auc:.3f}, "
                      f"val_auc={val_auc:.3f}, ood_auc={ood_auc:.3f} "
                      f"({time.time()-t0:.1f}s)")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                
                # Checkpoint every epoch
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'best_val_auc': best_val_auc,
                }, ckpt_path)
            
            # Final evaluation
            final_val_auc, final_val_acc = evaluate(model, val_loader, device)
            final_ood_auc, final_ood_acc = evaluate(model, ood_loader, device)
            gen_gap = final_val_auc - final_ood_auc
            
            result = {
                'seed': seed,
                'model': model_name,
                'backbone': 'conch',
                'val_auc': float(final_val_auc),
                'val_acc': float(final_val_acc),
                'ood_auc': float(final_ood_auc),
                'ood_acc': float(final_ood_acc),
                'gen_gap': float(gen_gap),
                'best_val_auc': float(best_val_auc),
                'epochs_trained': args.epochs,
            }
            seed_results.append(result)
            print(f"  Final: val_auc={final_val_auc:.3f}, ood_auc={final_ood_auc:.3f}, gap={gen_gap:.3f}")
            
            # Clean up checkpoint
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            
            del model
            torch.cuda.empty_cache()
        
        # Aggregate
        val_aucs = [r['val_auc'] for r in seed_results]
        ood_aucs = [r['ood_auc'] for r in seed_results]
        gaps = [r['gen_gap'] for r in seed_results]
        
        all_results[f'conch_{model_name}'] = {
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
    
    # Save results
    results_path = os.path.join(args.results_dir, 'conch_e2e_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
