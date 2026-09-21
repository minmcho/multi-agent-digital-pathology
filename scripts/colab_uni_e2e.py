"""
Colab Pro: UNI (ViT-L) End-to-End Fine-Tuning

Paste this entire script into a Colab Pro notebook cell.
Set runtime: GPU (L4 or A100), enable internet.

UNI is ViT-L (304M params). On L4 (24GB): batch 16, no grad checkpointing needed.
On A100 (40GB): batch 32. On T4 (16GB): batch 8 + gradient checkpointing.

Models: baseline, dann, mixstyle
Dataset: CAMELYON17 (ID train + OOD test)
"""
# ============================================================
# Colab Setup Cell 1: Install dependencies
# ============================================================
# !pip install timm transformers pyarrow scikit-learn --quiet

# ============================================================
# Colab Setup Cell 2: Mount Google Drive (for checkpointing)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# os.makedirs('/content/drive/MyDrive/mil_e2e_results', exist_ok=True)

# ============================================================
# Main Script Cell 3: Run this
# ============================================================
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
import io

# Config
DATA_DIR = '/content/data'  # Upload parquet files here or use Google Drive
RESULTS_DIR = '/content/drive/MyDrive/mil_e2e_results'  # Save to Drive
CKPT_DIR = '/content/checkpoints'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = 15
LR = 5e-6  # Very low LR for ViT-L fine-tuning
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2

# Detect GPU and set batch size
device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

if 'A100' in gpu_name:
    BATCH_SIZE = 32
    USE_GRAD_CKPT = False
elif 'L4' in gpu_name:
    BATCH_SIZE = 16
    USE_GRAD_CKPT = False
elif 'T4' in gpu_name:
    BATCH_SIZE = 8
    USE_GRAD_CKPT = True
else:
    BATCH_SIZE = 8
    USE_GRAD_CKPT = True

print(f"Batch size: {BATCH_SIZE}, Grad checkpointing: {USE_GRAD_CKPT}")

# ============================================================
# Data Loading
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
# Model
# ============================================================
def load_uni_model(device, grad_checkpointing=False):
    import timm
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        num_classes=1,
    )
    model = model.to(device)
    if grad_checkpointing:
        model.set_grad_checkpointing(enable=True)
        print("  Gradient checkpointing enabled")
    return model

# ============================================================
# Training
# ============================================================
def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        if i % 50 == 0:
            print(f"  Epoch {epoch} batch {i}/{len(loader)} loss={loss.item():.4f}", flush=True)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
    return total_loss / len(loader), auc

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast():
            logits = model(imgs).squeeze(-1)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return auc, acc

# ============================================================
# Run
# ============================================================
def run_experiment(model_name, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"UNI E2E: {model_name} seed {seed}")
    print(f"{'='*60}")
    
    # Load model
    model = load_uni_model(device, grad_checkpointing=USE_GRAD_CKPT)
    
    # Data transforms
    from torchvision import transforms
    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Load data
    train_files = find_parquet_files(os.path.join(DATA_DIR, 'camelyon17'), 'id_train')
    val_files = find_parquet_files(os.path.join(DATA_DIR, 'camelyon17'), 'id_val')
    ood_files = find_parquet_files(os.path.join(DATA_DIR, 'camelyon17'), 'ood_test')
    
    train_df = load_parquet_data(train_files[0], sample_size=30000, balanced=True)
    val_df = load_parquet_data(val_files[0], sample_size=10000, balanced=True)
    ood_df = load_parquet_data(ood_files[0], sample_size=20000, balanced=True)
    
    train_ds = PatchDataset(train_df, train_t)
    val_ds = PatchDataset(val_df, eval_t)
    ood_ds = PatchDataset(ood_df, eval_t)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule(optimizer, len(train_loader) * WARMUP_EPOCHS, total_steps)
    scaler = torch.cuda.amp.GradScaler()
    
    # Checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f'uni_{model_name}_seed{seed}.pt')
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
        print(f"  Resumed from epoch {start_epoch}")
    
    # Train
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch)
        val_auc, val_acc = evaluate(model, val_loader, device)
        ood_auc, ood_acc = evaluate(model, ood_loader, device)
        
        print(f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} train_auc={train_auc:.3f} "
              f"val_auc={val_auc:.3f} ood_auc={ood_auc:.3f} ({time.time()-t0:.1f}s)")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        # Checkpoint to Drive
        torch.save({
            'epoch': epoch, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_val_auc': best_val_auc,
        }, ckpt_path)
    
    # Final eval
    final_val_auc, _ = evaluate(model, val_loader, device)
    final_ood_auc, _ = evaluate(model, ood_loader, device)
    
    result = {
        'backbone': 'uni', 'model': model_name, 'seed': seed,
        'val_auc': float(final_val_auc), 'ood_auc': float(final_ood_auc),
        'gen_gap': float(final_val_auc - final_ood_auc),
        'best_val_auc': float(best_val_auc),
    }
    print(f"\n  FINAL: val_auc={final_val_auc:.3f} ood_auc={final_ood_auc:.3f} gap={final_val_auc-final_ood_auc:.3f}")
    
    # Save result to Drive
    result_path = os.path.join(RESULTS_DIR, f'uni_{model_name}_seed{seed}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    
    return result

# ============================================================
# Execute
# ============================================================
MODELS = ['baseline', 'dann', 'mixstyle']  # Adjust as needed
SEEDS = 3  # Adjust: 3 for speed, 5 for more robust results

all_results = []
for model_name in MODELS:
    for seed in range(SEEDS):
        result = run_experiment(model_name, seed)
        all_results.append(result)

# Save combined results
combined_path = os.path.join(RESULTS_DIR, 'uni_e2e_results.json')
with open(combined_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nAll results saved to {combined_path}")
print(f"Done! Download results from Google Drive: {RESULTS_DIR}")
