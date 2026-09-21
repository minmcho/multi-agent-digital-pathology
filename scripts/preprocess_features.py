"""
Preprocessing and feature extraction for multi-agent pathology AI framework.
- Loads Camelyon17 and PCam from parquet (dynamic file discovery)
- Creates synthetic domains via stain augmentation (id_train only)
- Extracts ResNet-18 (ImageNet) features
- Caches features + metadata to .npz
- Skips splits that already have a cached .npz
"""
import pyarrow.parquet as pq
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io, os, glob, time
import collections

# ============ CONFIGURATION ============
DATA_DIR = "/mnt/shared-workspace/shared/datasets"
FEATURE_DIR = "/workspace/features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# Subsample sizes (for CPU feasibility)
ID_TRAIN_SAMPLE = 30000  # balanced 15K normal, 15K tumor
ID_VAL_SAMPLE = 10000
OOD_TEST_SAMPLE = 20000
OOD_VAL_SAMPLE = 10000
PCAM_TRAIN_SAMPLE = 30000
PCAM_TEST_SAMPLE = 10000
PCAM_VAL_SAMPLE = 10000

# Stain augmentation parameters for synthetic domains
DOMAIN_AUGS = [
    {"hue": 0, "sat": 1.0, "val": 1.0, "contrast": 1.0},     # Domain 0: original
    {"hue": 0.08, "sat": 1.25, "val": 1.0, "contrast": 1.0},  # Domain 1: hue shift +
    {"hue": -0.08, "sat": 0.85, "val": 1.15, "contrast": 1.0},# Domain 2: hue shift -
    {"hue": 0, "sat": 1.0, "val": 0.90, "contrast": 1.30},    # Domain 3: contrast +
]

IMAGE_SIZE = 224
BATCH_SIZE = 128

# ============ HELPER FUNCTIONS ============

def find_parquet(dataset, split_keyword):
    """Find parquet file(s) matching a split keyword in a dataset dir."""
    ddir = os.path.join(DATA_DIR, dataset)
    files = sorted(glob.glob(os.path.join(ddir, f"*{split_keyword}*.parquet")))
    return files

def decode_image(image_bytes, size=IMAGE_SIZE):
    """Decode bytes to PIL Image and resize."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return img

def apply_stain_aug(img_pil, aug_params):
    """Apply stain-like augmentation in HSV space."""
    img = img_pil.convert('HSV')
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = (arr[:, :, 0] + aug_params["hue"] * 255) % 255
    arr[:, :, 1] = np.clip(arr[:, :, 1] * aug_params["sat"], 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * aug_params["val"], 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), 'HSV').convert('RGB')
    if aug_params["contrast"] != 1.0:
        arr_rgb = np.array(img, dtype=np.float32)
        arr_rgb = (arr_rgb - 128) * aug_params["contrast"] + 128
        img = Image.fromarray(np.clip(arr_rgb, 0, 255).astype(np.uint8), 'RGB')
    return img

def load_parquet_data(filepath, sample_size=None, balanced=True, random_seed=42):
    """Load data from parquet, optionally subsampling."""
    t = pq.read_table(filepath)
    images = t.column('image').to_pylist()
    labels = t.column('label').to_pylist()
    
    if sample_size and sample_size < len(labels):
        rng = np.random.RandomState(random_seed)
        if balanced:
            idx_0 = [i for i, l in enumerate(labels) if l == 0]
            idx_1 = [i for i, l in enumerate(labels) if l == 1]
            n_per = sample_size // 2
            if len(idx_0) > 0 and len(idx_1) > 0:
                sel_0 = rng.choice(idx_0, min(n_per, len(idx_0)), replace=False)
                sel_1 = rng.choice(idx_1, min(n_per, len(idx_1)), replace=False)
                sel = np.concatenate([sel_0, sel_1])
                rng.shuffle(sel)
            else:
                sel = rng.choice(len(labels), sample_size, replace=False)
        else:
            sel = rng.choice(len(labels), sample_size, replace=False)
        images = [images[i] for i in sel]
        labels = [labels[i] for i in sel]
    
    return images, labels

def extract_features(model, images_pil, device='cpu', batch_size=BATCH_SIZE):
    """Extract features from a list of PIL images."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images_pil), batch_size):
            batch = images_pil[i:i+batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            feats = model(tensors)
            all_features.append(feats.cpu().numpy())
            if (i // batch_size) % 50 == 0:
                print(f"    Batch {i//batch_size}/{len(images_pil)//batch_size}", flush=True)
    
    return np.concatenate(all_features, axis=0)

def save_features(name, features, labels, domains):
    """Save features to npz. Squeeze any trailing (1,1) dims."""
    if features.ndim > 2:
        features = features.squeeze()
    outpath = os.path.join(FEATURE_DIR, f"{name}.npz")
    np.savez_compressed(outpath,
                        features=features,
                        labels=np.array(labels),
                        domains=np.array(domains))
    print(f"  Saved {outpath}: features={features.shape}, "
          f"labels={collections.Counter(np.array(labels).tolist())}, "
          f"domains={collections.Counter(np.array(domains).tolist())}")

# ============ MAIN PROCESSING ============

def main():
    t_start = time.time()
    
    # Load ResNet-18 (ImageNet pretrained, remove classification head)
    print("Loading pre-trained ResNet-18...")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove FC, keep avgpool
    model = model.to('cpu')
    model.eval()
    torch.set_num_threads(8)
    print(f"  ResNet-18 loaded. Output dim: 512")
    
    # ===== CAMELYON17 ID_TRAIN (with synthetic domains) =====
    train_out = os.path.join(FEATURE_DIR, "camelyon17_id_train.npz")
    if os.path.exists(train_out):
        print(f"\n[SKIP] camelyon17_id_train.npz already exists ({os.path.getsize(train_out)/1e6:.1f} MB)")
    else:
        print("\n--- Loading Camelyon17 id_train ---")
        id_train_files = find_parquet("camelyon17", "id_train")
        print(f"  Found {len(id_train_files)} files: {[os.path.basename(f) for f in id_train_files]}")
        
        id_train_images_raw = []
        id_train_labels = []
        per_file = ID_TRAIN_SAMPLE // max(len(id_train_files), 1)
        for fpath in id_train_files:
            imgs, lbls = load_parquet_data(fpath, sample_size=per_file, balanced=True)
            id_train_images_raw.extend(imgs)
            id_train_labels.extend(lbls)
            print(f"  {os.path.basename(fpath)}: loaded {len(imgs)} samples")
        
        # Subsample to target if we overshot
        rng = np.random.RandomState(42)
        if len(id_train_labels) > ID_TRAIN_SAMPLE:
            sel = rng.choice(len(id_train_labels), ID_TRAIN_SAMPLE, replace=False)
            id_train_images_raw = [id_train_images_raw[i] for i in sel]
            id_train_labels = [id_train_labels[i] for i in sel]
        
        print(f"  Total id_train: {len(id_train_labels)} samples, labels={collections.Counter(id_train_labels)}")
        
        print("  Decoding images...")
        id_train_pil = [decode_image(img['bytes']) for img in id_train_images_raw]
        
        print("  Creating synthetic domains (4 stain augmentations)...")
        all_features = []
        all_labels = []
        all_domains = []
        
        for dom_idx, aug_params in enumerate(DOMAIN_AUGS):
            print(f"    Domain {dom_idx}: {aug_params}")
            if dom_idx == 0:
                augmented = id_train_pil
            else:
                augmented = [apply_stain_aug(img, aug_params) for img in id_train_pil]
            
            feats = extract_features(model, augmented)
            all_features.append(feats)
            all_labels.extend(id_train_labels)
            all_domains.extend([dom_idx] * len(id_train_labels))
            print(f"    Domain {dom_idx} features: {feats.shape}")
        
        id_train_features = np.concatenate(all_features, axis=0)
        save_features("camelyon17_id_train", id_train_features, all_labels, all_domains)
    
    # ===== CAMELYON17 ID_VAL =====
    val_out = os.path.join(FEATURE_DIR, "camelyon17_id_val.npz")
    if os.path.exists(val_out):
        print(f"\n[SKIP] camelyon17_id_val.npz already exists")
    else:
        print("\n--- Loading Camelyon17 id_val ---")
        val_files = find_parquet("camelyon17", "id_val")
        print(f"  Found: {[os.path.basename(f) for f in val_files]}")
        imgs, lbls = load_parquet_data(val_files[0], sample_size=ID_VAL_SAMPLE, balanced=True)
        print(f"  id_val: {len(lbls)} samples, labels={collections.Counter(lbls)}")
        pil_imgs = [decode_image(img['bytes']) for img in imgs]
        feats = extract_features(model, pil_imgs)
        save_features("camelyon17_id_val", feats, lbls, [0]*len(lbls))
    
    # ===== CAMELYON17 OOD_TEST =====
    ood_test_out = os.path.join(FEATURE_DIR, "camelyon17_ood_test.npz")
    if os.path.exists(ood_test_out):
        print(f"\n[SKIP] camelyon17_ood_test.npz already exists")
    else:
        print("\n--- Loading Camelyon17 ood_test ---")
        ood_files = find_parquet("camelyon17", "ood_test")
        print(f"  Found: {[os.path.basename(f) for f in ood_files]}")
        imgs, lbls = load_parquet_data(ood_files[0], sample_size=OOD_TEST_SAMPLE, balanced=True)
        print(f"  ood_test: {len(lbls)} samples, labels={collections.Counter(lbls)}")
        pil_imgs = [decode_image(img['bytes']) for img in imgs]
        feats = extract_features(model, pil_imgs)
        save_features("camelyon17_ood_test", feats, lbls, [1]*len(lbls))
    
    # ===== CAMELYON17 OOD_VAL =====
    ood_val_out = os.path.join(FEATURE_DIR, "camelyon17_ood_val.npz")
    if os.path.exists(ood_val_out):
        print(f"\n[SKIP] camelyon17_ood_val.npz already exists")
    else:
        print("\n--- Loading Camelyon17 ood_val ---")
        oodv_files = find_parquet("camelyon17", "ood_val")
        print(f"  Found: {[os.path.basename(f) for f in oodv_files]}")
        imgs, lbls = load_parquet_data(oodv_files[0], sample_size=OOD_VAL_SAMPLE, balanced=True)
        print(f"  ood_val: {len(lbls)} samples, labels={collections.Counter(lbls)}")
        pil_imgs = [decode_image(img['bytes']) for img in imgs]
        feats = extract_features(model, pil_imgs)
        save_features("camelyon17_ood_val", feats, lbls, [1]*len(lbls))
    
    # ===== PCAM =====
    print("\n" + "="*60)
    print("PROCESSING PCAM")
    print("="*60)
    
    pcam_splits = [("train", PCAM_TRAIN_SAMPLE), ("test", PCAM_TEST_SAMPLE), ("validation", PCAM_VAL_SAMPLE)]
    for split, sample_size in pcam_splits:
        outpath = os.path.join(FEATURE_DIR, f"pcam_{split}.npz")
        if os.path.exists(outpath):
            print(f"\n[SKIP] pcam_{split}.npz already exists")
            continue
        
        print(f"\n--- Loading PCam {split} ---")
        # For train, there are 2 files; for test/val, 1 file each
        files = find_parquet("pcam", split)
        # Filter: "train" keyword also matches in "validation" — handle that
        if split == "train":
            files = [f for f in files if "train" in os.path.basename(f)]
        elif split == "test":
            files = [f for f in files if "test" in os.path.basename(f)]
        elif split == "validation":
            files = [f for f in files if "validation" in os.path.basename(f)]
        
        print(f"  Found: {[os.path.basename(f) for f in files]}")
        
        all_imgs = []
        all_lbls = []
        per_file = sample_size // max(len(files), 1)
        for fpath in files:
            imgs, lbls = load_parquet_data(fpath, sample_size=per_file, balanced=True)
            all_imgs.extend(imgs)
            all_lbls.extend(lbls)
            print(f"  {os.path.basename(fpath)}: loaded {len(imgs)} samples")
        
        # Subsample if overshot
        rng = np.random.RandomState(42)
        if len(all_lbls) > sample_size:
            sel = rng.choice(len(all_lbls), sample_size, replace=False)
            all_imgs = [all_imgs[i] for i in sel]
            all_lbls = [all_lbls[i] for i in sel]
        
        print(f"  {split}: {len(all_lbls)} samples, labels={collections.Counter(all_lbls)}")
        pil_imgs = [decode_image(img['bytes']) for img in all_imgs]
        feats = extract_features(model, pil_imgs)
        save_features(f"pcam_{split}", feats, all_lbls, [0]*len(all_lbls))
    
    # ===== SUMMARY =====
    t_end = time.time()
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION COMPLETE in {t_end - t_start:.1f}s ({(t_end - t_start)/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\nFeature files in {FEATURE_DIR}:")
    for fname in sorted(os.listdir(FEATURE_DIR)):
        fpath = os.path.join(FEATURE_DIR, fname)
        size = os.path.getsize(fpath) / 1e6
        print(f"  {fname}: {size:.1f} MB")

if __name__ == "__main__":
    main()
