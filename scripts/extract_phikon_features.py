"""
Memory-efficient Phikon feature extraction with HED stain augmentation.
Processes images in chunks to avoid OOM.

Domain design:
  Domain 0: Camelyon17 ID train (original)
  Domain 1: Camelyon17 ID train (HED: H×1.3, E×0.8)
  Domain 2: Camelyon17 ID train (HED: H×0.7, E×1.3)
  Domain 3: PCam train (real cross-dataset)

Usage:
  python extract_phikon_features.py --backbone vit_tiny
  python extract_phikon_features.py --backbone vit_b
"""
import pyarrow.parquet as pq
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor
from skimage.color import rgb2hed, hed2rgb
from PIL import Image
import io, os, glob, time, argparse, collections

DATA_DIR = "/mnt/shared-workspace/shared/datasets"
SHARED_DIR = "/mnt/shared-workspace/shared"

BACKBONE_CONFIGS = {
    'vit_tiny': {
        'model_id': '1aurent/phikon-distil-vit-tiny-patch16-224-kather2016',
        'feature_dim': 192, 'batch_size': 64, 'chunk_size': 1000,
        'id_train_sample': 30000, 'pcam_train_sample': 30000,
        'id_val_sample': 10000, 'ood_test_sample': 20000,
        'pcam_test_sample': 10000, 'pcam_val_sample': 10000,
    },
    'vit_b': {
        'model_id': 'owkin/phikon',
        'feature_dim': 768, 'batch_size': 16, 'chunk_size': 500,
        'id_train_sample': 15000, 'pcam_train_sample': 15000,
        'id_val_sample': 5000, 'ood_test_sample': 10000,
        'pcam_test_sample': 5000, 'pcam_val_sample': 5000,
    },
}

HED_AUGS = [
    None,
    {'h_scale': 1.3, 'e_scale': 0.8},
    {'h_scale': 0.7, 'e_scale': 1.3},
]

def find_parquet(dataset, split_keyword):
    ddir = os.path.join(DATA_DIR, dataset)
    files = sorted(glob.glob(os.path.join(ddir, f"*{split_keyword}*.parquet")))
    if split_keyword == "train":
        files = [f for f in files if "train" in os.path.basename(f) and "validation" not in os.path.basename(f)]
    elif split_keyword == "test":
        files = [f for f in files if "test" in os.path.basename(f)]
    elif split_keyword == "validation":
        files = [f for f in files if "validation" in os.path.basename(f)]
    return files

def decode_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def hed_augment(img_pil, h_scale=1.0, e_scale=1.0):
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    hed = rgb2hed(img_arr)
    hed[:, :, 0] *= h_scale
    hed[:, :, 1] *= e_scale
    hed = np.clip(hed, 0, 1)
    rgb_aug = hed2rgb(hed)
    rgb_aug = np.clip(rgb_aug * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_aug)

def load_parquet_indices(filepath, sample_size=None, balanced=True, random_seed=42):
    t = pq.read_table(filepath, columns=['label'])
    labels = t.column('label').to_pylist()
    n_total = len(labels)
    if sample_size and sample_size < n_total:
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
                sel = rng.choice(n_total, sample_size, replace=False)
        else:
            sel = rng.choice(n_total, sample_size, replace=False)
        return sel, [labels[i] for i in sel]
    return np.arange(n_total), labels

def load_image_bytes_by_index(filepath, indices):
    t = pq.read_table(filepath, columns=['image'])
    all_images = t.column('image').to_pylist()
    return [all_images[i] for i in indices]

def extract_features_chunked(model, processor, image_bytes_list, aug_params,
                              batch_size=64, chunk_size=1000):
    all_features = []
    n_total = len(image_bytes_list)
    model.eval()
    for chunk_start in range(0, n_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_total)
        chunk_bytes = image_bytes_list[chunk_start:chunk_end]
        pil_imgs = [decode_image(b['bytes']) for b in chunk_bytes]
        if aug_params is not None:
            pil_imgs = [hed_augment(img, **aug_params) for img in pil_imgs]
        with torch.no_grad():
            for i in range(0, len(pil_imgs), batch_size):
                batch = pil_imgs[i:i+batch_size]
                inputs = processor(images=batch, return_tensors="pt")
                outputs = model(**inputs)
                feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_features.append(feats)
        del pil_imgs
        if (chunk_start // chunk_size) % 10 == 0:
            pct = chunk_end / n_total * 100
            print(f"    Chunk {chunk_start//chunk_size}/{n_total//chunk_size} ({pct:.0f}%)", flush=True)
    return np.concatenate(all_features, axis=0)

def save_features(outpath, features, labels, domains):
    if features.ndim > 2:
        features = features.squeeze()
    # Save to local disk first (S3 FUSE doesn't support ZIP seek ops)
    import shutil, subprocess
    tmpdir = "/workspace/_feat_tmp"
    os.makedirs(tmpdir, exist_ok=True)
    tmp_path = os.path.join(tmpdir, os.path.basename(outpath))
    np.savez_compressed(tmp_path, features=features,
                        labels=np.array(labels), domains=np.array(domains))
    # Ensure dest dir exists on S3 mount, then copy (cp handles FUSE better than shutil)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    subprocess.run(['cp', tmp_path, outpath], check=True)
    os.remove(tmp_path)
    print(f"  Saved {outpath}: features={features.shape}, "
          f"labels={collections.Counter(np.array(labels).tolist())}, "
          f"domains={collections.Counter(np.array(domains).tolist())}")

def process_split(model, processor, cfg, feature_dir, dataset, split_kw, out_name,
                   sample_size, domain_label, aug_params=None, is_train_with_domains=False):
    """Process a single split (eval or train domain)."""
    outpath = os.path.join(feature_dir, f"{out_name}.npz")
    if os.path.exists(outpath):
        print(f"\n[SKIP] {out_name}.npz already exists")
        return

    print(f"\n--- Loading {dataset} {split_kw} (sample={sample_size}) ---")
    files = find_parquet(dataset, split_kw)

    if is_train_with_domains:
        # Multi-file loading for train splits
        all_image_bytes = []
        all_labels = []
        per_file = sample_size // max(len(files), 1)
        for fpath in files:
            sel, lbls = load_parquet_indices(fpath, sample_size=per_file, balanced=True)
            img_bytes = load_image_bytes_by_index(fpath, sel)
            all_image_bytes.extend(img_bytes)
            all_labels.extend(lbls)
            print(f"  {os.path.basename(fpath)}: {len(lbls)} samples")

        rng = np.random.RandomState(42)
        if len(all_labels) > sample_size:
            sel = rng.choice(len(all_labels), sample_size, replace=False)
            all_image_bytes = [all_image_bytes[i] for i in sel]
            all_labels = [all_labels[i] for i in sel]
    else:
        sel, lbls = load_parquet_indices(files[0], sample_size=sample_size, balanced=True)
        all_image_bytes = load_image_bytes_by_index(files[0], sel)
        all_labels = lbls

    print(f"  Total: {len(all_labels)}, labels={collections.Counter(all_labels)}")

    if aug_params is not None:
        # Extract features for each HED domain
        all_features = []
        all_domains = []
        all_labels_full = []
        for dom_idx, ap in enumerate(HED_AUGS):
            print(f"  Domain {dom_idx}: {ap or 'original'}")
            t0 = time.time()
            feats = extract_features_chunked(model, processor, all_image_bytes, ap,
                                              batch_size=cfg['batch_size'], chunk_size=cfg['chunk_size'])
            print(f"    Domain {dom_idx}: {feats.shape} ({time.time()-t0:.1f}s)")
            all_features.append(feats)
            all_domains.extend([dom_idx] * len(all_labels))
            all_labels_full.extend(all_labels)
        features = np.concatenate(all_features, axis=0)
        save_features(outpath, features, all_labels_full, all_domains)
        del all_image_bytes, all_features
    else:
        feats = extract_features_chunked(model, processor, all_image_bytes, None,
                                          batch_size=cfg['batch_size'], chunk_size=cfg['chunk_size'])
        save_features(outpath, feats, all_labels, [domain_label]*len(all_labels))
        del all_image_bytes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['vit_tiny', 'vit_b'], required=True)
    args = parser.parse_args()

    cfg = BACKBONE_CONFIGS[args.backbone]
    feature_dir = os.path.join(SHARED_DIR, "features_v2", args.backbone)
    os.makedirs(feature_dir, exist_ok=True)

    t_start = time.time()
    torch.set_num_threads(4)

    print(f"=== Phikon Feature Extraction: {args.backbone} ===")
    print(f"Model: {cfg['model_id']}, dim: {cfg['feature_dim']}")

    print(f"\nLoading {cfg['model_id']}...")
    processor = AutoImageProcessor.from_pretrained(cfg['model_id'])
    model = AutoModel.from_pretrained(cfg['model_id'])
    model.eval()
    print(f"  Loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Camelyon17 ID train with HED domains
    process_split(model, processor, cfg, feature_dir, "camelyon17", "id_train",
                  "camelyon17_id_train", cfg['id_train_sample'], 0,
                  aug_params="multi_domain", is_train_with_domains=True)

    # PCam train (domain 3)
    process_split(model, processor, cfg, feature_dir, "pcam", "train",
                  "pcam_train", cfg['pcam_train_sample'], 3, is_train_with_domains=True)

    # Eval splits
    process_split(model, processor, cfg, feature_dir, "camelyon17", "id_val",
                  "camelyon17_id_val", cfg['id_val_sample'], 0)
    process_split(model, processor, cfg, feature_dir, "camelyon17", "ood_test",
                  "camelyon17_ood_test", cfg['ood_test_sample'], 1)
    process_split(model, processor, cfg, feature_dir, "pcam", "test",
                  "pcam_test", cfg['pcam_test_sample'], 3)
    process_split(model, processor, cfg, feature_dir, "pcam", "validation",
                  "pcam_validation", cfg['pcam_val_sample'], 3)

    print(f"\n{'='*60}")
    print(f"COMPLETE in {(time.time()-t_start)/60:.1f} min")
    for fname in sorted(os.listdir(feature_dir)):
        fpath = os.path.join(feature_dir, fname)
        if os.path.isfile(fpath):
            print(f"  {fname}: {os.path.getsize(fpath)/1e6:.1f} MB")

if __name__ == "__main__":
    main()
