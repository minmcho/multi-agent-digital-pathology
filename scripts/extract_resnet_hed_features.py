"""
Memory-efficient ResNet-18 feature extraction with HED stain augmentation.
Same domain design as Phikon extraction. Processes in chunks.

Usage:
  python extract_resnet_hed_features.py
"""
import pyarrow.parquet as pq
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from skimage.color import rgb2hed, hed2rgb
from PIL import Image
import io, os, glob, time, collections

DATA_DIR = "/mnt/shared-workspace/shared/datasets"
SHARED_DIR = "/mnt/shared-workspace/shared"
FEATURE_DIR = os.path.join(SHARED_DIR, "features_v2", "resnet18")

ID_TRAIN_SAMPLE = 30000
PCAM_TRAIN_SAMPLE = 30000
ID_VAL_SAMPLE = 10000
OOD_TEST_SAMPLE = 20000
PCAM_TEST_SAMPLE = 10000
PCAM_VAL_SAMPLE = 10000

IMAGE_SIZE = 224
BATCH_SIZE = 128
CHUNK_SIZE = 2000

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

def extract_resnet_chunked(model, image_bytes_list, aug_params,
                            batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
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
                tensors = torch.stack([transform(img) for img in batch])
                feats = model(tensors).squeeze(-1).squeeze(-1).cpu().numpy()
                all_features.append(feats)
        del pil_imgs
        if (chunk_start // chunk_size) % 5 == 0:
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

def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    t_start = time.time()
    torch.set_num_threads(4)

    print("=== ResNet-18 HED Feature Extraction ===")
    print(f"Output: {FEATURE_DIR}")

    print("\nLoading ResNet-18...")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = nn.Sequential(*list(resnet18(weights=weights).children())[:-1])
    model.eval()
    print("  Loaded. Output dim: 512")

    # ===== CAMELYON17 ID_TRAIN (HED domains 0, 1, 2) =====
    train_out = os.path.join(FEATURE_DIR, "camelyon17_id_train.npz")
    if os.path.exists(train_out):
        print(f"\n[SKIP] camelyon17_id_train.npz exists")
    else:
        print(f"\n--- Camelyon17 id_train (sample={ID_TRAIN_SAMPLE}) ---")
        files = find_parquet("camelyon17", "id_train")
        all_image_bytes, all_labels = [], []
        per_file = ID_TRAIN_SAMPLE // max(len(files), 1)
        for fpath in files:
            sel, lbls = load_parquet_indices(fpath, sample_size=per_file, balanced=True)
            all_image_bytes.extend(load_image_bytes_by_index(fpath, sel))
            all_labels.extend(lbls)
        rng = np.random.RandomState(42)
        if len(all_labels) > ID_TRAIN_SAMPLE:
            sel = rng.choice(len(all_labels), ID_TRAIN_SAMPLE, replace=False)
            all_image_bytes = [all_image_bytes[i] for i in sel]
            all_labels = [all_labels[i] for i in sel]
        print(f"  Total: {len(all_labels)}, labels={collections.Counter(all_labels)}")

        all_features, all_domains, all_labels_full = [], [], []
        for dom_idx, aug_params in enumerate(HED_AUGS):
            print(f"  Domain {dom_idx}: {aug_params or 'original'}")
            t0 = time.time()
            feats = extract_resnet_chunked(model, all_image_bytes, aug_params)
            print(f"  Domain {dom_idx}: {feats.shape} ({time.time()-t0:.1f}s)")
            all_features.append(feats)
            all_domains.extend([dom_idx] * len(all_labels))
            all_labels_full.extend(all_labels)
        save_features(train_out, np.concatenate(all_features), all_labels_full, all_domains)
        del all_image_bytes, all_features

    # ===== PCAM TRAIN (domain 3) =====
    pcam_out = os.path.join(FEATURE_DIR, "pcam_train.npz")
    if os.path.exists(pcam_out):
        print(f"\n[SKIP] pcam_train.npz exists")
    else:
        print(f"\n--- PCam train (sample={PCAM_TRAIN_SAMPLE}) ---")
        files = find_parquet("pcam", "train")
        all_image_bytes, all_labels = [], []
        per_file = PCAM_TRAIN_SAMPLE // max(len(files), 1)
        for fpath in files:
            sel, lbls = load_parquet_indices(fpath, sample_size=per_file, balanced=True)
            all_image_bytes.extend(load_image_bytes_by_index(fpath, sel))
            all_labels.extend(lbls)
        rng = np.random.RandomState(42)
        if len(all_labels) > PCAM_TRAIN_SAMPLE:
            sel = rng.choice(len(all_labels), PCAM_TRAIN_SAMPLE, replace=False)
            all_image_bytes = [all_image_bytes[i] for i in sel]
            all_labels = [all_labels[i] for i in sel]
        feats = extract_resnet_chunked(model, all_image_bytes, None)
        save_features(pcam_out, feats, all_labels, [3]*len(all_labels))
        del all_image_bytes

    # ===== EVAL SPLITS =====
    eval_splits = [
        ("camelyon17", "id_val", "camelyon17_id_val", ID_VAL_SAMPLE, 0),
        ("camelyon17", "ood_test", "camelyon17_ood_test", OOD_TEST_SAMPLE, 1),
        ("pcam", "test", "pcam_test", PCAM_TEST_SAMPLE, 3),
        ("pcam", "validation", "pcam_validation", PCAM_VAL_SAMPLE, 3),
    ]
    for dataset, split_kw, out_name, sample_size, domain_label in eval_splits:
        outpath = os.path.join(FEATURE_DIR, f"{out_name}.npz")
        if os.path.exists(outpath):
            print(f"\n[SKIP] {out_name}.npz exists")
            continue
        print(f"\n--- {dataset} {split_kw} (sample={sample_size}) ---")
        files = find_parquet(dataset, split_kw)
        sel, lbls = load_parquet_indices(files[0], sample_size=sample_size, balanced=True)
        img_bytes = load_image_bytes_by_index(files[0], sel)
        print(f"  {out_name}: {len(lbls)}, labels={collections.Counter(lbls)}")
        feats = extract_resnet_chunked(model, img_bytes, None)
        save_features(outpath, feats, lbls, [domain_label]*len(lbls))
        del img_bytes

    print(f"\n{'='*60}")
    print(f"COMPLETE in {(time.time()-t_start)/60:.1f} min")
    for fname in sorted(os.listdir(FEATURE_DIR)):
        fpath = os.path.join(FEATURE_DIR, fname)
        if os.path.isfile(fpath):
            print(f"  {fname}: {os.path.getsize(fpath)/1e6:.1f} MB")

if __name__ == "__main__":
    main()
