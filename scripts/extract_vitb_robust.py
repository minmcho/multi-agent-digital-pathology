"""
Robust ViT-B feature extraction: saves each domain/split separately.
If the process crashes, completed domains are preserved on disk.
Can resume by re-running (skips existing files).

Usage:
  python extract_vitb_robust.py
"""
import pyarrow.parquet as pq
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor
from skimage.color import rgb2hed, hed2rgb
from PIL import Image
import io, os, glob, time, subprocess, collections

DATA_DIR = "/mnt/shared-workspace/shared/datasets"
SHARED_DIR = "/mnt/shared-workspace/shared"
FEATURE_DIR = os.path.join(SHARED_DIR, "features_v2", "vit_b")
TMP_DIR = "/workspace/_feat_tmp"

MODEL_ID = "owkin/phikon"
FEATURE_DIM = 768
BATCH_SIZE = 16
CHUNK_SIZE = 500

# Sample sizes (reduced for speed)
ID_TRAIN_SAMPLE = 10000  # per domain (30K total for 3 HED domains)
PCAM_TRAIN_SAMPLE = 10000
ID_VAL_SAMPLE = 5000
OOD_TEST_SAMPLE = 8000
PCAM_TEST_SAMPLE = 5000
PCAM_VAL_SAMPLE = 5000

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

def save_npz(outpath, features, labels, domains):
    """Save to local disk, then cp to S3 mount."""
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    tmp_path = os.path.join(TMP_DIR, os.path.basename(outpath))
    np.savez_compressed(tmp_path, features=features,
                        labels=np.array(labels), domains=np.array(domains))
    subprocess.run(['cp', tmp_path, outpath], check=True)
    os.remove(tmp_path)
    print(f"  Saved {outpath}: features={features.shape}, "
          f"labels={collections.Counter(np.array(labels).tolist())}, "
          f"domains={collections.Counter(np.array(domains).tolist())}", flush=True)

def extract_features_chunked(model, processor, image_bytes_list, aug_params,
                              batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE):
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

def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    t_start = time.time()
    torch.set_num_threads(4)

    print(f"=== Robust ViT-B Feature Extraction ===", flush=True)
    print(f"Model: {MODEL_ID}, dim: {FEATURE_DIM}", flush=True)

    print(f"\nLoading {MODEL_ID}...", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    print(f"  Loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    # ===== CAMELYON17 ID_TRAIN: Save each HED domain separately =====
    print(f"\n--- Loading camelyon17 id_train ---", flush=True)
    files = find_parquet("camelyon17", "id_train")
    all_image_bytes, all_labels = [], []
    per_file = ID_TRAIN_SAMPLE // max(len(files), 1)
    for fpath in files:
        sel, lbls = load_parquet_indices(fpath, sample_size=per_file, balanced=True)
        all_image_bytes.extend(load_image_bytes_by_index(fpath, sel))
        all_labels.extend(lbls)
        print(f"  {os.path.basename(fpath)}: {len(lbls)} samples", flush=True)
    rng = np.random.RandomState(42)
    if len(all_labels) > ID_TRAIN_SAMPLE:
        sel = rng.choice(len(all_labels), ID_TRAIN_SAMPLE, replace=False)
        all_image_bytes = [all_image_bytes[i] for i in sel]
        all_labels = [all_labels[i] for i in sel]
    print(f"  Total: {len(all_labels)}, labels={collections.Counter(all_labels)}", flush=True)

    # Extract and save each domain separately
    for dom_idx, aug_params in enumerate(HED_AUGS):
        dom_name = f"camelyon17_id_train_domain{dom_idx}"
        outpath = os.path.join(FEATURE_DIR, f"{dom_name}.npz")
        if os.path.exists(outpath):
            print(f"\n  [SKIP] {dom_name}.npz already exists", flush=True)
            continue
        print(f"\n  Domain {dom_idx}: {aug_params or 'original'}", flush=True)
        t0 = time.time()
        feats = extract_features_chunked(model, processor, all_image_bytes, aug_params)
        print(f"    Domain {dom_idx}: {feats.shape} ({time.time()-t0:.1f}s)", flush=True)
        save_npz(outpath, feats, all_labels, [dom_idx]*len(all_labels))
        del feats

    # ===== PCAM TRAIN (domain 3) =====
    pcam_out = os.path.join(FEATURE_DIR, "pcam_train.npz")
    if os.path.exists(pcam_out):
        print(f"\n[SKIP] pcam_train.npz exists", flush=True)
    else:
        print(f"\n--- PCam train (sample={PCAM_TRAIN_SAMPLE}) ---", flush=True)
        files = find_parquet("pcam", "train")
        all_img, all_lbl = [], []
        per_file = PCAM_TRAIN_SAMPLE // max(len(files), 1)
        for fpath in files:
            sel, lbls = load_parquet_indices(fpath, sample_size=per_file, balanced=True)
            all_img.extend(load_image_bytes_by_index(fpath, sel))
            all_lbl.extend(lbls)
        rng = np.random.RandomState(42)
        if len(all_lbl) > PCAM_TRAIN_SAMPLE:
            sel = rng.choice(len(all_lbl), PCAM_TRAIN_SAMPLE, replace=False)
            all_img = [all_img[i] for i in sel]
            all_lbl = [all_lbl[i] for i in sel]
        feats = extract_features_chunked(model, processor, all_img, None)
        save_npz(pcam_out, feats, all_lbl, [3]*len(all_lbl))
        del all_img, feats

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
            print(f"\n[SKIP] {out_name}.npz exists", flush=True)
            continue
        print(f"\n--- {dataset} {split_kw} (sample={sample_size}) ---", flush=True)
        files = find_parquet(dataset, split_kw)
        sel, lbls = load_parquet_indices(files[0], sample_size=sample_size, balanced=True)
        img_bytes = load_image_bytes_by_index(files[0], sel)
        feats = extract_features_chunked(model, processor, img_bytes, None)
        save_npz(outpath, feats, lbls, [domain_label]*len(lbls))
        del img_bytes, feats

    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE in {(time.time()-t_start)/60:.1f} min", flush=True)
    for fname in sorted(os.listdir(FEATURE_DIR)):
        fpath = os.path.join(FEATURE_DIR, fname)
        if os.path.isfile(fpath):
            print(f"  {fname}: {os.path.getsize(fpath)/1e6:.1f} MB", flush=True)

if __name__ == "__main__":
    main()
