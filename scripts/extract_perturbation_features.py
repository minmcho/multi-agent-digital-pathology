"""
Extract features for perturbed PCam test images.
Applies 5 perturbation types × 3 severity levels = 15 conditions
at the IMAGE level, then extracts features using the specified backbone.

Usage:
  python extract_perturbation_features.py --backbone vit_tiny
  python extract_perturbation_features.py --backbone resnet18
  python extract_perturbation_features.py --backbone vit_b --sample_size 5000
"""
import pyarrow.parquet as pq
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel, AutoImageProcessor
from skimage.color import rgb2hed, hed2rgb
from PIL import Image, ImageFilter
import io, os, glob, time, argparse, collections
import json
import shutil
import subprocess

def _save_npz(outpath, features, labels):
    """Save npz to local disk first, then copy to S3-backed shared workspace."""
    tmpdir = "/workspace/_feat_tmp"
    os.makedirs(tmpdir, exist_ok=True)
    tmp_path = os.path.join(tmpdir, os.path.basename(outpath))
    np.savez_compressed(tmp_path, features=features, labels=np.array(labels))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    subprocess.run(['cp', tmp_path, outpath], check=True)
    os.remove(tmp_path)

# ============ CONFIGURATION ============
DATA_DIR = "/mnt/shared-workspace/shared/datasets"
SHARED_DIR = "/mnt/shared-workspace/shared"

BACKBONE_CONFIGS = {
    'vit_tiny': {
        'model_id': '1aurent/phikon-distil-vit-tiny-patch16-224-kather2016',
        'feature_dim': 192,
        'batch_size': 64,
        'sample_size': 10000,
    },
    'vit_b': {
        'model_id': 'owkin/phikon',
        'feature_dim': 768,
        'batch_size': 16,
        'sample_size': 5000,
    },
    'resnet18': {
        'model_id': None,
        'feature_dim': 512,
        'batch_size': 128,
        'sample_size': 10000,
    },
}

# ============ PERTURBATION FUNCTIONS ============

def apply_gaussian_noise(img_pil, sigma=0.05):
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    noise = np.random.randn(*arr.shape) * sigma
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_stain_shift(img_pil, magnitude=0.2):
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    hed = rgb2hed(img_arr)
    h_shift = 1.0 + np.random.choice([-1, 1]) * magnitude
    e_shift = 1.0 - np.random.choice([-1, 1]) * magnitude
    hed[:, :, 0] *= h_shift
    hed[:, :, 1] *= e_shift
    hed = np.clip(hed, 0, 1)
    rgb_aug = hed2rgb(hed)
    rgb_aug = np.clip(rgb_aug * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_aug)

def apply_gaussian_blur(img_pil, sigma=1.0):
    return img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

def apply_jpeg_compression(img_pil, quality=20):
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def apply_brightness_contrast(img_pil, delta=0.2):
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    brightness = np.random.choice([-1, 1]) * delta
    contrast = 1.0 - delta * 0.5
    arr = (arr - 0.5) * contrast + 0.5 + brightness
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

PERTURBATIONS = {
    'gaussian_noise': {'func': apply_gaussian_noise, 'severities': [0.02, 0.05, 0.10], 'labels': ['mild', 'moderate', 'severe']},
    'stain_shift': {'func': apply_stain_shift, 'severities': [0.1, 0.2, 0.4], 'labels': ['mild', 'moderate', 'severe']},
    'gaussian_blur': {'func': apply_gaussian_blur, 'severities': [0.5, 1.0, 2.0], 'labels': ['mild', 'moderate', 'severe']},
    'jpeg_compression': {'func': apply_jpeg_compression, 'severities': [50, 20, 5], 'labels': ['mild', 'moderate', 'severe']},
    'brightness_contrast': {'func': apply_brightness_contrast, 'severities': [0.1, 0.2, 0.4], 'labels': ['mild', 'moderate', 'severe']},
}

# ============ HELPER FUNCTIONS ============

def decode_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def load_pcam_test(sample_size=10000):
    """Load PCam test images from parquet."""
    test_file = sorted(glob.glob(os.path.join(DATA_DIR, "pcam", "*test*.parquet")))[0]
    t = pq.read_table(test_file, columns=['image', 'label'])
    images = t.column('image').to_pylist()
    labels = t.column('label').to_pylist()
    if sample_size < len(labels):
        rng = np.random.RandomState(42)
        sel = rng.choice(len(labels), sample_size, replace=False)
        images = [images[i] for i in sel]
        labels = [labels[i] for i in sel]
    return images, labels

def extract_phikon_features(model, processor, images_pil, batch_size=64):
    all_features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images_pil), batch_size):
            batch = images_pil[i:i+batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_features.append(feats)
            if (i // batch_size) % 50 == 0:
                print(f"    Batch {i//batch_size}/{len(images_pil)//batch_size}", flush=True)
    return np.concatenate(all_features, axis=0)

def extract_resnet_features(model, images_pil, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    all_features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images_pil), batch_size):
            batch = images_pil[i:i+batch_size]
            tensors = torch.stack([transform(img) for img in batch])
            feats = model(tensors).squeeze(-1).squeeze(-1).cpu().numpy()
            all_features.append(feats)
            if (i // batch_size) % 50 == 0:
                print(f"    Batch {i//batch_size}/{len(images_pil)//batch_size}", flush=True)
    return np.concatenate(all_features, axis=0)

# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['vit_tiny', 'vit_b', 'resnet18'], required=True)
    parser.add_argument('--sample_size', type=int, default=None)
    args = parser.parse_args()

    cfg = BACKBONE_CONFIGS[args.backbone]
    sample_size = args.sample_size or cfg['sample_size']
    pert_dir = os.path.join(SHARED_DIR, "features_v2", args.backbone, "perturbations")
    os.makedirs(pert_dir, exist_ok=True)

    t_start = time.time()
    torch.set_num_threads(8)

    print(f"=== Perturbation Feature Extraction: {args.backbone} ===")
    print(f"Sample size: {sample_size}")
    print(f"Output dir: {pert_dir}")

    # Load backbone model
    if args.backbone == 'resnet18':
        print("\nLoading ResNet-18...")
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = nn.Sequential(*list(resnet18(weights=weights).children())[:-1])
        model.eval()
        extract_fn = lambda imgs: extract_resnet_features(model, imgs, cfg['batch_size'])
    else:
        print(f"\nLoading {cfg['model_id']}...")
        processor = AutoImageProcessor.from_pretrained(cfg['model_id'])
        model = AutoModel.from_pretrained(cfg['model_id'])
        model.eval()
        extract_fn = lambda imgs: extract_phikon_features(model, processor, imgs, cfg['batch_size'])

    # Load PCam test images
    print(f"\nLoading PCam test images (sample={sample_size})...")
    images_raw, labels = load_pcam_test(sample_size)
    pil_imgs = [decode_image(img['bytes']) for img in images_raw]
    print(f"  Loaded {len(pil_imgs)} images, labels={collections.Counter(labels)}")

    # Extract baseline (unperturbed) features
    baseline_path = os.path.join(pert_dir, "baseline.npz")
    if os.path.exists(baseline_path):
        print(f"\n[SKIP] baseline.npz already exists")
    else:
        print("\nExtracting baseline features...")
        feats = extract_fn(pil_imgs)
        _save_npz(baseline_path, feats, labels)
        print(f"  Saved baseline: {feats.shape}")

    # Extract perturbed features for each condition
    for ptype, pconfig in PERTURBATIONS.items():
        for severity, sev_label in zip(pconfig['severities'], pconfig['labels']):
            cond_name = f"{ptype}_{sev_label}"
            outpath = os.path.join(pert_dir, f"{cond_name}.npz")
            if os.path.exists(outpath):
                print(f"\n[SKIP] {cond_name}.npz already exists")
                continue

            print(f"\n--- {cond_name} (severity={severity}) ---")
            np.random.seed(42)  # Reproducible perturbations
            perturbed = [pconfig['func'](img, severity) for img in pil_imgs]
            t0 = time.time()
            feats = extract_fn(perturbed)
            elapsed = time.time() - t0
            _save_npz(outpath, feats, labels)
            print(f"  Saved {cond_name}: {feats.shape} ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print(f"COMPLETE in {(time.time()-t_start)/60:.1f} min")
    print(f"Files in {pert_dir}:")
    for fname in sorted(os.listdir(pert_dir)):
        print(f"  {fname}: {os.path.getsize(os.path.join(pert_dir, fname))/1e6:.1f} MB")

if __name__ == "__main__":
    main()
