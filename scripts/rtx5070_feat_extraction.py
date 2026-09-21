"""
RTX 5070 Feature Extraction: UNI, Virchow, CONCH
Extracts features from CAMELYON17, PCam, Kather2016, and WILDS patches.

Usage (on machine with RTX 5070):
  python rtx5070_feat_extraction.py --backbone uni
  python rtx5070_feat_extraction.py --backbone virchow
  python rtx5070_feat_extraction.py --backbone conch
  python rtx5070_feat_extraction.py --backbone all

Requirements:
  pip install torch torchvision transformers timm pillow scikit-image
  # For WILDS: pip install wilds

Notes:
  - RTX 5070 has 12GB VRAM, 672 GB/s bandwidth, 5th gen tensor cores
  - All models fit in 12GB for inference (FP16)
  - Uses FP16 (torch.cuda.amp) for speed
  - Saves features as .npz files
"""
import os
import sys
import time
import argparse
import collections
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io

# ============================================================
# Configuration
# ============================================================

BACKBONE_CONFIGS = {
    'uni': {
        'model_id': 'mahmoodlab/UNI',
        'feature_dim': 1024,
        'batch_size': 64,
        'input_size': 224,
        'type': 'timm',  # UNI uses timm interface
    },
    'virchow': {
        'model_id': 'paige-ai/Virchow',
        'feature_dim': 1024,
        'batch_size': 32,  # ViT-L, slightly larger
        'input_size': 224,
        'type': 'hf',  # HuggingFace transformers
    },
    'conch': {
        'model_id': 'MahmoodLab/CONCH',
        'feature_dim': 512,
        'batch_size': 128,
        'input_size': 448,  # CONCH uses 448x448
        'type': 'hf',
    },
}

# Dataset paths (adjust to your local setup)
DATA_DIR = os.environ.get('DATA_DIR', '/mnt/shared-workspace/shared/datasets')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './features')

DATASETS = {
    'camelyon17_id_train': {
        'path': os.path.join(DATA_DIR, 'camelyon17'),
        'split': 'id_train',
        'sample_size': 30000,  # subsample for manageable size
    },
    'camelyon17_id_val': {
        'path': os.path.join(DATA_DIR, 'camelyon17'),
        'split': 'id_val',
        'sample_size': 10000,
    },
    'camelyon17_ood_test': {
        'path': os.path.join(DATA_DIR, 'camelyon17'),
        'split': 'ood_test',
        'sample_size': 20000,
    },
    'pcam_train': {
        'path': os.path.join(DATA_DIR, 'pcam'),
        'split': 'train',
        'sample_size': 30000,
    },
    'pcam_test': {
        'path': os.path.join(DATA_DIR, 'pcam'),
        'split': 'test',
        'sample_size': 10000,
    },
}

# ============================================================
# Model Loading
# ============================================================

def load_uni_model(device):
    """Load UNI model (ViT-L, 304M params) from HuggingFace via timm."""
    import timm
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        num_classes=0,
    )
    model = model.to(device).half()  # FP16
    model.eval()
    
    # Get transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
    return model, transform, 1024


def load_virchow_model(device):
    """Load Virchow model (ViT-L, 300M params) from HuggingFace."""
    from transformers import AutoModel, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("paige-ai/Virchow")
    model = AutoModel.from_pretrained("paige-ai/Virchow")
    model = model.to(device).half()
    model.eval()
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
    return model, transform, 1024


def load_conch_model(device):
    """Load CONCH model (ViT-B, 86M params) from HuggingFace."""
    from transformers import AutoModel, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("MahmoodLab/CONCH")
    model = AutoModel.from_pretrained("MahmoodLab/CONCH")
    model = model.to(device).half()
    model.eval()
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
    return model, transform, 512


def load_backbone(name, device):
    """Load backbone model by name."""
    if name == 'uni':
        return load_uni_model(device)
    elif name == 'virchow':
        return load_virchow_model(device)
    elif name == 'conch':
        return load_conch_model(device)
    else:
        raise ValueError(f"Unknown backbone: {name}")


# ============================================================
# Data Loading
# ============================================================

def find_parquet_files(dataset_path, split_keyword):
    """Find parquet files for a given split."""
    import glob
    files = sorted(glob.glob(os.path.join(dataset_path, f"*{split_keyword}*.parquet")))
    if split_keyword == "train":
        files = [f for f in files if "train" in os.path.basename(f) and "validation" not in os.path.basename(f)]
    elif split_keyword == "test":
        files = [f for f in files if "test" in os.path.basename(f)]
    elif split_keyword == "validation":
        files = [f for f in files if "validation" in os.path.basename(f)]
    return files


def load_parquet_data(filepath, sample_size=None, balanced=True, seed=42):
    """Load images and labels from parquet file."""
    import pyarrow.parquet as pq
    t = pq.read_table(filepath, columns=['image', 'label'])
    df = t.to_pandas()
    n_total = len(df)
    
    if sample_size and sample_size < n_total:
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
            sel = rng.choice(n_total, sample_size, replace=False)
        df = df.iloc[sel].reset_index(drop=True)
    
    return df


def decode_image(image_bytes, target_size=224):
    """Decode image bytes to PIL Image."""
    if isinstance(image_bytes, dict):
        image_bytes = image_bytes.get('bytes', image_bytes)
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    return img


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(model, transform, images, device, batch_size=64, input_size=224):
    """Extract features from a list of PIL images."""
    all_features = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            # Transform images
            batch_tensors = torch.stack([transform(img) for img in batch_imgs])
            batch_tensors = batch_tensors.to(device).half()
            
            # Forward pass
            outputs = model(batch_tensors)
            
            # Extract CLS token or pooled features
            if hasattr(outputs, 'last_hidden_state'):
                # HuggingFace ViT: use CLS token
                feats = outputs.last_hidden_state[:, 0, :]
            elif isinstance(outputs, torch.Tensor):
                # timm model: direct output
                feats = outputs
            else:
                feats = outputs
            
            all_features.append(feats.float().cpu().numpy())
            
            if (i // batch_size) % 50 == 0:
                pct = min(i + batch_size, len(images)) / len(images) * 100
                print(f"    Batch {i//batch_size}/{len(images)//batch_size} ({pct:.0f}%)", flush=True)
    
    return np.concatenate(all_features, axis=0)


def process_dataset(model, transform, feature_dim, device, dataset_name, dataset_cfg,
                    backbone_name, output_dir, batch_size, input_size):
    """Process a single dataset split."""
    print(f"\n--- Processing {dataset_name} ---")
    
    files = find_parquet_files(dataset_cfg['path'], dataset_cfg['split'])
    if not files:
        print(f"  No parquet files found for {dataset_name}")
        return
    
    # Load data from all matching files
    all_images = []
    all_labels = []
    per_file = dataset_cfg['sample_size'] // max(len(files), 1)
    
    for fpath in files:
        df = load_parquet_data(fpath, sample_size=per_file, balanced=True)
        for _, row in df.iterrows():
            img = decode_image(row['image'], input_size)
            all_images.append(img)
            all_labels.append(row['label'])
    
    print(f"  Loaded {len(all_labels)} images, labels={collections.Counter(all_labels)}")
    
    # Extract features
    t0 = time.time()
    features = extract_features(model, transform, all_images, device,
                                 batch_size=batch_size, input_size=input_size)
    elapsed = time.time() - t0
    print(f"  Extracted features: {features.shape} in {elapsed:.1f}s")
    
    # Save
    outpath = os.path.join(output_dir, f"{dataset_name}.npz")
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(outpath,
                        features=features,
                        labels=np.array(all_labels))
    print(f"  Saved to {outpath}")
    
    # Clear memory
    del all_images
    torch.cuda.empty_cache()


# ============================================================
# WILDS MIL Feature Extraction
# ============================================================

def extract_wilds_mil_features(model, transform, feature_dim, device, backbone_name,
                                output_dir, batch_size, input_size, max_patches_per_slide=1000):
    """Extract features from WILDS CAMELYON17 patches for MIL, grouped by slide."""
    print(f"\n--- WILDS MIL Feature Extraction ---")
    
    try:
        from wilds import get_dataset
        import pandas as pd
    except ImportError:
        print("  WILDS not installed. Skipping MIL feature extraction.")
        print("  Install with: pip install wilds")
        return
    
    # Download WILDS dataset if needed
    wilds_root = os.environ.get('WILDS_DIR', './wilds_data')
    dataset = get_dataset("camelyon17", download=True, root_dir=wilds_root)
    
    # Load metadata
    meta_path = os.path.join(wilds_root, 'camelyon17_v1.0', 'metadata.csv')
    meta = pd.read_csv(meta_path)
    
    # Group by slide
    slides = meta.groupby('slide').agg(
        n_patches=('tumor', 'count'),
        center=('center', 'first'),
        patient=('patient', 'first'),
    ).reset_index()
    
    print(f"  {len(slides)} slides, {len(meta)} total patches")
    
    patch_dir = os.path.join(wilds_root, 'camelyon17_v1.0', 'patches')
    mil_output_dir = os.path.join(output_dir, 'mil')
    os.makedirs(mil_output_dir, exist_ok=True)
    
    rng = np.random.RandomState(42)
    
    for _, slide_info in slides.iterrows():
        slide_id = slide_info['slide']
        center = slide_info['center']
        
        # Get patches for this slide
        slide_meta = meta[meta['slide'] == slide_id].copy()
        
        # Subsample if needed
        if len(slide_meta) > max_patches_per_slide:
            slide_meta = slide_meta.sample(n=max_patches_per_slide, random_state=42)
        
        # Load images
        images = []
        patch_labels = []
        for _, row in slide_meta.iterrows():
            patient = str(row['patient']).zfill(3)
            node = row['node']
            x = row['x_coord']
            y = row['y_coord']
            patch_path = os.path.join(
                patch_dir,
                f'patient_{patient}_node_{node}',
                f'patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            )
            if os.path.exists(patch_path):
                img = Image.open(patch_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                patch_labels.append(row['tumor'])
        
        if not images:
            print(f"  Slide {slide_id}: no images found, skipping")
            continue
        
        # Extract features
        features = extract_features(model, transform, images, device,
                                     batch_size=batch_size, input_size=input_size)
        
        # Compute slide-level info
        tumor_fraction = np.mean(patch_labels)
        
        # Save
        outpath = os.path.join(mil_output_dir, f'slide_{slide_id}.npz')
        np.savez_compressed(outpath,
                            features=features,
                            patch_labels=np.array(patch_labels),
                            slide_id=slide_id,
                            center=center,
                            tumor_fraction=tumor_fraction)
        
        print(f"  Slide {slide_id}: {len(images)} patches, frac={tumor_fraction:.3f}, "
              f"center={center}, shape={features.shape}")
        
        del images
        torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='RTX 5070 Feature Extraction')
    parser.add_argument('--backbone', choices=['uni', 'virchow', 'conch', 'all'],
                        required=True, help='Which backbone to use')
    parser.add_argument('--output_dir', default='./features',
                        help='Output directory for features')
    parser.add_argument('--skip_mil', action='store_true',
                        help='Skip WILDS MIL feature extraction')
    parser.add_argument('--skip_datasets', action='store_true',
                        help='Skip standard dataset feature extraction')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: No GPU detected. Running on CPU (will be slow).")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    torch.set_num_threads(4)
    
    # Determine which backbones to process
    backbones = ['uni', 'virchow', 'conch'] if args.backbone == 'all' else [args.backbone]
    
    for bb_name in backbones:
        print(f"\n{'='*60}")
        print(f"Backbone: {bb_name.upper()}")
        print(f"{'='*60}")
        
        cfg = BACKBONE_CONFIGS[bb_name]
        output_dir = os.path.join(args.output_dir, bb_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        print(f"Loading {cfg['model_id']}...")
        t0 = time.time()
        model, transform, feature_dim = load_backbone(bb_name, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Loaded in {time.time()-t0:.1f}s, {n_params:.1f}M params, dim={feature_dim}")
        
        # Check VRAM usage
        if device.type == 'cuda':
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM used: {vram_used:.2f} GB")
        
        # Process standard datasets
        if not args.skip_datasets:
            for ds_name, ds_cfg in DATASETS.items():
                process_dataset(model, transform, feature_dim, device,
                               ds_name, ds_cfg, bb_name, output_dir,
                               cfg['batch_size'], cfg['input_size'])
        
        # Process WILDS for MIL
        if not args.skip_mil:
            extract_wilds_mil_features(model, transform, feature_dim, device,
                                       bb_name, output_dir,
                                       cfg['batch_size'], cfg['input_size'])
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        print(f"\n{bb_name.upper()} complete.")
    
    print(f"\n{'='*60}")
    print("ALL BACKBONES COMPLETE")
    print(f"{'='*60}")
    
    # Print summary
    for bb_name in backbones:
        bb_dir = os.path.join(args.output_dir, bb_name)
        if os.path.exists(bb_dir):
            files = [f for f in os.listdir(bb_dir) if f.endswith('.npz')]
            total_size = sum(os.path.getsize(os.path.join(bb_dir, f)) for f in files)
            print(f"  {bb_name}: {len(files)} files, {total_size/1e6:.1f} MB")


if __name__ == '__main__':
    main()
