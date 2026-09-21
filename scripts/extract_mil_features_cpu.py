"""
CPU-based MIL Feature Extraction — Optimized version
Reads each parquet file once, extracts all needed patches.
"""
import os, sys, time, glob, collections
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
import io

PARQUET_DIR = "/workspace/hf_cache"
META_PATH = "/mnt/shared-workspace/shared/wilds_metadata.csv"
OUTPUT_DIR = "/mnt/shared-workspace/shared/mil_features"
MAX_PATCHES_PER_SLIDE = 1000
SEED = 42

def main():
    device = torch.device('cpu')
    torch.set_num_threads(4)
    
    # Load metadata
    meta = pd.read_csv(META_PATH)
    print(f"Metadata: {len(meta)} patches, {meta['slide'].nunique()} slides")
    
    # Sample patches per slide
    sampled = []
    for slide_id in sorted(meta['slide'].unique()):
        slide_meta = meta[meta['slide'] == slide_id]
        n = min(len(slide_meta), MAX_PATCHES_PER_SLIDE)
        sampled.append(slide_meta.sample(n=n, random_state=SEED))
    sampled_meta = pd.concat(sampled, ignore_index=False)
    print(f"Sampled: {len(sampled_meta)} patches across {sampled_meta['slide'].nunique()} slides")
    
    # Group by source file — read each file ONCE
    parquet_files = sorted(glob.glob(f"{PARQUET_DIR}/**/data/*.parquet", recursive=True))
    file_map = {os.path.basename(f): f for f in parquet_files}
    
    # Build a lookup: for each source file, which row indices do we need?
    file_to_indices = {}
    for idx, row in sampled_meta.iterrows():
        fname = row['source_file']
        if fname not in file_to_indices:
            file_to_indices[fname] = []
        file_to_indices[fname].append((row['row_idx'], idx))
    
    print(f"Need to read {len(file_to_indices)} parquet files")
    
    # Load all needed images, organized by slide
    slide_images = {}  # slide_id -> list of (image, label, center)
    
    for fname, indices in file_to_indices.items():
        fpath = file_map.get(fname)
        if not fpath:
            continue
        print(f"  Loading {fname} ({len(indices)} patches)...")
        t0 = time.time()
        
        # Read image column once
        t = pq.read_table(fpath, columns=['image'])
        all_images = t.column('image').to_pylist()
        
        for row_idx, meta_idx in indices:
            row = sampled_meta.loc[meta_idx]
            img_bytes = all_images[row_idx]
            if isinstance(img_bytes, dict):
                img_bytes = img_bytes['bytes']
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            slide_id = row['slide']
            if slide_id not in slide_images:
                slide_images[slide_id] = {'images': [], 'labels': [], 'center': int(row['center'])}
            slide_images[slide_id]['images'].append(img)
            slide_images[slide_id]['labels'].append(int(row['label']))
        
        print(f"    Done in {time.time()-t0:.1f}s")
    
    print(f"\nLoaded images for {len(slide_images)} slides")
    
    # Process each backbone
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    backbones = [
        ('resnet18', 'resnet18', 512),
        ('vit_tiny', 'vit_tiny', 192),
        ('vit_b', 'vit_b', 768),
    ]
    
    for bb_name, bb_key, feat_dim in backbones:
        print(f"\n{'='*60}")
        print(f"Backbone: {bb_name.upper()} (dim={feat_dim})")
        print(f"{'='*60}")
        
        bb_dir = os.path.join(OUTPUT_DIR, bb_name, 'mil')
        os.makedirs(bb_dir, exist_ok=True)
        
        # Load model
        print(f"  Loading model...")
        t0 = time.time()
        
        if bb_name == 'resnet18':
            import torchvision.models as models
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Identity()
        elif bb_name == 'vit_tiny':
            from transformers import AutoModel
            model = AutoModel.from_pretrained('1aurent/phikon-distil-vit-tiny-patch16-224-kather2016')
        elif bb_name == 'vit_b':
            from transformers import AutoModel
            model = AutoModel.from_pretrained('owkin/phikon')
        
        model = model.to(device).eval()
        print(f"  Loaded in {time.time()-t0:.1f}s")
        
        # Extract features per slide
        for slide_id in sorted(slide_images.keys()):
            slide_file = f'slide_{slide_id}.npz'
            outpath = os.path.join(bb_dir, slide_file)
            if os.path.exists(outpath):
                print(f"  Slide {slide_id}: already exists, skipping")
                continue
            
            data = slide_images[slide_id]
            images = data['images']
            labels = data['labels']
            center = data['center']
            
            t1 = time.time()
            with torch.no_grad():
                batch_size = 16
                all_feats = []
                for i in range(0, len(images), batch_size):
                    batch = images[i:i+batch_size]
                    tensors = torch.stack([transform(img) for img in batch]).to(device)
                    outputs = model(tensors)
                    if hasattr(outputs, 'last_hidden_state'):
                        feats = outputs.last_hidden_state[:, 0, :]
                    else:
                        feats = outputs
                    all_feats.append(feats.cpu().numpy())
            
            features = np.concatenate(all_feats, axis=0)
            elapsed = time.time() - t1
            tumor_frac = np.mean(labels)
            
            np.savez_compressed(outpath,
                                features=features.astype(np.float32),
                                patch_labels=np.array(labels),
                                slide_id=slide_id,
                                center=center,
                                tumor_fraction=tumor_frac)
            
            print(f"  Slide {slide_id}: {len(images)} patches, frac={tumor_frac:.3f}, "
                  f"center={center}, {elapsed:.1f}s")
        
        del model
        print(f"  {bb_name.upper()} complete!")
    
    print(f"\n{'='*60}")
    print("ALL BACKBONES COMPLETE")
    print(f"{'='*60}")
    
    # Summary
    for bb_name, _, _ in backbones:
        bb_dir = os.path.join(OUTPUT_DIR, bb_name, 'mil')
        if os.path.exists(bb_dir):
            files = [f for f in os.listdir(bb_dir) if f.endswith('.npz')]
            total_size = sum(os.path.getsize(os.path.join(bb_dir, f)) for f in files)
            print(f"  {bb_name}: {len(files)} slides, {total_size/1e6:.1f} MB")

if __name__ == '__main__':
    main()
