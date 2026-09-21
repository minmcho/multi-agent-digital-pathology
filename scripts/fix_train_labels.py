"""
Post-processing: fix labels in train npz files where labels weren't repeated per domain.
The buggy extraction code saved 30K labels for 90K features (3 HED domains).
This script repeats the labels to match the features.

Usage:
  python fix_train_labels.py --backbone vit_tiny
  python fix_train_labels.py --backbone resnet18
"""
import numpy as np
import os, argparse, subprocess, collections

SHARED_DIR = "/mnt/shared-workspace/shared"
TMP_DIR = "/workspace/_feat_tmp"

def fix_labels(backbone):
    feature_dir = os.path.join(SHARED_DIR, "features_v2", backbone)
    train_path = os.path.join(feature_dir, "camelyon17_id_train.npz")
    
    if not os.path.exists(train_path):
        print(f"  {train_path} not found, skipping")
        return
    
    data = np.load(train_path)
    features = data['features']
    labels = data['labels']
    domains = data['domains']
    
    print(f"  Loaded: features={features.shape}, labels={labels.shape}, domains={domains.shape}")
    
    if len(labels) == len(features):
        print(f"  Labels already correct ({len(labels)} == {len(features)}), no fix needed")
        return
    
    if len(labels) * 3 == len(features):
        print(f"  Fixing: repeating labels {len(labels)} -> {len(features)}")
        labels_fixed = np.tile(labels, 3)
        print(f"  Fixed labels: {collections.Counter(labels_fixed.tolist())}")
        print(f"  Domains: {collections.Counter(domains.tolist())}")
        
        # Save to local disk, then cp to S3 (delete existing first — S3 FUSE can't overwrite)
        os.makedirs(TMP_DIR, exist_ok=True)
        tmp_path = os.path.join(TMP_DIR, "camelyon17_id_train_fixed.npz")
        np.savez_compressed(tmp_path, features=features, labels=labels_fixed, domains=domains)
        try:
            os.remove(train_path)
        except Exception:
            subprocess.run(['rm', '-f', train_path])
        subprocess.run(['cp', tmp_path, train_path], check=True)
        os.remove(tmp_path)
        print(f"  Saved fixed file to {train_path}")
        
        # Verify
        data2 = np.load(train_path)
        print(f"  Verified: features={data2['features'].shape}, labels={data2['labels'].shape}, domains={data2['domains'].shape}")
    else:
        print(f"  ERROR: labels length {len(labels)} is not 1/3 of features length {len(features)}")
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, choices=['vit_tiny', 'resnet18', 'vit_b'])
    args = parser.parse_args()
    fix_labels(args.backbone)

if __name__ == "__main__":
    main()
