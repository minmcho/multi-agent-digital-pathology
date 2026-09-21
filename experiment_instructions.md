# Experiment Execution Instructions
## RTX 5070 (Local) + Colab Pro — 1 Week, ~$10

---

## Overview

| Phase | Where | Time | Cost |
|-------|-------|------|------|
| 1. Data setup | Local | 30 min | $0 |
| 2. Feature extraction (UNI, Virchow, CONCH) | RTX 5070 | ~5h | $0 |
| 3. CONCH end-to-end fine-tuning | RTX 5070 | ~28h (3 days) | $0 |
| 4. LC25000 experiments | RTX 5070 | ~1.5h | $0 |
| 5. UNI end-to-end fine-tuning | Colab Pro (L4) | ~21h | ~$10 |
| 6. Virchow end-to-end fine-tuning | Colab Pro (L4) | ~21h | (same month) |
| 7. Upload results | Local | 15 min | $0 |

**Total: ~1 week, $10 (one month of Colab Pro)**

---

## Phase 1: Data Setup (Local, 30 min)

### 1a. Clone the repo

```bash
git clone https://github.com/minmcho/multi-agent-digital-pathology.git
cd multi-agent-digital-pathology
```

### 1b. Download CAMELYON17 parquet files

The CAMELYON17 data is on HuggingFace. Download it:

```bash
pip install huggingface_hub datasets pyarrow

python -c "
from huggingface_hub import snapshot_download
snapshot_download('wltjr1007/Camelyon17-WILDS', repo_type='dataset',
                  local_dir='./data/camelyon17_wilds')
"
```

This downloads ~9 GB of parquet files. The WILDS version has 40 slides from 4 centers (0, 2, 3, 4).

### 1c. Download PCam

PCam is also on HuggingFace:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('bryanluk/pcam', repo_type='dataset',
                  local_dir='./data/pcam')
"
```

If that doesn't work, use the GitHub version:
```bash
# Alternative: download from PCam GitHub
git clone https://github.com/basveeling/pcas.git
```

### 1d. Download Kather2016

```bash
# CRC-VAL-HE-7K (validation set, 1,974 images)
wget https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip
unzip CRC-VAL-HE-7K.zip -d ./data/kather2016
```

### 1e. Download LC25000

```bash
# Lung and colon cancer histopathological images
wget https://academictorrents.com/details/7de99c409c30f1c5793dffd75a76b93d7e7c93b8
# Or from Google Drive: https://drive.google.com/file/d/1rtDDqQRDFYFwZTBaGBjE5pYkZcXz6r7I/view
# Extract to ./data/lc25000
```

### 1f. Set up Python environment

```bash
# Create conda/venv environment
conda create -n pathology python=3.11 -y
conda activate pathology

# Core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers timm pillow scikit-learn scikit-image pandas numpy
pip install huggingface_hub datasets pyarrow
pip install matplotlib seaborn

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Should show: CUDA: True, GPU: NVIDIA GeForce RTX 5070
```

### 1g. Set environment variables

```bash
export DATA_DIR=./data
export OUTPUT_DIR=./features
export E2E_RESULTS_DIR=./e2e_results
export LC25000_DIR=./data/lc25000
```

---

## Phase 2: Feature Extraction (RTX 5070, ~5h)

Extract frozen features from UNI (ViT-L), Virchow (ViT-L), and CONCH (ViT-B) for all datasets.

```bash
# Run all three backbones (or run them one at a time)
python scripts/rtx5070_feat_extraction.py --backbone all

# Or individually:
python scripts/rtx5070_feat_extraction.py --backbone uni      # ~1.5h
python scripts/rtx5070_feat_extraction.py --backbone virchow   # ~1.5h
python scripts/rtx5070_feat_extraction.py --backbone conch     # ~2h (448x448 input)
```

**What this does:**
- Loads each foundation model in FP16 (fits in 12GB VRAM)
- Extracts features from CAMELYON17 (ID train 30K, ID val 10K, OOD test 20K)
- Extracts features from PCam (train 30K, test 10K)
- Extracts features from Kather2016 (1,974 images)
- Saves as `.npz` files to `./features/{backbone}/`

**Expected output:**
```
features/
  uni/
    camelyon17_id_train.npz
    camelyon17_id_val.npz
    camelyon17_ood_test.npz
    pcam_train.npz
    pcam_test.npz
    kather2016.npz
  virchow/
    ... (same structure)
  conch/
    ... (same structure)
```

**Upload results when done:**
Zip the features folder and upload to the shared workspace, or push to a release on GitHub.

---

## Phase 3: CONCH End-to-End Fine-Tuning (RTX 5070, ~28h over 3 days)

This is the longest local run. CONCH (ViT-B, 86M params) fits in 12GB with batch 32.

```bash
# Run all models with 5 seeds (baseline, DANN, MixStyle)
python scripts/rtx5070_conch_e2e.py --model all --seeds 5

# Or run one model at a time:
python scripts/rtx5070_conch_e2e.py --model baseline --seeds 5   # ~9h
python scripts/rtx5070_conch_e2e.py --model dann --seeds 5        # ~10h
python scripts/rtx5070_conch_e2e.py --model mixstyle --seeds 5    # ~9h
```

**What this does:**
- Fine-tunes CONCH end-to-end (all layers, not frozen)
- 15 epochs, LR 1e-5, batch 32, FP16
- Tests on CAMELYON17 OOD + PCam external
- 5 random seeds per model
- Saves results to `./e2e_results/conch/`

**Tips:**
- You can run this overnight. Each model×seed takes ~1.5-2h.
- Monitor GPU temperature: `nvidia-smi -l 5`
- If you get OOM, reduce batch size: edit `BATCH_SIZE = 16` in the script

---

## Phase 4: LC25000 Experiments (RTX 5070, ~1.5h)

Cross-cancer transfer to lung and colon adenocarcinoma.

```bash
python scripts/lc25000_experiments.py
```

**What this does:**
- Extracts features from LC25000 (lung + colon) using all 6 backbones
- Runs frozen-feature experiments (5 models × 10 seeds)
- Tests cross-cancer transfer from CAMELYON17-trained models

---

## Phase 5: UNI End-to-End (Colab Pro, ~21h on L4)

UNI is ViT-L (304M params) — too large for RTX 5070's 12GB for end-to-end training. Use Colab Pro.

### 5a. Set up Colab notebook

1. Go to colab.research.google.com
2. Create a new notebook
3. Set runtime: **Runtime → Change runtime type → L4 GPU** (or A100 if available)
4. Run the following setup cell:

```python
# Cell 1: Setup
!pip install torch torchvision transformers timm pillow scikit-learn pandas numpy pyarrow huggingface_hub

import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
# Should show L4 or A100
```

### 5b. Upload data to Colab

Option A: Upload parquet files to Google Drive, mount in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
# Copy data from Drive
!cp /content/drive/MyDrive/pathology_data/camelyon17/*.parquet ./data/camelyon17/
!cp /content/drive/MyDrive/pathology_data/pcam/*.parquet ./data/pcam/
```

Option B: Download directly in Colab (faster, uses Colab's bandwidth):
```python
from huggingface_hub import snapshot_download
snapshot_download('wltjr1007/Camelyon17-WILDS', repo_type='dataset', local_dir='./data/camelyon17_wilds')
# For PCam, upload to your Drive or use an alternative source
```

### 5c. Paste and run the UNI e2e script

Paste the entire contents of `scripts/colab_uni_e2e.py` into a Colab cell and run.

The script auto-detects the GPU (L4/A100/T4) and adjusts batch size accordingly:
- L4 (24GB): batch 16
- A100 (40GB): batch 32
- T4 (16GB): batch 8 (with gradient checkpointing)

**Expected time on L4:** ~21h for 3 models × 5 seeds
**Compute units used:** ~36 CU (L4 at 1.7 CU/h × 21h)

### 5d. Download results

```python
# Save results to Drive
!cp -r ./e2e_results/uni/ /content/drive/MyDrive/pathology_results/
```

---

## Phase 6: Virchow End-to-End (Colab Pro, ~21h on L4)

Same as Phase 5 but with Virchow instead of UNI.

### 6a. In the same Colab notebook (or a new one):

Paste the entire contents of `scripts/colab_virchow_e2e.py` into a cell and run.

**Expected time on L4:** ~21h for 3 models × 5 seeds
**Compute units used:** ~36 CU

**Total Colab CU used:** ~72 CU (fits within 100 CU/month on Pro)

### 6b. Download results

```python
!cp -r ./e2e_results/virchow/ /content/drive/MyDrive/pathology_results/
```

---

## Phase 7: Upload All Results

After all experiments complete, upload the results so they can be integrated into the article:

### 7a. Zip all results

```bash
# On your local machine:
cd multi-agent-digital-pathology

# Features
tar czf features_new_backbones.tar.gz features/uni features/virchow features/conch

# E2E results
tar czf e2e_results.tar.gz e2e_results/

# LC25000 results
tar czf lc25000_results.tar.gz lc25000_results/
```

### 7b. Upload to GitHub release

```bash
# Create a release with the results attached
gh release create v1.0-experimental-results \
  features_new_backbones.tar.gz \
  e2e_results.tar.gz \
  lc25000_results.tar.gz \
  --title "Experimental Results" \
  --notes "Feature extraction + e2e fine-tuning results for all 6 backbones"
```

Or upload to Google Drive and share the link.

---

## Troubleshooting

### RTX 5070 OOM
- Reduce batch size in the script (`BATCH_SIZE` variable)
- Enable gradient checkpointing: add `model.gradient_checkpointing_enable()`
- Use FP16: ensure `.half()` is called on the model

### Colab timeout
- Colab Pro sessions can run up to 24h
- Save checkpoints frequently (the scripts do this)
- If disconnected, reconnect and resume from last checkpoint

### HuggingFace model access
- UNI (`MahmoodLab/UNI`) and CONCH (`MahmoodLab/CONCH`) require HF account and model access approval
- Request access at: https://huggingface.co/MahmoodLab/UNI
- Virchow (`paige-ai/Virchow`) is publicly available
- After approval, run `huggingface-cli login` and enter your HF token

### Data format issues
- CAMELYON17 WILDS parquets have columns: `image`, `label`, `center`, `image_id`, `patient`, `node`, `x_coord`, `y_coord`, `slide`
- Images are stored as bytes in the `image` column (PIL.Image.open(io.BytesIO(bytes)))
- Patches are RGBA — scripts convert to RGB automatically

---

## Summary Checklist

- [ ] Clone repo
- [ ] Download CAMELYON17, PCam, Kather2016, LC25000 data
- [ ] Set up Python environment with CUDA
- [ ] **Phase 2:** Run feature extraction (UNI, Virchow, CONCH) — ~5h
- [ ] **Phase 3:** Run CONCH e2e (3 models × 5 seeds) — ~28h
- [ ] **Phase 4:** Run LC25000 experiments — ~1.5h
- [ ] **Phase 5:** Colab Pro: UNI e2e (3 models × 5 seeds) — ~21h
- [ ] **Phase 6:** Colab Pro: Virchow e2e (3 models × 5 seeds) — ~21h
- [ ] **Phase 7:** Upload all results
- [ ] Notify when done so article can be updated with new results
