**Table 1.** Dataset summary.

| Dataset | Source | Task | Image Size | Train | Val | Test | Domain Role |
|---------|--------|------|-----------|-------|-----|------|-------------|
| Camelyon17 ID | Bandi et al. [1] | Binary tumour detection | 96×96 | 30,000 | 10,000 | — | Domains 0–2 (original + HED augmentations) |
| PCam | Veeling et al. [3] | Binary tumour detection | 96×96 | 30,000 | 10,000 | 10,000 | Domain 3 (real cross-dataset) |
| Camelyon17 OOD | Bandi et al. [1] | Binary tumour detection | 96×96 | — | — | 20,000 | Unseen medical centre (OOD test) |
| Kather2016 (CRC-VAL-HE-7K) | Kather et al. [K1] | TUM vs NORM | 224×224 | — | — | 1,974 | Cross-cancer-type OOD test |

**Note:** Camelyon17 ID training (30,000) was augmented with HED stain deconvolution to create 3 synthetic domains (90,000 total). Combined with PCam training (30,000), the total training set was 120,000 samples across 4 domains. Kather2016 CRC-VAL-HE-7K is the official held-out validation set from the NCT-CRC-HE-100K collection, used here as a cross-cancer-type OOD test (colorectal adenocarcinoma vs normal colon mucosa).
