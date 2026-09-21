# Multi-Agent Digital Pathology: Foundation Model Quality Sets the Generalization Ceiling

A systematic multi-backbone benchmark of architectural interventions for robust digital pathology diagnosis.

## Paper

**Title:** Foundation Model Quality Sets the Generalization Ceiling in Digital Pathology: A Systematic Multi-Backbone Benchmark of Architectural Interventions

**Target:** British Journal of Biomedical Science (BJBS) Special Issue: Digital Pathology and Artificial Intelligence

## Repository Structure

```
multi-agent-digital-pathology/
├── README.md
├── scripts/                    # Pipeline scripts (existing + new)
│   ├── pathology_models.py     # Core model architectures
│   ├── pathology_models_v3.py  # Extended models (MixStyle, FACT, orchestrator variants)
│   ├── extract_phikon_features.py    # Feature extraction (existing backbones)
│   ├── run_experiments_v3.py   # Experiment runner (5 models × 10 seeds)
│   ├── rtx5070_feat_extraction.py    # NEW: Foundation model feature extraction
│   ├── rtx5070_conch_e2e.py    # NEW: CONCH end-to-end fine-tuning
│   ├── colab_uni_e2e.py        # NEW: UNI end-to-end (Colab Pro)
│   ├── colab_virchow_e2e.py    # NEW: Virchow end-to-end (Colab Pro)
│   ├── mil_pipeline.py         # NEW: WSI-level MIL (ABMIL + CLAM)
│   ├── run_frozen_features.py  # NEW: Frozen-feature experiments on new backbones
│   └── lc25000_experiments.py  # NEW: LC25000 lung/colon experiments
├── article/                    # Manuscript
│   ├── article.md              # BJBS-compliant article
│   ├── article.docx            # Word version
│   └── plan.md                 # Implementation plan
├── figures/                    # All figures (PNG + SVG)
├── tables/                     # All tables (Markdown)
└── features/                   # Feature storage (gitignored)
```

## Compute Setup

| Resource | Hardware | Role | Cost |
|---|---|---|---|
| Local GPU | RTX 5070 (12GB GDDR7) | Feature extraction, CONCH e2e, LC25000 e2e | $0 |
| Cloud GPU | Colab Pro (L4 24GB / A100 40GB) | UNI + Virchow end-to-end | $10/month |
| CPU | Any | Frozen-feature experiments, MIL training | $0 |

## Backbones

| Backbone | Architecture | Pre-training | Feature Dim |
|---|---|---|---|
| ResNet-18 | CNN | ImageNet | 512 |
| ViT-Tiny (Phikon) | ViT-T | Kather2016 (SSL) | 192 |
| ViT-B (Phikon) | ViT-B | TCGA (SSL) | 768 |
| CONCH | ViT-B | Pathology (CLIP) | 512 |
| UNI | ViT-L | 100M pathology tiles (SSL) | 1024 |
| Virchow | ViT-L | 1.5M WSI (SSL) | 1024 |

## Models

1. **Baseline** — Linear/MLP classifier on frozen features
2. **DANN** — Domain-Adversarial Neural Network
3. **Disentanglement** — Bio/domain disentangled representations
4. **Multi-Agent** — 6 specialist agents + orchestrator fusion
5. **MixStyle** — Instance-level feature statistic mixing

## Datasets

| Dataset | Cancer Type | Patches | Centers |
|---|---|---|---|
| CAMELYON17 | Breast (lymph node) | 455,954 | 5 |
| PCam | Breast (lymph node) | 50,000 | — |
| Kather2016 | Colorectal | 5,000 | — |
| LC25000 | Lung + Colon | 25,000 | — |

## Reproduction

### 1. Feature Extraction (RTX 5070, ~5h)
```bash
python scripts/rtx5070_feat_extraction.py --backbone uni
python scripts/rtx5070_feat_extraction.py --backbone virchow
python scripts/rtx5070_feat_extraction.py --backbone conch
```

### 2. Frozen-Feature Experiments (CPU, ~6h)
```bash
python scripts/run_frozen_features.py --backbone uni
python scripts/run_frozen_features.py --backbone virchow
python scripts/run_frozen_features.py --backbone conch
```

### 3. WSI-Level MIL (CPU, ~6h)
```bash
python scripts/mil_pipeline.py --backbone vit_b --model abmil
python scripts/mil_pipeline.py --backbone vit_b --model clam
```

### 4. End-to-End Fine-Tuning
```bash
# RTX 5070 (CONCH, ~28h)
python scripts/rtx5070_conch_e2e.py --model baseline --seeds 5

# Colab Pro (UNI/Virchow, ~21h each)
# Paste colab_uni_e2e.py into Colab notebook
# Paste colab_virchow_e2e.py into Colab notebook
```

### 5. LC25000 (RTX 5070 + CPU, ~5h)
```bash
python scripts/lc25000_experiments.py
```

## License

MIT
