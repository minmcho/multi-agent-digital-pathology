# Plan: Roadmap to 95% BJBS Acceptance — RTX 5070 + Colab Pro

## Summary

Transform the current article from ~60-70% to ~95% acceptance probability by executing the four experiments the article itself identifies as "future work" and "limitations": (1) foundation model backbones (UNI, Virchow, CONCH), (2) WSI-level MIL (ABMIL + CLAM), (3) end-to-end fine-tuning, and (4) a third dataset (LC25000 lung/colon). Compute uses RTX 5070 (12GB GDDR7, 672 GB/s, 5th gen tensor cores) for 80% of work locally + Colab Pro ($10/month) for ViT-L end-to-end training + CPU sandbox for frozen-feature experiments and MIL training. Article improvements run in parallel. Timeline: **1 week, $10 total**. GitHub repo created for reproducibility and journal submission.

## Current State

- **Article**: BJBS-compliant (12/12 formatting), 4,925 words, 54 refs, balanced framing
- **Experiments**: 3 backbones (ResNet-18, ViT-Tiny, ViT-B), 5 models, 10 seeds, 8 RQs
- **Datasets**: CAMELYON17, PCam, Kather2016
- **Statistical rigor**: Power analysis, bootstrap CIs, DeLong tests, per-class analysis
- **Article's own identified gaps** (from Limitations and Future Directions sections):
  - "WSI-level evaluation with attention-based aggregation" → future work
  - "Evaluation with larger foundation models (UNI, Virchow, CHIEF)" → future work
  - "96×96 pixel patches rather than full WSIs" → limitation
  - "Frozen feature extraction rather than end-to-end fine-tuning" → limitation

## What 95% Requires (and Why)

| Gap | Why reviewers will demand it | What we do |
|---|---|---|
| Foundation models | UNI/Virchow/CONCH are standard by 2026. A pathology AI paper without them is immediately suspect. | Extract features, rerun all 5 models × 10 seeds on 3 new backbones (CPU, no GPU needed) |
| WSI-level analysis | Clinical pathology operates at slide level. Patch-only is insufficient for clinical claims. | ABMIL + CLAM on WILDS CAMELYON17, 50 slides, attention-based aggregation |
| End-to-end fine-tuning | Reviewers ask "would results change with full fine-tuning?" Need empirical answer. | Baseline + DANN + MixStyle for CONCH (ViT-B) and UNI (ViT-L), 3-5 seeds |
| Third dataset/organ | Only breast + CRC. Need lung/colon for generalizability. | LC25000 (lung + colon, 25K patches), frozen features + CONCH e2e |
| No pathologist co-author | Weakens clinical credibility. Must compensate with stronger literature grounding. | Comprehensive clinical utility analysis, literature-cited clinical relevance, comparison to published pathologist studies |

## Resources

| Resource | Capacity | Role | Cost |
|---|---|---|---|
| RTX 5070 (local) | 12GB GDDR7, 672 GB/s, 5th gen tensor cores, unlimited | Feature extraction, CONCH e2e, LC25000 e2e, MIL features | $0 |
| Colab Pro | 100 CU/month, L4 (24GB) or A100 (40GB), 24h sessions | UNI + Virchow end-to-end (72 CU on L4) | $10/month |
| Sandbox CPU | Unlimited | Frozen-feature experiments, MIL training, article work | $0 |
| GitHub repo | Public | Code, scripts, reproducibility, journal submission | $0 |

## Week-by-Week Schedule (Compressed: 1 Week)

### Day 1: Feature Extraction + Start E2E
**RTX 5070 (~5h):**
- Feature extraction: UNI + Virchow + CONCH on CAMELYON17 + PCam + Kather2016 (~4.5h)
- MIL feature extraction from WILDS patches (1000 patches/slide, 3 existing backbones, ~0.8h)

**Colab Pro (launch, runs in background):**
- Start UNI end-to-end: baseline + DANN + MixStyle, 3 seeds, CAMELYON17 (~21h on L4)

**CPU sandbox (~6h, starts after features ready):**
- Rerun all 5 models × 10 seeds on UNI features (~2h)
- Rerun all 5 models × 10 seeds on Virchow features (~2h)
- Rerun all 5 models × 10 seeds on CONCH features (~2h)

**Article (parallel):**
- Update Results tables with 3 new backbones
- Update abstract to mention 6 backbones

### Day 2-4: CONCH E2E + MIL + LC25000
**RTX 5070 (~30h, overnight + weekend):**
- CONCH end-to-end: baseline + DANN + MixStyle, 5 seeds, CAMELYON17 + PCam (~28.5h)
- LC25000 feature extraction (6 backbones, 25K patches) (~1h)
- LC25000 CONCH e2e: baseline, 3 seeds (~1.5h)

**Colab Pro (continues):**
- UNI e2e finishes (~21h on L4, done by Day 2)
- Start Virchow e2e: baseline + DANN + MixStyle, 3 seeds, CAMELYON17 (~21h on L4)

**CPU sandbox (~10h):**
- MIL training: ABMIL + CLAM, 6 backbones × 10 seeds (~4h)
- LC25000 frozen-feature experiments: 5 models × 10 seeds × 6 backbones (~3h)
- MIL feature extraction for 3 new backbones (UNI, Virchow, CONCH) (~2h)

**Article (parallel):**
- Write WSI-Level MIL results section
- Add LC25000 to cross-dataset transfer section
- Create MIL figure (slide-level performance by backbone)

### Day 5-7: Virchow E2E + Statistics + Article
**Colab Pro (finishes):**
- Virchow e2e completes (~21h on L4, done by Day 5)

**CPU sandbox (~5h):**
- Update power analysis with 6-backbone data (~1h)
- Bootstrap CIs for all new comparisons (~1h)
- DeLong tests for foundation model comparisons (~1h)
- MIL statistical analysis (~1h)
- Final article integration (~1h)

**Article (major work):**
- Restructure figures (merge to stay within 10 limit)
- Add end-to-end fine-tuning results section
- Rewrite Discussion with all new findings
- Update Limitations (remove addressed items)
- Update references (~15-20 new)
- Final BJBS compliance check
- Generate final DOCX
- Prepare cover letter + reviewer response template

### Day 7+: Buffer
- Any failed runs repeated on RTX 5070 or Colab
- Final article polish

## Compute Budget Summary

| Resource | Hours | Cost |
|---|---|---|
| RTX 5070 (local) | 35h | $0 |
| Colab Pro (1 month) | 42h on L4 (72 CU of 100) | $10 |
| CPU sandbox | 15h | $0 |
| **Total** | **~1 week wall-clock** | **$10** |

## GitHub Repository

Create a public GitHub repo for reproducibility and journal submission:
- Repo name: `multi-agent-digital-pathology` (or user preference)
- Structure: single repo with all code, scripts, and article
- Contains: feature extraction scripts, MIL pipeline, experiment runner, article source, figures
- README with setup instructions for RTX 5070 + Colab Pro workflow

## Article Changes Summary

### New Results Sections
1. **Foundation Model Backbones** (Week 1-2): 6-backbone comparison (ResNet-18 → ViT-Tiny → ViT-B → CONCH → UNI → Virchow), all 5 models, 10 seeds, frozen features. Shows the quality ceiling effect across the full spectrum.
2. **WSI-Level MIL Analysis** (Week 1-2): ABMIL + CLAM on 50 WILDS CAMELYON17 slides. Patch-level AUC + slide-level tumor fraction correlation. Shows backbone quality effect persists at slide level.
3. **End-to-End Fine-Tuning** (Week 4-5): CONCH + UNI + Virchow, baseline + DANN + MixStyle, 3-5 seeds. Shows the ceiling moves with fine-tuning but backbone ranking persists.
4. **Third Dataset (LC25000)** (Week 2, 5): Lung + colon cancer, 6 backbones frozen features + CONCH e2e. Shows cross-organ generalizability.

### Updated Sections
- **Abstract**: Reflect 6 backbones, MIL, e2e, 3 datasets (still 350 words)
- **Introduction**: Strengthen foundation model context with 2025-2026 references
- **Materials and Methods**: Add MIL architectures, e2e training protocol, LC25000 description
- **Results**: All tables updated with 6 backbones; new MIL and e2e subsections
- **Discussion**: Integrate new findings; compare to published SOTA; strengthen clinical relevance
- **Limitations**: Remove addressed limitations (frozen features, patch-only, no foundation models); add new ones (no pathologist co-author, MIL on subsampled patches, e2e limited seeds for ViT-L)
- **Future Directions**: Update — most prior future work now done; new directions (larger MIL datasets, clinical deployment)
- **Summary table**: Update "what this work adds" bullets
- **Concluding statement**: Update to reflect broader findings (still ≤200 chars)
- **References**: Add ~15-20 new references (2025-2026 foundation model, MIL, clinical pathology papers)

### Figure/Table Restructuring (stay within 10 limit)
- **Fig 1**: Multi-agent architecture (keep)
- **Fig 2**: Diagnostic pipeline (keep)
- **Fig 3**: Diagnostic performance — updated to 6 backbones (keep, update)
- **Fig 4**: Perturbation robustness (keep)
- **Fig 5**: Calibration (keep)
- **Fig 6**: Combined: disentanglement ablation + clinical utility (merge old Fig 6+7)
- **Fig 7**: NEW — MIL slide-level results (ABMIL vs CLAM, backbone comparison, OOD gap)
- **Fig 8**: NEW — End-to-end vs frozen comparison (ceiling moves but ranking persists)
- **Table 1**: Dataset summary (keep, add LC25000)
- **Table 2**: Main results — updated to 6 backbones (keep, update)
- **Table 3**: Cross-dataset transfer — updated with LC25000 (keep, update)
- **Table 4**: NEW — MIL + end-to-end summary
- Total: 8 figures + 4 tables = 12... need to reduce to 10

**Revised plan**: 7 figures + 3 tables = 10
- Merge Fig 3 + Fig 8 into one (frozen + e2e side by side)
- Merge Table 2 + Table 4 into one (all results in one comprehensive table)
- Final: 7 figs + 3 tables = 10

## Kaggle/Colab Notebook Structure

Prepare 2 Colab notebooks + 4 local scripts (paste-and-run):

1. **`colab_uni_e2e.py`** (Colab Pro, ~21h on L4)
   - Load UNI model (ViT-L, FP16, from HuggingFace)
   - Fine-tune baseline + DANN + MixStyle on CAMELYON17
   - 3 seeds, checkpoint every epoch
   - Save results to Google Drive / download

2. **`colab_virchow_e2e.py`** (Colab Pro, ~21h on L4)
   - Same as UNI but with Virchow model

3. **`rtx5070_feat_extraction.py`** (local RTX 5070, ~5h)
   - Load UNI, Virchow, CONCH from HuggingFace
   - Extract features for CAMELYON17, PCam, Kather2016, WILDS patches
   - Save features as .npz files

4. **`rtx5070_conch_e2e.py`** (local RTX 5070, ~28.5h)
   - Load CONCH model
   - Fine-tune baseline + DANN + MixStyle on CAMELYON17 + PCam
   - 5 seeds, checkpoint every epoch

5. **`mil_pipeline.py`** (CPU sandbox, ~6h)
   - WILDS download, MIL feature extraction, ABMIL + CLAM training, evaluation

6. **`run_frozen_features.py`** (CPU sandbox, ~6h)
   - Rerun all 5 models × 10 seeds on new backbone features

## Acceptance Criteria

The plan is complete when:
1. ✅ 6 backbones (3 existing + UNI + Virchow + CONCH) have frozen-feature results for all 5 models × 10 seeds
2. ✅ MIL (ABMIL + CLAM) results on 50 WILDS CAMELYON17 slides for all 6 backbones
3. ✅ End-to-end fine-tuning: CONCH (3 models × 5 seeds), UNI (3 models × 3-5 seeds), Virchow (3 models × 3-5 seeds)
4. ✅ LC25000 (lung + colon) frozen-feature results for 6 backbones + CONCH e2e
5. ✅ All statistical analyses updated (power, bootstrap CIs, DeLong)
6. ✅ Article updated with all new results, BJBS-compliant (12/12), ≤10 figures/tables
7. ✅ Final DOCX generated
8. ✅ Cover letter and reviewer response template prepared

## Assumptions

1. Kaggle T4 x2 remains available at 30h/week throughout (stable policy as of 2026)
2. WILDS CAMELYON17 download URL remains active
3. UNI, Virchow, CONCH models remain available on HuggingFace
4. LC25000 dataset is accessible (publicly available on Kaggle/academic sites)
5. User can commit ~30 min/week to monitor Kaggle runs and restart sessions
6. User can commit ~2-3h/week to review article changes
7. Sandbox CPU is available for frozen-feature experiments and MIL training
8. No pathologist co-author — compensate with literature and clinical utility analysis
9. 7 weeks is acceptable timeline (user confirmed 4-8 weeks)
10. Figure/table restructuring stays within BJBS 10-item limit

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Kaggle session timeout (12h limit) | Checkpoint every epoch; resume from last checkpoint |
| Kaggle quota exhaustion | Budget 141-171h of 210h available (20-32% margin) |
| WILDS download fails | Alternative: use CAMELYON16 from Kaggle datasets |
| UNI/Virchow model too large for T4 16GB | Use FP16 + gradient checkpointing, batch size 8 |
| MIL results don't show backbone effect | Still report honestly — negative results are valid |
| E2E results contradict frozen-feature findings | Report both — this is scientifically interesting |
| Article exceeds word limit after additions | Trim redundant paragraphs, move detail to supplementary |
| Figure/table count exceeds 10 | Merge figures per plan above; move details to text |
