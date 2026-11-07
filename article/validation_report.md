# Manuscript validation report

Checked `article/article.md` against `tables/`, `results/` and `scripts/` (commit afcc235, 6 Nov 2026).
Severity: **Critical** = a reviewer would likely reject on it; **Major** = must fix before submission; **Minor** = tidy-up.

## Critical

### 1. The "six specialist agents" are not what the Methods describe
The Methods (article.md line 45) describe an image-quality agent "detecting artefacts and blur", a tissue-classification agent "identifying tissue type", a biomarker agent "estimating continuous biomarker scores" and a "disentangled representation agent". The training code (`scripts/run_experiments_v3.py` lines 255–284) shows:

| Agent (as described) | What it is actually trained on |
|---|---|
| Image quality (artefacts, blur) | Whether the patch is from domain 0, i.e. *not* HED-augmented: `(d == 0)` |
| Tissue classification | The tumour label `y`, the same target as the tumour agent |
| Tumour detection | The tumour label `y` |
| Biomarker quantification | MSE regression onto the binary tumour label `y`; no biomarker data exists in these datasets |
| Uncertainty (MC-Dropout) | The tumour label `y` |
| Disentangled representation | Not a separate agent: the 6th orchestrator input is `tumor_prob` repeated |

None of CAMELYON17, PCam or CRC-VAL-HE-7K has artefact, tissue-type or biomarker labels. So five of the six "agents" are tumour classifiers with different sizes and dropout, plus one domain classifier. A pathology reviewer reading the code or the data description will see this.

**Fix (pick one):** (a) describe them honestly as "proxy-supervised heads" (domain-identity head; tumour heads of different capacity and dropout) and retitle the system accordingly; or (b) give the agents real tasks: blur/artefact detection with synthetic corruptions (the `image_perturbations.py` code already exists), tissue type from the 9-class NCT-CRC-HE labels, and drop "biomarker" unless a dataset with biomarker labels is added.

### 2. The headline "uncertainty agent 0.829 vs multi-agent 0.626" needs re-checking
The uncertainty agent and the tumour agent are both MLPs trained on the same label and features; the main difference is dropout (0.2 versus 0.1). A gap of 0.83 versus 0.57 out-of-distribution AUC from that alone is surprising. The abstract's central claim ("the bottleneck lies in agent-level robustness") rests on it. **Check:** that the ablation evaluates each agent the same way (deterministic versus MC-averaged predictions, same seeds, same checkpoint selection), and report per-seed values.

### 3. Most results cannot be reproduced from the repository
`scripts/generate_tables_v3.py` reads `results_v2/` and `results_v3/` from a shared workspace that is not in the repo; only `results/all_mil_results.json` is committed. Tables 2, 2b and 3 and every number in the abstract cannot currently be traced to a file. **Fix:** commit the result JSONs (they are small) and point the table and figure scripts at `results/`.

## Major

4. **Dataset name and citation are wrong.** "Kather2016" with reference [50] (Kather et al., *Sci Rep* 2016, the 5,000-tile texture dataset) is used for **CRC-VAL-HE-7K**, which comes from Kather et al., *PLoS Med* 2019;16(1):e1002730 (Zenodo 1214456). Rename it throughout (e.g. "CRC-VAL-HE-7K") and fix the reference. Table 1 also cites a placeholder "[K1]". The Limitations section calls NCT-CRC-HE-100K "the Kather2016 training set", which is also wrong.
5. **The ViT-Tiny backbone is misdescribed.** The code loads `1aurent/phikon-distil-vit-tiny-patch16-224-kather2016`, a community distilled model, not an official Owkin Phikon release. The paper says "Phikon ViT-Tiny … pre-trained on NCT-CRC-HE-100K via DINOv2". Check the model card and describe it exactly. If it was trained *with labels* on colorectal tiles, the cross-cancer result for ViT-Tiny is confounded beyond what the Limitations admit.
6. **The abstract claim "Temperature scaling … ΔECE < 0.02" is false.** Per Table 3: ResNet-18 Baseline 0.4464 → 0.4193 (Δ 0.027); ResNet-18 SSL Δ 0.022; ViT-T DANN gets *worse* by 0.025; ViT-T Disent. worse by 0.024. Rephrase as "changed ECE by at most 0.03 and sometimes increased it".
7. **The ECE values look implausible.** ECE of 0.44–0.50 alongside out-of-distribution AUC of 0.96–0.99 means predictions are ranked well but thresholds have shifted, for example most OOD patches scored above 0.5. That may be a genuine finding, but check the ECE code and report accuracy at 0.5. Readers will ask.
8. **The disentanglement loss is misdescribed.** The paper says the loss "minimises mutual information"; the code uses an orthogonality loss (`orthogonality_loss`, run_experiments_v3.py line 219). Describe the actual loss.
9. **The MC-Dropout pass count doesn't match.** The paper says 20 stochastic passes; the code defaults are 30 (`predict_with_uncertainty`) and 50 (`estimate_uncertainty`). State the value actually used.
10. **The "SSL" description is inaccurate.** In the multi-agent loss, "SSL" is cosine similarity between features with and without Gaussian noise, not SimCLR. Check the standalone SSL model too.

## Minor

11. "All architectures achieved OOD AUC above 0.985" with ViT-B: the Baseline is 0.9849. Say "≥ 0.985".
12. "ViT-Tiny sensitivity at 95% specificity was 0.03": the Baseline is 0.024, the multi-agent 0.035. Name which model.
13. The Data Availability statement still says "[GitHub URL placeholder]".
14. The "domains" are three HED augmentations of one centre plus PCam, so the domain-probe accuracies measure augmentation detectability, not institution. Say so where probes are discussed.
15. Submission deadline: the editor's email gives **5 Feb 2027**; the journal page shows 30 Jun 2027. Plan for February.

## What checks out
- Abstract figures for ViT-Tiny Baseline (0.537, CI 0.501–0.572), multi-agent (0.626), DANN (0.663), the best orchestrator variant (0.637), the ViT-B generalisation gap (0.007–0.014) and the cross-cancer ranges (0.37–0.41; 0.54–0.64) all match Tables 2 and 2b.
- CRC-VAL-HE-7K tile count (1,974 = 741 NORM + 1,233 TUM) matches the official archive.
- 54 references, all cited in the text.
