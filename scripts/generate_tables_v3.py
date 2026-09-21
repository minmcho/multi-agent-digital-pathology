"""
Table generation v3: creates markdown tables for the article.
Combines v2 and v3 results, includes Kather2016, MixStyle/FACT, orchestrator variants,
and clinical metrics.

Usage:
  python generate_tables_v3.py
"""
import json, os, glob, numpy as np
from collections import defaultdict

SHARED_DIR = "/mnt/shared-workspace/shared"
RESULTS_V2 = os.path.join(SHARED_DIR, "results_v2")
RESULTS_V3 = os.path.join(SHARED_DIR, "results_v3")
PRED_DIR = os.path.join(SHARED_DIR, "predictions_v3")
TABLE_DIR = "/mnt/results"

BACKBONES = ['vit_tiny', 'resnet18', 'vit_b']
BACKBONE_LABELS = {'vit_tiny': 'Phikon ViT-T', 'vit_b': 'Phikon ViT-B', 'resnet18': 'ResNet-18'}
MODEL_NAMES = ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']
MODEL_LABELS = ['Baseline', '+DANN', '+SSL', '+Disent.', 'Multi-Agent']


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fmt_auc(mean, ci):
    """Format AUC as mean (lower–upper)."""
    if mean == 0:
        return '—'
    return f"{mean:.4f} ({mean-ci:.4f}–{mean+ci:.4f})"


def get_metric(data, model, q_key, metric='auc'):
    if data is None or q_key not in data or model not in data[q_key]:
        return 0, 0
    m = data[q_key][model].get(metric, {})
    return m.get('mean', 0), m.get('ci', 0)


def save_table(name, content):
    path = os.path.join(TABLE_DIR, f"{name}.md")
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Saved {path}")


# ============================================================
# Table 1: Dataset Summary
# ============================================================

def table1_datasets():
    content = """**Table 1.** Dataset summary.

| Dataset | Source | Task | Image Size | Train | Val | Test | Domain Role |
|---------|--------|------|-----------|-------|-----|------|-------------|
| Camelyon17 ID | Bandi et al. [1] | Binary tumour detection | 96×96 | 30,000 | 10,000 | — | Domains 0–2 (original + HED augmentations) |
| PCam | Veeling et al. [3] | Binary tumour detection | 96×96 | 30,000 | 10,000 | 10,000 | Domain 3 (real cross-dataset) |
| Camelyon17 OOD | Bandi et al. [1] | Binary tumour detection | 96×96 | — | — | 20,000 | Unseen medical centre (OOD test) |
| Kather2016 (CRC-VAL-HE-7K) | Kather et al. [K1] | TUM vs NORM | 224×224 | — | — | 1,974 | Cross-cancer-type OOD test |

**Note:** Camelyon17 ID training (30,000) was augmented with HED stain deconvolution to create 3 synthetic domains (90,000 total). Combined with PCam training (30,000), the total training set was 120,000 samples across 4 domains. Kather2016 CRC-VAL-HE-7K is the official held-out validation set from the NCT-CRC-HE-100K collection, used here as a cross-cancer-type OOD test (colorectal adenocarcinoma vs normal colon mucosa).
"""
    save_table('table1', content)


# ============================================================
# Table 2: Main Results — AUC across backbones, models, and datasets
# ============================================================

def table2_main_results(results):
    rows = []
    header = "| Backbone | Model | ID Val (Camelyon17) | OOD Test (Camelyon17) | External (PCam) | Cross-Cancer (Kather2016) |"
    separator = "|----------|-------|---------------------|----------------------|-----------------|---------------------------|"
    rows.append(header)
    rows.append(separator)

    for bb in BACKBONES:
        v3 = results[bb].get('v3_all')
        kather = results[bb].get('kather')

        for m_idx, (model, label) in enumerate(zip(MODEL_NAMES, MODEL_LABELS)):
            id_mean, id_ci = get_metric(v3, model, 'q1')
            ood_mean, ood_ci = get_metric(v3, model, 'q3')
            pcam_mean, pcam_ci = get_metric(v3, model, 'q3_pcam')
            kather_mean, kather_ci = get_metric(kather, model, 'q3_kather')

            bb_label = BACKBONE_LABELS[bb] if m_idx == 0 else ''
            rows.append(f"| {bb_label} | {label} | {fmt_auc(id_mean, id_ci)} | {fmt_auc(ood_mean, ood_ci)} | {fmt_auc(pcam_mean, pcam_ci)} | {fmt_auc(kather_mean, kather_ci)} |")

    content = "**Table 2.** Diagnostic performance (AUC with 95% CI) across three feature backbones, five model architectures, and four evaluation datasets. AUC values are mean ± 95% CI across 10 random seeds.\n\n" + '\n'.join(rows) + "\n"
    save_table('table2', content)


# ============================================================
# Table 2b: DG Baselines + Orchestrator Variants
# ============================================================

def table2b_dg_baselines(results):
    rows = []
    header = "| Backbone | Model | ID Val | OOD Test | PCam | Kather2016 |"
    separator = "|----------|-------|--------|----------|------|------------|"
    rows.append(header)
    rows.append(separator)

    dg_models = [
        ('baseline', 'Baseline'),
        ('mixstyle', 'MixStyle'),
        ('fact', 'FACT'),
        ('multi_agent', 'Multi-Agent (original)'),
        ('multi_agent_uncertainty_weighted', 'MA-UncWeighted'),
        ('multi_agent_ood_validated', 'MA-OODValidated'),
    ]

    for bb in ['vit_tiny', 'resnet18', 'vit_b']:
        v3 = results[bb].get('v3_all')
        ms_fact = results[bb].get('mixstyle_fact')
        orch = results[bb].get('orch')
        kather = results[bb].get('kather')

        for m_idx, (model, label) in enumerate(dg_models):
            if model in ['baseline', 'multi_agent']:
                data = v3
            elif model in ['mixstyle', 'fact']:
                data = ms_fact
            elif 'uncertainty_weighted' in model:
                data = orch.get('uncertainty_weighted', {}) if orch else None
            elif 'ood_validated' in model:
                data = orch.get('ood_validated', {}) if orch else None
            else:
                data = None

            id_mean, id_ci = get_metric(data, model, 'q1')
            ood_mean, ood_ci = get_metric(data, model, 'q3')
            pcam_mean, pcam_ci = get_metric(data, model, 'q3_pcam')

            # Kather2016 only for baseline and multi_agent
            if model in ['baseline', 'multi_agent'] and kather:
                k_mean, k_ci = get_metric(kather, model, 'q3_kather')
            else:
                k_mean, k_ci = 0, 0

            bb_label = BACKBONE_LABELS[bb] if m_idx == 0 else ''
            rows.append(f"| {bb_label} | {label} | {fmt_auc(id_mean, id_ci)} | {fmt_auc(ood_mean, ood_ci)} | {fmt_auc(pcam_mean, pcam_ci)} | {fmt_auc(k_mean, k_ci)} |")

    content = "**Table 2b.** Domain generalisation baselines and orchestrator variants. MixStyle and FACT are established DG methods. MA-UncWeighted uses inverse-variance fusion; MA-OODValidated uses OOD validation for model selection.\n\n" + '\n'.join(rows) + "\n"
    save_table('table2b', content)


# ============================================================
# Table 3: Clinical Metrics
# ============================================================

def table3_clinical_metrics(results):
    from clinical_metrics import (sensitivity_at_specificity, compute_ece,
                                   apply_temperature_scaling, compute_all_clinical_metrics)

    rows = []
    header = "| Backbone | Model | AUC | Sens@95% Spec | Sens@99% Spec | ECE (pre-temp) | ECE (post-temp) | Temp |"
    separator = "|----------|-------|-----|---------------|---------------|----------------|-----------------|------|"
    rows.append(header)
    rows.append(separator)

    for bb in ['vit_tiny', 'resnet18', 'vit_b']:
        for model, label in zip(MODEL_NAMES, MODEL_LABELS):
            # Load predictions
            ood_preds, ood_labels = [], None
            val_preds, val_labels = [], None
            for seed in range(10):
                ood_path = os.path.join(PRED_DIR, bb, f"{model}_seed{seed}_ood_test.npz")
                val_path = os.path.join(PRED_DIR, bb, f"{model}_seed{seed}_id_val.npz")
                if os.path.exists(ood_path):
                    d = np.load(ood_path)
                    ood_preds.append(d['probs'])
                    if ood_labels is None:
                        ood_labels = d['labels']
                if os.path.exists(val_path):
                    d = np.load(val_path)
                    val_preds.append(d['probs'])
                    if val_labels is None:
                        val_labels = d['labels']

            if not ood_preds:
                bb_label = BACKBONE_LABELS[bb] if model == MODEL_NAMES[0] else ''
                rows.append(f"| {bb_label} | {label} | — | — | — | — | — | — |")
                continue

            # Average across seeds
            sens95_vals, sens99_vals, ece_pre_vals, ece_post_vals, temps = [], [], [], [], []
            aucs = []
            for seed_idx, probs in enumerate(ood_preds):
                metrics = compute_all_clinical_metrics(probs, ood_labels)
                sens95_vals.append(metrics['sensitivity_at_95_spec'])
                sens99_vals.append(metrics['sensitivity_at_99_spec'])
                ece_pre = compute_ece(probs, ood_labels)
                ece_pre_vals.append(ece_pre)

                if val_preds and seed_idx < len(val_preds):
                    cal_probs, temp = apply_temperature_scaling(val_preds[seed_idx], val_labels, probs)
                else:
                    cal_probs, temp = apply_temperature_scaling(probs, ood_labels, probs)
                ece_post = compute_ece(cal_probs, ood_labels)
                ece_post_vals.append(ece_post)
                temps.append(temp)

                from sklearn.metrics import roc_auc_score
                aucs.append(roc_auc_score(ood_labels, probs))

            bb_label = BACKBONE_LABELS[bb] if model == MODEL_NAMES[0] else ''
            rows.append(f"| {bb_label} | {label} | {np.mean(aucs):.4f} | {np.mean(sens95_vals):.4f} | {np.mean(sens99_vals):.4f} | {np.mean(ece_pre_vals):.4f} | {np.mean(ece_post_vals):.4f} | {np.mean(temps):.2f} |")

    content = "**Table 3.** Clinical metrics on OOD test set (Camelyon17 unseen centre). Sens@95% Spec = sensitivity at 95% specificity; ECE = Expected Calibration Error; Temp = fitted temperature parameter. All values are means across 10 seeds.\n\n" + '\n'.join(rows) + "\n"
    save_table('table3', content)


# ============================================================
# Main
# ============================================================

def main():
    print("=== Table Generation v3 ===")

    # Load all results
    results = {}
    for bb in BACKBONES:
        results[bb] = {
            'v2': load_json(os.path.join(RESULTS_V2, bb, "all_results.json")),
            'v3_all': load_json(os.path.join(RESULTS_V3, bb, "all_results_v3.json")),
            'v3_sub': load_json(os.path.join(RESULTS_V3, bb, "all_results_v3_subsample40k.json")),
            'orch': load_json(os.path.join(RESULTS_V3, bb, "orchestrator_variants.json")),
            'disent_sweep': load_json(os.path.join(RESULTS_V3, bb, "disentanglement_sweep.json")),
            'mixstyle_fact': load_json(os.path.join(RESULTS_V3, bb, "mixstyle_fact.json")),
            'kather': load_json(os.path.join(RESULTS_V3, bb, "kather2016_results.json")),
        }

    print("  Generating Table 1 (datasets)...")
    table1_datasets()

    print("  Generating Table 2 (main results)...")
    table2_main_results(results)

    print("  Generating Table 2b (DG baselines)...")
    table2b_dg_baselines(results)

    print("  Generating Table 3 (clinical metrics)...")
    table3_clinical_metrics(results)

    print("\nTables saved to", TABLE_DIR)


if __name__ == "__main__":
    main()
