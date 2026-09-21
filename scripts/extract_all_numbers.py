"""
Extract all numbers needed for article placeholders.
Run this after all experiments complete to get a structured summary
of every number that needs to go into the article.
"""
import json, os, numpy as np
from collections import defaultdict

SHARED_DIR = "/mnt/shared-workspace/shared"
RESULTS_V2 = os.path.join(SHARED_DIR, "results_v2")
RESULTS_V3 = os.path.join(SHARED_DIR, "results_v3")
PRED_DIR = os.path.join(SHARED_DIR, "predictions_v3")

BACKBONES = ['vit_tiny', 'resnet18', 'vit_b']
MODELS = ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']
MODEL_LABELS = {'baseline': 'Baseline', 'dann': '+DANN', 'ssl': '+SSL',
                'disentanglement': '+Disent.', 'multi_agent': 'Multi-Agent'}

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def get_auc(data, model, q_key):
    if data is None or q_key not in data or model not in data[q_key]:
        return None, None
    auc = data[q_key][model].get('auc', {})
    if isinstance(auc, dict):
        return auc.get('mean', 0), auc.get('ci', 0)
    return auc, 0

def get_metric(data, model, q_key, metric='auc'):
    if data is None or q_key not in data or model not in data[q_key]:
        return None, None
    m = data[q_key][model].get(metric, {})
    if isinstance(m, dict):
        return m.get('mean', 0), m.get('ci', 0)
    return m, 0

def fmt(mean, ci):
    if mean is None or mean == 0:
        return 'N/A'
    return f"{mean:.4f} ({mean-ci:.4f}-{mean+ci:.4f})"

def main():
    print("=" * 80)
    print("ARTICLE NUMBER EXTRACTION")
    print("=" * 80)

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

    # ===== Q1: Diagnostic Validity (ID Val) =====
    print("\n### Q1: Diagnostic Validity (ID Val AUC) ###")
    for bb in BACKBONES:
        v3 = results[bb]['v3_all']
        if v3 is None:
            print(f"  {bb}: N/A (v3 results not yet available)")
            continue
        print(f"  {bb}:")
        for model in MODELS:
            mean, ci = get_auc(v3, model, 'q1')
            print(f"    {MODEL_LABELS[model]}: {fmt(mean, ci)}")

    # ===== Q3: External Generalization (OOD Test) =====
    print("\n### Q3: External Generalization (OOD Test AUC) ###")
    for bb in BACKBONES:
        v3 = results[bb]['v3_all']
        if v3 is None:
            print(f"  {bb}: N/A")
            continue
        print(f"  {bb}:")
        for model in MODELS:
            mean, ci = get_auc(v3, model, 'q3')
            print(f"    {MODEL_LABELS[model]}: {fmt(mean, ci)}")

    # ===== Q3 PCam =====
    print("\n### Q3: PCam External (PCam Test AUC) ###")
    for bb in BACKBONES:
        v3 = results[bb]['v3_all']
        if v3 is None:
            print(f"  {bb}: N/A")
            continue
        print(f"  {bb}:")
        for model in MODELS:
            mean, ci = get_auc(v3, model, 'q3_pcam')
            print(f"    {MODEL_LABELS[model]}: {fmt(mean, ci)}")

    # ===== Q3 Kather2016 =====
    print("\n### Q3: Cross-Cancer (Kather2016 AUC) ###")
    for bb in ['vit_tiny', 'resnet18']:
        kather = results[bb]['kather']
        if kather is None:
            print(f"  {bb}: N/A (Kather2016 not yet run)")
            continue
        print(f"  {bb}:")
        for model in MODELS:
            mean, ci = get_auc(kather, model, 'q3_kather')
            print(f"    {MODEL_LABELS[model]}: {fmt(mean, ci)}")

    # ===== Q5: Agent Ablation =====
    print("\n### Q5: Agent Ablation (OOD Test AUC) ###")
    for bb in BACKBONES:
        v3 = results[bb]['v3_all']
        if v3 is None or 'q5_ablation' not in v3:
            print(f"  {bb}: N/A")
            continue
        print(f"  {bb}:")
        for agent in ['full_system', 'tumor_only', 'tissue_only', 'uncertainty_only']:
            mean, ci = get_auc(v3, agent, 'q5_ablation')
            print(f"    {agent}: {fmt(mean, ci)}")

    # ===== Orchestrator Variants =====
    print("\n### Orchestrator Variants (OOD Test AUC) ###")
    for bb in BACKBONES:
        orch = results[bb]['orch']
        if orch is None:
            print(f"  {bb}: N/A")
            continue
        print(f"  {bb}:")
        for variant in ['uncertainty_weighted', 'ood_validated']:
            vdata = orch.get(variant, {})
            model_key = f'multi_agent_{variant}'
            mean, ci = get_auc(vdata, model_key, 'q3')
            print(f"    {variant}: {fmt(mean, ci)}")
            # Also check q5_ablation for this variant
            abl_key = f'full_system_{variant}'
            if 'q5_ablation' in vdata and abl_key in vdata['q5_ablation']:
                abl_auc = vdata['q5_ablation'][abl_key].get('auc', {})
                if isinstance(abl_auc, dict):
                    print(f"      full_system_{variant}: {fmt(abl_auc.get('mean',0), abl_auc.get('ci',0))}")

    # ===== MixStyle + FACT =====
    print("\n### MixStyle + FACT (OOD Test AUC) ###")
    for bb in BACKBONES:
        ms = results[bb]['mixstyle_fact']
        if ms is None:
            print(f"  {bb}: N/A")
            continue
        print(f"  {bb}:")
        for model in ['mixstyle', 'fact']:
            mean, ci = get_auc(ms, model, 'q3')
            print(f"    {model}: {fmt(mean, ci)}")
            # Also ID val and PCam
            id_mean, id_ci = get_auc(ms, model, 'q1')
            pcam_mean, pcam_ci = get_auc(ms, model, 'q3_pcam')
            print(f"      ID Val: {fmt(id_mean, id_ci)}, PCam: {fmt(pcam_mean, pcam_ci)}")

    # ===== Disentanglement Sweep =====
    print("\n### Disentanglement λ Sweep ###")
    for bb in BACKBONES:
        sweep = results[bb]['disent_sweep']
        if sweep is None:
            print(f"  {bb}: N/A")
            continue
        print(f"  {bb}:")
        for lam in [0.1, 0.3, 0.5, 1.0, 2.0]:
            key = f'lambda_{lam}'
            model_key = f'disent_lam{lam}'
            if key in sweep:
                ood_mean, ood_ci = get_auc(sweep[key], model_key, 'q3')
                id_mean, id_ci = get_auc(sweep[key], model_key, 'q1')
                # Check for probe accuracies
                dom_probe = None
                lbl_probe = None
                if 'q5_disent' in sweep[key] and model_key in sweep[key]['q5_disent']:
                    dp = sweep[key]['q5_disent'][model_key].get('domain_probe_acc', {})
                    lp = sweep[key]['q5_disent'][model_key].get('label_probe_acc', {})
                    if isinstance(dp, dict):
                        dom_probe = dp.get('mean', 0)
                    if isinstance(lp, dict):
                        lbl_probe = lp.get('mean', 0)
                print(f"    λ={lam}: OOD={fmt(ood_mean, ood_ci)}, ID={fmt(id_mean, id_ci)}, "
                      f"dom_probe={dom_probe}, lbl_probe={lbl_probe}")

    # ===== Confound Control (Subsample 40K) =====
    print("\n### Confound Control (ViT-Tiny 40K vs 120K) ###")
    sub = results['vit_tiny']['v3_sub']
    v3_full = results['vit_tiny']['v3_all']
    vitb = results['vit_b']['v3_all']
    if sub:
        print("  ViT-Tiny 40K:")
        for model in MODELS:
            mean, ci = get_auc(sub, model, 'q3')
            print(f"    {MODEL_LABELS[model]}: {fmt(mean, ci)}")
    else:
        print("  ViT-Tiny 40K: N/A")
    if v3_full:
        print("  ViT-Tiny 120K (for comparison):")
        for model in MODELS:
            mean, ci = get_auc(v3_full, model, 'q3')
            print(f"    {MODEL_LABELS[model]}: {fmt(mean, ci)}")
    if vitb:
        print("  ViT-B 40K (for comparison):")
        for model in MODELS:
            mean, ci = get_auc(vitb, model, 'q3')
            print(f"    {MODEL_LABELS[model]}: {fmt(mean, ci)}")

    # ===== Clinical Metrics (from predictions) =====
    print("\n### Clinical Metrics (from raw predictions) ###")
    try:
        from clinical_metrics import (sensitivity_at_specificity, compute_ece,
                                       apply_temperature_scaling, compute_all_clinical_metrics,
                                       decision_curve_analysis)
        for bb in ['vit_tiny', 'resnet18']:
            print(f"\n  {bb}:")
            for model in MODELS:
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
                    print(f"    {MODEL_LABELS[model]}: N/A (no predictions)")
                    continue

                sens95_vals, sens99_vals, ece_pre_vals, ece_post_vals, temps, aucs = [], [], [], [], [], []
                for i, probs in enumerate(ood_preds):
                    from sklearn.metrics import roc_auc_score
                    aucs.append(roc_auc_score(ood_labels, probs))
                    s95, _ = sensitivity_at_specificity(probs, ood_labels, 0.95)
                    s99, _ = sensitivity_at_specificity(probs, ood_labels, 0.99)
                    sens95_vals.append(s95)
                    sens99_vals.append(s99)
                    ece_pre = compute_ece(probs, ood_labels)
                    ece_pre_vals.append(ece_pre)
                    if val_preds and i < len(val_preds):
                        cal_probs, temp = apply_temperature_scaling(val_preds[i], val_labels, probs)
                    else:
                        cal_probs, temp = apply_temperature_scaling(probs, ood_labels, probs)
                    ece_post = compute_ece(cal_probs, ood_labels)
                    ece_post_vals.append(ece_post)
                    temps.append(temp)

                print(f"    {MODEL_LABELS[model]}: AUC={np.mean(aucs):.4f}, "
                      f"Sens@95%={np.mean(sens95_vals):.4f}, Sens@99%={np.mean(sens99_vals):.4f}, "
                      f"ECE_pre={np.mean(ece_pre_vals):.4f}, ECE_post={np.mean(ece_post_vals):.4f}, "
                      f"Temp={np.mean(temps):.2f}")
    except Exception as e:
        print(f"  Error computing clinical metrics: {e}")

    # ===== DeLong Tests (from v2) =====
    print("\n### DeLong Tests (from v2, seed 0) ###")
    for bb in BACKBONES:
        v2 = results[bb]['v2']
        if v2 and 'delong_tests' in v2:
            dl = v2['delong_tests']
            if 'ood_test' in dl:
                print(f"  {bb} OOD DeLong:")
                for test in dl['ood_test']:
                    if test['model_1'] == 'baseline' and test['model_2'] == 'multi_agent':
                        print(f"    baseline vs multi_agent: p={test['p_value']:.6f}, significant={test['significant']}")

    # ===== Generalization Gap =====
    print("\n### Generalization Gap (ID AUC - OOD AUC) ###")
    for bb in BACKBONES:
        v3 = results[bb]['v3_all']
        if v3 is None:
            print(f"  {bb}: N/A")
            continue
        for model in MODELS:
            id_mean, _ = get_auc(v3, model, 'q1')
            ood_mean, _ = get_auc(v3, model, 'q3')
            if id_mean and ood_mean:
                gap = id_mean - ood_mean
                print(f"  {bb} {MODEL_LABELS[model]}: gap={gap:.4f} (ID={id_mean:.4f}, OOD={ood_mean:.4f})")

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
