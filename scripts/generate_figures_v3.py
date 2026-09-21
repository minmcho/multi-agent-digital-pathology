"""
Figure generation v3: combines v2 and v3 results, includes Kather2016,
MixStyle/FACT, orchestrator variants, disentanglement sweep, and clinical metrics.

Generates data-driven figures (Fig 3-7) with error bars and CI bands.
Schematic figures (Fig 1-2) are generated separately via GenerateImage.

Usage:
  python generate_figures_v3.py
"""
import json, os, glob, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# Font and style settings
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

SHARED_DIR = "/mnt/shared-workspace/shared"
RESULTS_V2 = os.path.join(SHARED_DIR, "results_v2")
RESULTS_V3 = os.path.join(SHARED_DIR, "results_v3")
PRED_DIR = os.path.join(SHARED_DIR, "predictions_v3")
FIGURE_DIR = "/mnt/results/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Color palette (Phylo)
COLORS = {
    'baseline': '#0279EE',
    'dann': '#FF9400',
    'ssl': '#75A025',
    'disentanglement': '#FD9BED',
    'multi_agent': '#E9ED4C',
    'mixstyle': '#000000',
    'fact': '#75A025',
    'multi_agent_uncertainty_weighted': '#FF9400',
    'multi_agent_ood_validated': '#FD9BED',
}
BACKBONE_COLORS = {
    'vit_tiny': '#0279EE',
    'vit_b': '#75A025',
    'resnet18': '#FF9400',
}
MODEL_NAMES = ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']
MODEL_LABELS = ['Baseline', '+DANN', '+SSL', '+Disent.', 'Multi-Agent']
DG_MODELS = ['baseline', 'mixstyle', 'fact', 'multi_agent', 'multi_agent_uncertainty_weighted']
DG_LABELS = ['Baseline', 'MixStyle', 'FACT', 'Multi-Agent', 'MA-UncW']
BACKBONE_LABELS = {'vit_tiny': 'Phikon ViT-T', 'vit_b': 'Phikon ViT-B', 'resnet18': 'ResNet-18'}
BACKBONES = ['vit_tiny', 'resnet18', 'vit_b']


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_all_results():
    """Load v2 and v3 results for all backbones."""
    results = {}
    for bb in BACKBONES:
        results[bb] = {}
        # v2 results (has Q2 robustness, Q4 uncertainty, Q5 probes, threshold opt)
        v2 = load_json(os.path.join(RESULTS_V2, bb, "all_results.json"))
        if v2:
            results[bb]['v2'] = v2
        # v3 results
        results[bb]['v3_all'] = load_json(os.path.join(RESULTS_V3, bb, "all_results_v3.json"))
        results[bb]['v3_sub'] = load_json(os.path.join(RESULTS_V3, bb, "all_results_v3_subsample40k.json"))
        results[bb]['orch'] = load_json(os.path.join(RESULTS_V3, bb, "orchestrator_variants.json"))
        results[bb]['disent_sweep'] = load_json(os.path.join(RESULTS_V3, bb, "disentanglement_sweep.json"))
        results[bb]['mixstyle_fact'] = load_json(os.path.join(RESULTS_V3, bb, "mixstyle_fact.json"))
        results[bb]['kather'] = load_json(os.path.join(RESULTS_V3, bb, "kather2016_results.json"))
    return results


def load_predictions(backbone, model_name, dataset_name, n_seeds=10):
    """Load raw predictions for a given model/dataset across seeds."""
    preds = []
    labels = None
    for seed in range(n_seeds):
        path = os.path.join(PRED_DIR, backbone, f"{model_name}_seed{seed}_{dataset_name}.npz")
        if os.path.exists(path):
            data = np.load(path)
            preds.append(data['probs'])
            if labels is None:
                labels = data['labels']
    return preds, labels


def _save_fig(name):
    import subprocess as sp
    os.makedirs(FIGURE_DIR, exist_ok=True)
    tmp_dir = "/workspace/_fig_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    for ext in ['.png', '.svg']:
        tmp_path = os.path.join(tmp_dir, f"{name}{ext}")
        dst_path = os.path.join(FIGURE_DIR, f"{name}{ext}")
        plt.savefig(tmp_path, dpi=300, bbox_inches='tight')
        sp.run(['cp', tmp_path, dst_path], check=True)
        os.remove(tmp_path)
    print(f"  Saved {name}.png and {name}.svg")


def get_auc(data, model, q_key):
    """Extract AUC mean and CI from results dict."""
    if data is None or q_key not in data or model not in data[q_key]:
        return 0, 0
    auc = data[q_key][model].get('auc', {})
    return auc.get('mean', 0), auc.get('ci', 0)


# ============================================================
# Fig 3: Diagnostic Performance (AUC) — 4 datasets × 3 backbones × 5 models
# ============================================================

def fig3_diagnostic_performance(results):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    datasets = [
        ('q1', 'ID Val\n(Camelyon17)', 'v3_all'),
        ('q3', 'OOD Test\n(Camelyon17)', 'v3_all'),
        ('q3_pcam', 'External\n(PCam)', 'v3_all'),
        ('q3_kather', 'Cross-Cancer\n(Kather2016)', 'kather'),
    ]

    for ax_idx, (q_key, title, source) in enumerate(datasets):
        ax = axes[ax_idx]
        backbones = [bb for bb in BACKBONES if bb in results]
        n_bbs = len(backbones)
        width = 0.8 / n_bbs

        for bb_idx, bb in enumerate(backbones):
            data = results[bb].get(source)
            if data is None:
                continue
            x_pos = np.arange(len(MODEL_NAMES)) + (bb_idx - n_bbs/2 + 0.5) * width
            for m_idx, model in enumerate(MODEL_NAMES):
                mean, ci = get_auc(data, model, q_key)
                if mean == 0:
                    continue
                color = BACKBONE_COLORS[bb]
                ax.bar(x_pos[m_idx], mean, width=width*0.9, color=color, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
                ax.errorbar(x_pos[m_idx], mean, yerr=ci, fmt='none',
                           ecolor='black', capsize=2, linewidth=0.8)

        ax.set_xticks(np.arange(len(MODEL_NAMES)))
        ax.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=30, ha='right')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylim(0.5, 1.01)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
        if ax_idx == 0:
            ax.set_ylabel('AUC', fontsize=11)

    handles = [Patch(facecolor=BACKBONE_COLORS[bb], label=BACKBONE_LABELS[bb])
               for bb in BACKBONES if bb in results]
    fig.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    _save_fig('fig3_diagnostic_performance')
    plt.close()


# ============================================================
# Fig 4: Perturbation Robustness Heatmap (from v2)
# ============================================================

def fig4_perturbation_robustness(results):
    bb = 'vit_tiny'
    v2 = results.get(bb, {}).get('v2')
    if v2 is None or 'q2_robustness' not in v2:
        print("  Skipping fig4: no v2 robustness data")
        return

    q2 = v2['q2_robustness']
    perturbation_types = ['gaussian_noise', 'stain_shift', 'gaussian_blur',
                          'jpeg_compression', 'brightness_contrast']
    pert_labels = ['Gaussian\nNoise', 'Stain\nShift', 'Gaussian\nBlur',
                   'JPEG\nCompression', 'Brightness/\nContrast']
    sev_labels = ['Mild', 'Moderate', 'Severe']

    fig, ax = plt.subplots(figsize=(10, 5))
    data = np.zeros((len(perturbation_types) * 3, len(MODEL_NAMES)))

    for p_idx, ptype in enumerate(perturbation_types):
        for s_idx in range(3):
            row = p_idx * 3 + s_idx
            for m_idx, model in enumerate(MODEL_NAMES):
                if model in q2 and 'perturbations' in q2[model]:
                    perts = q2[model]['perturbations']
                    if ptype in perts and s_idx < len(perts[ptype]):
                        data[row, m_idx] = perts[ptype][s_idx]['auc']['mean']

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(np.arange(len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_LABELS, fontsize=9)
    y_labels = []
    for p_idx, plabel in enumerate(pert_labels):
        for s_idx, slabel in enumerate(sev_labels):
            y_labels.append(f"{plabel.replace(chr(10), ' ')} - {slabel}")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.3f}", ha='center', va='center',
                   fontsize=6, color='black' if data[i,j] > 0.7 else 'white')

    plt.colorbar(im, ax=ax, label='AUC', shrink=0.8)
    ax.set_title(f'Perturbation Robustness ({BACKBONE_LABELS[bb]})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save_fig('fig4_perturbation_robustness')
    plt.close()


# ============================================================
# Fig 5: Calibration — ECE + Temperature Scaling
# ============================================================

def fig5_calibration(results):
    from clinical_metrics import compute_ece, apply_temperature_scaling

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ECE pre vs post temperature scaling (OOD test)
    backbones = [bb for bb in BACKBONES if results[bb].get('v3_all')]
    width = 0.35
    x = np.arange(len(MODEL_NAMES))

    for bb_idx, bb in enumerate(backbones):
        ece_pre_vals = []
        ece_post_vals = []
        for model in MODEL_NAMES:
            # Load predictions for OOD test
            preds_list, labels = load_predictions(bb, model, 'ood_test')
            if not preds_list:
                ece_pre_vals.append(0)
                ece_post_vals.append(0)
                continue

            # Average ECE across seeds
            ece_pre_seeds = []
            ece_post_seeds = []
            for seed_idx, probs in enumerate(preds_list):
                # Use id_val as calibration set
                val_preds_list, val_labels = load_predictions(bb, model, 'id_val')
                if seed_idx < len(val_preds_list):
                    val_probs = val_preds_list[seed_idx]
                    val_labs = val_labels
                else:
                    val_probs = probs
                    val_labs = labels

                ece_pre = compute_ece(probs, labels)
                calibrated_probs, temp = apply_temperature_scaling(val_probs, val_labs, probs)
                ece_post = compute_ece(calibrated_probs, labels)
                ece_pre_seeds.append(ece_pre)
                ece_post_seeds.append(ece_post)

            ece_pre_vals.append(np.mean(ece_pre_seeds))
            ece_post_vals.append(np.mean(ece_post_seeds))

        offset = (bb_idx - len(backbones)/2 + 0.5) * width * 2
        color = BACKBONE_COLORS[bb]
        ax1.bar(x + offset - width/4, ece_pre_vals, width=width/2, color=color,
                alpha=0.5, edgecolor='black', linewidth=0.5, label=f'{BACKBONE_LABELS[bb]} (pre)')
        ax1.bar(x + offset + width/4, ece_post_vals, width=width/2, color=color,
                alpha=1.0, edgecolor='black', linewidth=0.5, label=f'{BACKBONE_LABELS[bb]} (post)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=30, ha='right')
    ax1.set_ylabel('ECE', fontsize=10)
    ax1.set_title('Calibration: ECE Before vs After Temperature Scaling (OOD)', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7, ncol=2)

    # Right: Reliability diagram for ViT-Tiny multi_agent (pre vs post temp)
    bb = 'vit_tiny'
    model = 'multi_agent'
    preds_list, labels = load_predictions(bb, model, 'ood_test')
    if preds_list:
        probs = preds_list[0]  # Use seed 0
        val_preds_list, val_labels = load_predictions(bb, model, 'id_val')
        val_probs = val_preds_list[0] if val_preds_list else probs
        val_labs = val_labels if val_preds_list else labels

        calibrated_probs, temp = apply_temperature_scaling(val_probs, val_labs, probs)

        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        for label, p, ls in [('Pre-calibration', probs, '-'), ('Post-calibration', calibrated_probs, '--')]:
            accs = []
            confs = []
            for i in range(n_bins):
                mask = (p >= bin_boundaries[i]) & (p < bin_boundaries[i + 1])
                if mask.sum() > 0:
                    accs.append(((p[mask] >= 0.5).astype(int) == labels[mask]).mean())
                    confs.append(p[mask].mean())
                else:
                    accs.append(np.nan)
                    confs.append(np.nan)
            ax2.plot(confs, accs, 'o' + ls, label=label, markersize=5)

        ax2.plot([0, 1], [0, 1], 'k--', linewidth=0.5, label='Perfect')
        ax2.set_xlabel('Confidence', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title(f'Reliability Diagram ({BACKBONE_LABELS[bb]}, Multi-Agent)', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    _save_fig('fig5_calibration')
    plt.close()


# ============================================================
# Fig 6: Disentanglement λ Sweep + Agent Ablation
# ============================================================

def fig6_disentanglement_ablation(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Disentanglement λ sweep (OOD AUC vs λ)
    bb = 'vit_tiny'
    sweep = results.get(bb, {}).get('disent_sweep')
    if sweep:
        lambdas = [0.1, 0.3, 0.5, 1.0, 2.0]
        ood_aucs = []
        ood_cis = []
        id_aucs = []
        for lam in lambdas:
            key = f'lambda_{lam}'
            if key in sweep and 'q3' in sweep[key]:
                model_key = f'disent_lam{lam}'
                if model_key in sweep[key]['q3']:
                    auc_data = sweep[key]['q3'][model_key].get('auc', {})
                    ood_aucs.append(auc_data.get('mean', 0))
                    ood_cis.append(auc_data.get('ci', 0))
                else:
                    ood_aucs.append(0)
                    ood_cis.append(0)
            else:
                ood_aucs.append(0)
                ood_cis.append(0)
            if key in sweep and 'q1' in sweep[key]:
                model_key = f'disent_lam{lam}'
                if model_key in sweep[key]['q1']:
                    id_aucs.append(sweep[key]['q1'][model_key].get('auc', {}).get('mean', 0))
                else:
                    id_aucs.append(0)
            else:
                id_aucs.append(0)

        ax1.plot(lambdas, ood_aucs, 'o-', color='#FF9400', label='OOD AUC', markersize=6)
        ax1.fill_between(lambdas, [a-c for a,c in zip(ood_aucs, ood_cis)],
                         [a+c for a,c in zip(ood_aucs, ood_cis)], color='#FF9400', alpha=0.2)
        ax1.plot(lambdas, id_aucs, 's-', color='#0279EE', label='ID AUC', markersize=6)
        ax1.set_xlabel('Disentanglement Loss Weight (λ)', fontsize=10)
        ax1.set_ylabel('AUC', fontsize=10)
        ax1.set_title('Disentanglement Loss Sweep', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.set_ylim(0.5, 1.01)
        ax1.set_xscale('log')
    else:
        # Fallback: v2 disentanglement probes
        v2 = results.get(bb, {}).get('v2')
        if v2 and 'q5_disentanglement' in v2:
            q5 = v2['q5_disentanglement']
            probe_models = ['baseline', 'dann', 'disentanglement', 'multi_agent']
            probe_labels = ['Baseline', '+DANN', '+Disent.', 'Multi-Agent']
            x = np.arange(len(probe_models))
            width = 0.35
            dom_accs = [q5[m]['bio_domain_probe_acc']['mean'] for m in probe_models if m in q5]
            lbl_accs = [q5[m]['bio_label_probe_acc']['mean'] for m in probe_models if m in q5]
            if dom_accs and lbl_accs:
                ax1.bar(x[:len(dom_accs)] - width/2, dom_accs, width, label='Domain probe (↓)',
                        color='#FF9400', alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.bar(x[:len(lbl_accs)] + width/2, lbl_accs, width, label='Label probe (↑)',
                        color='#0279EE', alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.set_xticks(x[:len(probe_labels)])
                ax1.set_xticklabels(probe_labels, fontsize=9)
                ax1.set_ylabel('Probe Accuracy', fontsize=10)
                ax1.set_title('Disentanglement Probes (Z_bio)', fontsize=11, fontweight='bold')
                ax1.legend(fontsize=8)
                ax1.set_ylim(0, 1.05)

    # Right: Agent ablation (OOD test) — compare original vs uncertainty_weighted
    v3_all = results.get(bb, {}).get('v3_all')
    orch = results.get(bb, {}).get('orch')
    if v3_all and 'q5_ablation' in v3_all:
        q5_abl = v3_all['q5_ablation']
        abl_names = ['full_system', 'tumor_only', 'tissue_only', 'uncertainty_only']
        abl_labels = ['Full\nSystem', 'Tumor\nOnly', 'Tissue\nOnly', 'Uncertainty\nOnly']
        aucs = []
        for name in abl_names:
            if name in q5_abl and 'auc' in q5_abl[name]:
                val = q5_abl[name]['auc']
                aucs.append(val.get('mean', val) if isinstance(val, dict) else val)
            else:
                aucs.append(0)

        x = np.arange(len(abl_labels))
        colors = ['#E9ED4C', '#0279EE', '#75A025', '#FD9BED']
        bars = ax2.bar(x, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5, width=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(abl_labels, fontsize=8)
        ax2.set_ylabel('AUC (OOD Test)', fontsize=10)
        ax2.set_title('Agent Ablation (Original MA)', fontsize=11, fontweight='bold')
        ax2.set_ylim(0.5, 1.01)
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

        # Add orchestrator variant OOD AUCs as horizontal lines
        if orch:
            for variant, color, ls in [('uncertainty_weighted', '#FF9400', '--'),
                                        ('ood_validated', '#FD9BED', ':')]:
                vdata = orch.get(variant, {})
                if 'q3' in vdata:
                    model_key = f'multi_agent_{variant}'
                    if model_key in vdata['q3']:
                        auc_val = vdata['q3'][model_key].get('auc', {}).get('mean', 0)
                        ax2.axhline(y=auc_val, color=color, linestyle=ls, linewidth=1.5,
                                   label=f'{variant.replace("_", " ").title()} ({auc_val:.3f})')
            ax2.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    _save_fig('fig6_disentanglement_ablation')
    plt.close()


# ============================================================
# Fig 7: Clinical Utility — Sensitivity@Specificity + Decision Curve
# ============================================================

def fig7_clinical_utility(results):
    from clinical_metrics import sensitivity_at_specificity, decision_curve_analysis

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Sensitivity at 95% and 99% specificity (OOD test, all backbones)
    backbones = [bb for bb in BACKBONES if results[bb].get('v3_all')]
    width = 0.8 / len(backbones)

    for spec_idx, (target_spec, spec_label) in enumerate([(0.95, '95%'), (0.99, '99%')]):
        ax = ax1 if spec_idx == 0 else ax2
        for bb_idx, bb in enumerate(backbones):
            x_pos = np.arange(len(MODEL_NAMES)) + (bb_idx - len(backbones)/2 + 0.5) * width
            sens_vals = []
            for model in MODEL_NAMES:
                preds_list, labels = load_predictions(bb, model, 'ood_test')
                if not preds_list:
                    sens_vals.append(0)
                    continue
                seed_sens = []
                for probs in preds_list:
                    sens, _ = sensitivity_at_specificity(probs, labels, target_spec)
                    seed_sens.append(sens)
                sens_vals.append(np.mean(seed_sens))

            color = BACKBONE_COLORS[bb]
            ax.bar(x_pos, sens_vals, width=width*0.9, color=color, alpha=0.8,
                   edgecolor='black', linewidth=0.5,
                   label=BACKBONE_LABELS[bb] if spec_idx == 0 else None)

        ax.set_xticks(np.arange(len(MODEL_NAMES)))
        ax.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel(f'Sensitivity at {spec_label} Specificity', fontsize=10)
        ax.set_title(f'Sensitivity at {spec_label} Specificity (OOD)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        if spec_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    _save_fig('fig7_clinical_utility')
    plt.close()


# ============================================================
# Fig 7 alternative: Decision Curve Analysis + Sensitivity@Specificity
# ============================================================

def fig7_decision_curve(results):
    from clinical_metrics import decision_curve_analysis, sensitivity_at_specificity

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Sensitivity at 95% specificity (OOD, ViT-Tiny, all models)
    bb = 'vit_tiny'
    for m_idx, model in enumerate(MODEL_NAMES):
        preds_list, labels = load_predictions(bb, model, 'ood_test')
        if not preds_list:
            continue
        seed_sens = []
        for probs in preds_list:
            sens, _ = sensitivity_at_specificity(probs, labels, 0.95)
            seed_sens.append(sens)
        mean_sens = np.mean(seed_sens)
        std_sens = np.std(seed_sens)
        color = COLORS[model]
        ax1.bar(m_idx, mean_sens, yerr=std_sens, color=color, alpha=0.8,
                edgecolor='black', linewidth=0.5, capsize=3, width=0.6)

    ax1.set_xticks(np.arange(len(MODEL_NAMES)))
    ax1.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=30, ha='right')
    ax1.set_ylabel('Sensitivity at 95% Specificity', fontsize=10)
    ax1.set_title(f'Clinical Sensitivity (OOD, {BACKBONE_LABELS[bb]})', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.05)

    # Right: Decision curve analysis (OOD, ViT-Tiny, baseline vs multi_agent)
    for model, color, label in [('baseline', '#0279EE', 'Baseline'),
                                 ('multi_agent', '#E9ED4C', 'Multi-Agent')]:
        preds_list, labels = load_predictions(bb, model, 'ood_test')
        if not preds_list:
            continue
        # Use seed 0 for DCA
        probs = preds_list[0]
        dca = decision_curve_analysis(probs, labels)
        thresholds = [d['threshold'] for d in dca]
        nb_model = [d['nb_model'] for d in dca]
        ax2.plot(thresholds, nb_model, color=color, label=label, linewidth=1.5)

    # Add treat-all and treat-none
    dca_ref = decision_curve_analysis(preds_list[0], labels) if preds_list else []
    if dca_ref:
        thresholds = [d['threshold'] for d in dca_ref]
        nb_all = [d['nb_all'] for d in dca_ref]
        ax2.plot(thresholds, nb_all, color='gray', linestyle='--', label='Treat All', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle=':', label='Treat None', linewidth=1)

    ax2.set_xlabel('Threshold Probability', fontsize=10)
    ax2.set_ylabel('Net Benefit', fontsize=10)
    ax2.set_title('Decision Curve Analysis (OOD)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(-0.1, 0.5)

    plt.tight_layout()
    _save_fig('fig7_clinical_utility')
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=== Figure Generation v3 ===")
    results = load_all_results()
    print(f"  Loaded results for backbones: {[bb for bb in BACKBONES if bb in results]}")

    has_v3 = any(results[bb].get('v3_all') for bb in BACKBONES)
    if not has_v3:
        print("  No v3 results found. Will use v2 results where available.")

    print("\nGenerating figures...")
    fig3_diagnostic_performance(results)
    fig4_perturbation_robustness(results)
    fig5_calibration(results)
    fig6_disentanglement_ablation(results)
    fig7_decision_curve(results)

    print(f"\nFigures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
