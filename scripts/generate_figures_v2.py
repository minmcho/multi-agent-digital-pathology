"""
Figure generation for v2 results (multi-seed, multi-backbone).
Generates data-driven figures (Fig 3-7 + Supp) with error bars and CI bands.
Schematic figures (Fig 1-2) are generated separately via GenerateImage.

Usage:
  python generate_figures_v2.py
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import glob

# Font and style settings
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

SHARED_DIR = "/mnt/shared-workspace/shared"
RESULTS_DIR = os.path.join(SHARED_DIR, "results_v2")
FIGURE_DIR = "/mnt/results/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Color palette
COLORS = {
    'baseline': '#0279EE',
    'dann': '#FF9400',
    'ssl': '#75A025',
    'disentanglement': '#FD9BED',
    'multi_agent': '#E9ED4C',
}
BACKBONE_COLORS = {
    'vit_tiny': '#0279EE',
    'vit_b': '#75A025',
    'resnet18': '#FF9400',
}
MODEL_NAMES = ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']
MODEL_LABELS = ['Baseline', '+DANN', '+SSL', '+Disent.', 'Multi-Agent']
BACKBONE_LABELS = {'vit_tiny': 'Phikon ViT-T', 'vit_b': 'Phikon ViT-B', 'resnet18': 'ResNet-18'}

def load_all_results():
    """Load results for all backbones."""
    results = {}
    for bb in ['vit_tiny', 'vit_b', 'resnet18']:
        path = os.path.join(RESULTS_DIR, bb, "all_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[bb] = json.load(f)
    return results

def errbar_mean_ci(ax, x_pos, mean, ci, color, width=0.15, label=None):
    """Draw a bar with error bar."""
    ax.bar(x_pos, mean, width=width, color=color, alpha=0.8, label=label,
           edgecolor='black', linewidth=0.5)
    ax.errorbar(x_pos, mean, yerr=ci, fmt='none', ecolor='black', capsize=3, linewidth=1)

# ============================================================
# Fig 3: Diagnostic Performance (AUC) — 3 backbones × 5 models
# ============================================================

def fig3_diagnostic_performance(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    datasets = [
        ('q1_diagnostic_validity', 'ID Validation (Camelyon17)'),
        ('q3_external_generalization', 'OOD Test (Camelyon17)'),
        ('q3_pcam_external', 'External (PCam)'),
    ]

    for ax_idx, (q_key, title) in enumerate(datasets):
        ax = axes[ax_idx]
        backbones = [bb for bb in ['vit_tiny', 'vit_b', 'resnet18'] if bb in all_results]
        n_bbs = len(backbones)
        width = 0.8 / n_bbs

        for bb_idx, bb in enumerate(backbones):
            if q_key not in all_results[bb]:
                continue
            q_data = all_results[bb][q_key]
            x_pos = np.arange(len(MODEL_NAMES)) + (bb_idx - n_bbs/2 + 0.5) * width

            for m_idx, model in enumerate(MODEL_NAMES):
                if model not in q_data:
                    continue
                auc_data = q_data[model].get('auc', {})
                mean = auc_data.get('mean', 0)
                ci = auc_data.get('ci', 0)
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

    # Legend
    handles = [Patch(facecolor=BACKBONE_COLORS[bb], label=BACKBONE_LABELS[bb])
               for bb in ['vit_tiny', 'vit_b', 'resnet18'] if bb in all_results]
    fig.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    _save_fig('fig3_diagnostic_performance')
    plt.close()

# ============================================================
# Fig 4: Perturbation Robustness Heatmap
# ============================================================

def fig4_perturbation_robustness(all_results):
    bb = 'vit_tiny'  # Primary backbone
    if bb not in all_results or 'q2_robustness' not in all_results[bb]:
        return

    q2 = all_results[bb]['q2_robustness']
    perturbation_types = ['gaussian_noise', 'stain_shift', 'gaussian_blur',
                          'jpeg_compression', 'brightness_contrast']
    pert_labels = ['Gaussian\nNoise', 'Stain\nShift', 'Gaussian\nBlur',
                   'JPEG\nCompression', 'Brightness/\nContrast']
    sev_labels = ['Mild', 'Moderate', 'Severe']

    fig, ax = plt.subplots(figsize=(10, 5))
    data = np.zeros((len(perturbation_types) * 3, len(MODEL_NAMES)))
    errors = np.zeros_like(data)

    for p_idx, ptype in enumerate(perturbation_types):
        for s_idx in range(3):
            row = p_idx * 3 + s_idx
            for m_idx, model in enumerate(MODEL_NAMES):
                if model in q2 and 'perturbations' in q2[model]:
                    perts = q2[model]['perturbations']
                    if ptype in perts and s_idx < len(perts[ptype]):
                        data[row, m_idx] = perts[ptype][s_idx]['auc']['mean']
                        errors[row, m_idx] = perts[ptype][s_idx]['auc']['ci']

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(np.arange(len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_LABELS, fontsize=9)
    y_labels = []
    for p_idx, plabel in enumerate(pert_labels):
        for s_idx, slabel in enumerate(sev_labels):
            y_labels.append(f"{plabel.replace(chr(10), ' ')} - {slabel}")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)

    # Add text annotations
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
# Fig 5: Risk-Coverage Curves + ECE
# ============================================================

def fig5_risk_coverage_ece(all_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ECE comparison across backbones
    backbones = [bb for bb in ['vit_tiny', 'vit_b', 'resnet18'] if bb in all_results]
    width = 0.8 / len(backbones)

    for bb_idx, bb in enumerate(backbones):
        if 'q4_uncertainty' not in all_results[bb]:
            continue
        q4 = all_results[bb]['q4_uncertainty']
        x_pos = np.arange(len(MODEL_NAMES)) + (bb_idx - len(backbones)/2 + 0.5) * width
        for m_idx, model in enumerate(MODEL_NAMES):
            if model not in q4:
                continue
            ece_data = q4[model].get('ece', {})
            mean = ece_data.get('mean', 0)
            ci = ece_data.get('ci', 0)
            color = BACKBONE_COLORS[bb]
            ax1.bar(x_pos[m_idx], mean, width=width*0.9, color=color, alpha=0.8,
                    edgecolor='black', linewidth=0.5)
            ax1.errorbar(x_pos[m_idx], mean, yerr=ci, fmt='none',
                        ecolor='black', capsize=2, linewidth=0.8)

    ax1.set_xticks(np.arange(len(MODEL_NAMES)))
    ax1.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=30, ha='right')
    ax1.set_ylabel('ECE (Expected Calibration Error)', fontsize=10)
    ax1.set_title('Uncertainty Calibration (OOD)', fontsize=11, fontweight='bold')

    handles = [Patch(facecolor=BACKBONE_COLORS[bb], label=BACKBONE_LABELS[bb])
               for bb in backbones]
    ax1.legend(handles=handles, fontsize=8)

    # Right: Mean |ΔAUC| under perturbations
    for bb_idx, bb in enumerate(backbones):
        if 'q2_robustness' not in all_results[bb]:
            continue
        q2 = all_results[bb]['q2_robustness']
        x_pos = np.arange(len(MODEL_NAMES)) + (bb_idx - len(backbones)/2 + 0.5) * width
        for m_idx, model in enumerate(MODEL_NAMES):
            if model not in q2:
                continue
            delta_data = q2[model].get('mean_abs_delta_auc', {})
            mean = delta_data.get('mean', 0)
            ci = delta_data.get('ci', 0)
            color = BACKBONE_COLORS[bb]
            ax2.bar(x_pos[m_idx], mean, width=width*0.9, color=color, alpha=0.8,
                    edgecolor='black', linewidth=0.5)
            ax2.errorbar(x_pos[m_idx], mean, yerr=ci, fmt='none',
                        ecolor='black', capsize=2, linewidth=0.8)

    ax2.set_xticks(np.arange(len(MODEL_NAMES)))
    ax2.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=30, ha='right')
    ax2.set_ylabel('Mean |ΔAUC|', fontsize=10)
    ax2.set_title('Robustness to Image Perturbations', fontsize=11, fontweight='bold')
    ax2.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    _save_fig('fig5_risk_coverage_ece')
    plt.close()

# ============================================================
# Fig 6: Disentanglement + Agent Ablation
# ============================================================

def fig6_disentanglement_ablation(all_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Disentanglement probes
    bb = 'vit_tiny'
    if bb in all_results and 'q5_disentanglement' in all_results[bb]:
        q5 = all_results[bb]['q5_disentanglement']
        probe_models = ['baseline', 'dann', 'disentanglement', 'multi_agent']
        probe_labels = ['Baseline', '+DANN', '+Disent.', 'Multi-Agent']

        # Check if Q5 data has model names (not seed numbers)
        has_model_names = all(m in q5 for m in probe_models)
        if has_model_names:
            x = np.arange(len(probe_models))
            width = 0.35

            dom_accs = [q5[m]['bio_domain_probe_acc']['mean'] for m in probe_models]
            dom_cis = [q5[m]['bio_domain_probe_acc'].get('ci', 0) for m in probe_models]
            lbl_accs = [q5[m]['bio_label_probe_acc']['mean'] for m in probe_models]
            lbl_cis = [q5[m]['bio_label_probe_acc'].get('ci', 0) for m in probe_models]

            ax1.bar(x - width/2, dom_accs, width, label='Domain probe acc (↓ better)',
                    color='#FF9400', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.errorbar(x - width/2, dom_accs, yerr=dom_cis, fmt='none', ecolor='black', capsize=2)
            ax1.bar(x + width/2, lbl_accs, width, label='Label probe acc (↑ better)',
                    color='#0279EE', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.errorbar(x + width/2, lbl_accs, yerr=lbl_cis, fmt='none', ecolor='black', capsize=2)

            ax1.set_xticks(x)
            ax1.set_xticklabels(probe_labels, fontsize=9)
            ax1.set_ylabel('Probe Accuracy', fontsize=10)
            ax1.set_title('Disentanglement Probes (Z_bio)', fontsize=11, fontweight='bold')
            ax1.legend(fontsize=8)
            ax1.set_ylim(0, 1.05)
        else:
            ax1.text(0.5, 0.5, 'Q5 data not yet available', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Disentanglement Probes (Z_bio)', fontsize=11, fontweight='bold')

    # Right: Agent ablation
    if bb in all_results and 'q5_ablation' in all_results[bb]:
        q5_abl = all_results[bb]['q5_ablation']
        abl_names = ['full_system', 'tumor_only', 'tissue_only', 'uncertainty_only']
        abl_labels = ['Full System', 'Tumor Only', 'Tissue Only', 'Uncertainty Only']
        # Check if ablation data has proper keys
        has_abl = all(name in q5_abl for name in abl_names)
        if has_abl:
            x = np.arange(len(abl_labels))
            aucs = []
            for name in abl_names:
                if isinstance(q5_abl[name], dict):
                    if 'auc' in q5_abl[name]:
                        val = q5_abl[name]['auc']
                        aucs.append(val.get('mean', val) if isinstance(val, dict) else val)
                    else:
                        aucs.append(0)
                else:
                    aucs.append(0)

            colors = ['#E9ED4C', '#0279EE', '#75A025', '#FD9BED']
            ax2.bar(x, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(abl_labels, fontsize=8, rotation=30, ha='right')
            ax2.set_ylabel('AUC (OOD Test)', fontsize=10)
            ax2.set_title('Agent Ablation Analysis', fontsize=11, fontweight='bold')
            ax2.set_ylim(0.5, 1.01)
            ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
        else:
            ax2.text(0.5, 0.5, 'Ablation data not yet available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Agent Ablation Analysis', fontsize=11, fontweight='bold')

    plt.tight_layout()
    _save_fig('fig6_disentanglement_ablation')
    plt.close()

# ============================================================
# Fig 7: Orchestrator Weights + Threshold Optimization
# ============================================================

def fig7_orchestrator_threshold(all_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Orchestrator agent weights
    bb = 'vit_tiny'
    if bb in all_results and 'q5_agent_weights' in all_results[bb]:
        weights_data = all_results[bb]['q5_agent_weights']
        agents = weights_data['agents']
        weights = np.array(weights_data['weights'])

        colors = ['#0279EE', '#75A025', '#E9ED4C', '#FF9400', '#FD9BED', '#E9ED4C']
        ax1.barh(agents, weights, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Weight', fontsize=10)
        ax1.set_title('Orchestrator Agent Weights', fontsize=11, fontweight='bold')
        ax1.invert_yaxis()

    # Right: Threshold optimization sweep
    if bb in all_results and 'threshold_optimization' in all_results[bb]:
        thresh_data = all_results[bb]['threshold_optimization']
        sweep = thresh_data.get('sweep', [])
        if sweep:
            thresholds = [s['threshold'] for s in sweep]
            flag_rates = [s['flag_rate'] for s in sweep]
            non_flag_aucs = [s['non_flagged_auc'] for s in sweep]

            ax2.plot(thresholds, flag_rates, 'o-', color='#FF9400', label='Flag rate', markersize=3)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(thresholds, non_flag_aucs, 's-', color='#0279EE',
                         label='Non-flagged AUC', markersize=3)
            ax2_twin.set_ylabel('Non-flagged AUC', fontsize=10, color='#0279EE')
            ax2_twin.spines['right'].set_visible(True)

            # Mark optimal threshold
            optimal = thresh_data.get('optimal', {})
            if optimal:
                ax2.axvline(x=optimal['threshold'], color='red', linestyle='--',
                           linewidth=1, label=f"Optimal ({optimal['threshold']:.3f})")

            ax2.set_xlabel('Disagreement Threshold', fontsize=10)
            ax2.set_ylabel('Flag Rate', fontsize=10, color='#FF9400')
            ax2.set_title('Threshold Optimization (ID Val)', fontsize=11, fontweight='bold')
            ax2.legend(loc='center left', fontsize=8)
            ax2_twin.legend(loc='center right', fontsize=8)

    plt.tight_layout()
    _save_fig('fig7_orchestrator_threshold')
    plt.close()

# ============================================================
# Supplementary: ROC Curves
# ============================================================

def fig_supp_roc_curves(all_results):
    """Generate supplementary ROC curve figure.
    Note: This requires raw predictions, which are saved per-seed.
    For the article, we'll use the AUC values to create a summary table instead.
    """
    # This will be a summary table of AUCs with CIs instead of ROC curves
    # since we don't save raw predictions in the aggregated results
    pass

# ============================================================
# Helper
# ============================================================

def _save_fig(name):
    """Save figure as PNG and SVG. Save to local disk first, then cp to S3 mount."""
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
    print(f"  Saved {name}.png and {name}.svg to {FIGURE_DIR}")

# ============================================================
# Main
# ============================================================

def main():
    print("=== Figure Generation v2 ===")
    all_results = load_all_results()
    print(f"  Loaded results for backbones: {list(all_results.keys())}")

    if not all_results:
        print("  No results found. Run experiments first.")
        return

    print("\nGenerating figures...")
    fig3_diagnostic_performance(all_results)
    fig4_perturbation_robustness(all_results)
    fig5_risk_coverage_ece(all_results)
    fig6_disentanglement_ablation(all_results)
    fig7_orchestrator_threshold(all_results)

    print(f"\nFigures saved to {FIGURE_DIR}")

if __name__ == "__main__":
    main()
