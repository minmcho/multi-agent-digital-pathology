"""
Generic re-aggregation + DeLong re-run for any backbone.
Usage: python reaggregate_delong.py --backbone vit_tiny
       python reaggregate_delong.py --backbone resnet18
"""
import numpy as np
import torch
import json, os, sys, time, argparse
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/mnt/shared-workspace/shared')
from run_experiments_v2 import (
    load_features, set_seeds, get_predictions, compute_ci, aggregate_results,
    train_baseline, train_dann, train_ssl, train_disentanglement, train_multi_agent,
    evaluate_model, evaluate_uncertainty, evaluate_disentanglement, optimize_threshold,
    BACKBONE_DIMS, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, N_DOMAINS, MC_SAMPLES
)
from pathology_models import (
    BaselineClassifier, DANNModel, SSLModel, DisentanglementModel, MultiAgentSystem
)
from delong_test import delong_paired_comparison, bonferroni_correct

SHARED_DIR = '/mnt/shared-workspace/shared'
MODEL_NAMES = ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['vit_tiny', 'vit_b', 'resnet18'], required=True)
    args = parser.parse_args()
    backbone = args.backbone

    RESULTS_DIR = os.path.join(SHARED_DIR, 'results_v2', backbone)
    CHECKPOINT_PATH = f'/workspace/checkpoint_{backbone}_seed9.json'
    ORIGINAL_RESULTS_PATH = os.path.join(RESULTS_DIR, 'all_results.json')

    t_start = time.time()
    print(f"=== {backbone} Re-aggregation + DeLong Re-run ===\n")

    # Step 1: Load checkpoint and re-aggregate
    print("Step 1: Loading checkpoint and re-aggregating Q1-Q6...")
    with open(CHECKPOINT_PATH) as f:
        per_seed = json.load(f)

    print(f"  Checkpoint keys: {list(per_seed.keys())}")
    for k in per_seed:
        if isinstance(per_seed[k], dict):
            n_seeds = len(per_seed[k])
            first = per_seed[k].get('0', {})
            models = list(first.keys())[:5] if isinstance(first, dict) else []
            print(f"    {k}: {n_seeds} seeds, models={models}")

    aggregated = aggregate_results({
        'q1_diagnostic_validity': per_seed['q1'],
        'q3_external_generalization': per_seed['q3'],
        'q3_pcam_external': per_seed['q3_pcam'],
        'q2_robustness': per_seed.get('q2', {}),
        'q4_uncertainty': per_seed['q4'],
        'q4_uncertainty_id': per_seed['q4_id'],
        'q5_disentanglement': per_seed['q5_disent'],
        'q5_ablation': per_seed['q5_ablation'],
        'q6_efficiency': per_seed['q6'],
    })
    print("\n  Aggregated results:")
    for k, v in aggregated.items():
        if isinstance(v, dict) and v:
            print(f"    {k}: models={list(v.keys())[:5]}")
        else:
            print(f"    {k}: {type(v).__name__}")

    # Step 2: Preserve non-aggregated data from original results
    print("\nStep 2: Loading threshold_optimization and q5_agent_weights from original results...")
    with open(ORIGINAL_RESULTS_PATH) as f:
        original = json.load(f)

    threshold_opt = original.get('threshold_optimization', {})
    q5_agent_weights = original.get('q5_agent_weights', {})
    training_info = original.get('training_info', {})
    print(f"  Threshold optimal: {threshold_opt.get('optimal', 'N/A')}")

    # Step 3: Re-run DeLong tests with seed-0 predictions
    print("\nStep 3: Re-training seed-0 models for DeLong tests...")
    feature_dim = BACKBONE_DIMS[backbone]

    print("  Loading features...")
    id_train_f, id_train_l, id_train_d = load_features(backbone, "camelyon17_id_train")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features(backbone, "pcam_train")
    id_val_f, id_val_l, id_val_d = load_features(backbone, "camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features(backbone, "camelyon17_ood_test")
    pcam_test_f, pcam_test_l, pcam_test_d = load_features(backbone, "pcam_test")

    train_f = np.concatenate([id_train_f, pcam_train_f])
    train_l = np.concatenate([id_train_l, pcam_train_l])
    train_d = np.concatenate([id_train_d, pcam_train_d])
    print(f"  Train: {train_f.shape}, ID val: {id_val_f.shape}, OOD: {ood_test_f.shape}, PCam: {pcam_test_f.shape}")

    set_seeds(0)

    print("  Training baseline...", end=' ')
    m1, auc1 = train_baseline(BaselineClassifier(input_dim=feature_dim), train_f, train_l, id_val_f, id_val_l)
    print(f"val AUC={auc1:.4f}")

    print("  Training DANN...", end=' ')
    m2, auc2 = train_dann(DANNModel(input_dim=feature_dim, n_domains=N_DOMAINS), train_f, train_l, train_d, id_val_f, id_val_l)
    print(f"val AUC={auc2:.4f}")

    print("  Training SSL...", end=' ')
    m3, auc3 = train_ssl(SSLModel(input_dim=feature_dim), train_f, train_l, train_d, id_val_f, id_val_l)
    print(f"val AUC={auc3:.4f}")

    print("  Training Disentanglement...", end=' ')
    m4, auc4 = train_disentanglement(DisentanglementModel(input_dim=feature_dim, n_domains=N_DOMAINS), train_f, train_l, train_d, id_val_f, id_val_l)
    print(f"val AUC={auc4:.4f}")

    print("  Training Multi-Agent...", end=' ')
    m5, auc5 = train_multi_agent(MultiAgentSystem(input_dim=feature_dim, n_domains=N_DOMAINS), train_f, train_l, train_d, id_val_f, id_val_l)
    print(f"val AUC={auc5:.4f}")

    models = {
        'baseline': (m1, 'baseline'),
        'dann': (m2, 'dann'),
        'ssl': (m3, 'ssl'),
        'disentanglement': (m4, 'disent'),
        'multi_agent': (m5, 'multi_agent'),
    }

    print("  Getting predictions...")
    predictions = {}
    for name, (model, mtype) in models.items():
        predictions[name] = {
            'id_val': get_predictions(model, id_val_f.astype(np.float32), mtype),
            'ood_test': get_predictions(model, ood_test_f.astype(np.float32), mtype),
            'pcam_test': get_predictions(model, pcam_test_f.astype(np.float32), mtype),
        }

    print("  Running DeLong tests...")
    delong_id = delong_paired_comparison(id_val_l,
        [predictions[n]['id_val'] for n in MODEL_NAMES], MODEL_NAMES)
    delong_id = bonferroni_correct(delong_id, alpha=0.05)

    delong_ood = delong_paired_comparison(ood_test_l,
        [predictions[n]['ood_test'] for n in MODEL_NAMES], MODEL_NAMES)
    delong_ood = bonferroni_correct(delong_ood, alpha=0.05)

    delong_pcam = delong_paired_comparison(pcam_test_l,
        [predictions[n]['pcam_test'] for n in MODEL_NAMES], MODEL_NAMES)
    delong_pcam = bonferroni_correct(delong_pcam, alpha=0.05)

    print(f"  DeLong id_val: {len(delong_id)} comparisons")
    for r in delong_id[:3]:
        print(f"    {r['model_1']} vs {r['model_2']}: AUC={r['auc_1']:.4f} vs {r['auc_2']:.4f}, z={r['z_score']:.2f}, p={r['p_value']:.4f}")
    print(f"  DeLong ood_test: {len(delong_ood)} comparisons")
    for r in delong_ood[:3]:
        print(f"    {r['model_1']} vs {r['model_2']}: AUC={r['auc_1']:.4f} vs {r['auc_2']:.4f}, z={r['z_score']:.2f}, p={r['p_value']:.4f}")

    # Step 3b: Re-evaluate Q5 disentanglement with fixed shuffle
    print("\n  Re-evaluating Q5 disentanglement (with fixed shuffle)...")
    q5_fixed = {}
    for name, (model, mtype) in [('baseline', (m1, 'baseline')), ('dann', (m2, 'dann')),
                                  ('disentanglement', (m4, 'disent')), ('multi_agent', (m5, 'multi_agent'))]:
        q5_fixed[name] = evaluate_disentanglement(model, train_f, train_l, train_d, mtype)
        print(f"    {name}: dom_probe={q5_fixed[name]['bio_domain_probe_acc']:.4f}, lbl_probe={q5_fixed[name]['bio_label_probe_acc']:.4f}")

    # Update aggregated Q5 with fixed seed-0 values (single-seed evaluation)
    if 'q5_disentanglement' in aggregated:
        for name in q5_fixed:
            if name in aggregated['q5_disentanglement']:
                for mk in ['bio_domain_probe_acc', 'bio_label_probe_acc', 'normalized_cross_covariance']:
                    if mk in aggregated['q5_disentanglement'][name]:
                        aggregated['q5_disentanglement'][name][mk] = {
                            'mean': q5_fixed[name][mk],
                            'ci': 0.0  # single-seed, no CI
                        }

    # Step 4: Assemble final results
    print("\nStep 4: Assembling final results...")
    final_results = dict(aggregated)
    final_results['delong_tests'] = {
        'id_val': delong_id,
        'ood_test': delong_ood,
        'pcam_test': delong_pcam,
    }
    final_results['threshold_optimization'] = threshold_opt
    final_results['q5_agent_weights'] = q5_agent_weights
    final_results['training_info'] = training_info

    # Save
    out_path = os.path.join(RESULTS_DIR, 'all_results.json')
    tmp_path = f'/workspace/{backbone}_results_fixed.json'
    with open(tmp_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    os.system(f'rm -f {out_path} && cp {tmp_path} {out_path}')
    print(f"\n  Results saved to {out_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    for q_name in ['q1_diagnostic_validity', 'q3_external_generalization', 'q3_pcam_external']:
        if q_name in final_results and final_results[q_name]:
            print(f"\n{q_name}:")
            for model_name, metrics in final_results[q_name].items():
                if 'auc' in metrics:
                    print(f"  {model_name:20s}: AUC={metrics['auc']['mean']:.4f} ± {metrics['auc']['ci']:.4f}")

    print(f"\nTotal time: {(time.time() - t_start)/60:.1f} min")

if __name__ == '__main__':
    main()
