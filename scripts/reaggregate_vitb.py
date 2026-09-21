"""
Re-aggregate ViT-B results from checkpoint files and re-run DeLong tests.
The original aggregation was broken (iterated over seed numbers instead of model names).
Checkpoint files have all per-seed data; we just need to re-aggregate.
DeLong tests need seed-0 predictions, so we re-train seed-0 models.
"""
import numpy as np
import torch
import json, os, sys, time
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, '/mnt/shared-workspace/shared')
from run_experiments_v2 import (
    load_features, set_seeds, get_predictions, compute_ci, aggregate_results,
    train_baseline, train_dann, train_ssl, train_disentanglement, train_multi_agent,
    evaluate_model, optimize_threshold,
    BACKBONE_DIMS, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, N_DOMAINS, MC_SAMPLES
)
from pathology_models import (
    BaselineClassifier, DANNModel, SSLModel, DisentanglementModel, MultiAgentSystem
)
from delong_test import delong_paired_comparison, bonferroni_correct

BACKBONE = 'vit_b'
SHARED_DIR = '/mnt/shared-workspace/shared'
RESULTS_DIR = os.path.join(SHARED_DIR, 'results_v2', BACKBONE)
CHECKPOINT_PATH = '/workspace/checkpoint_vit_b_seed9.json'
ORIGINAL_RESULTS_PATH = os.path.join(RESULTS_DIR, 'all_results.json')

MODEL_NAMES = ['baseline', 'dann', 'ssl', 'disentanglement', 'multi_agent']

def main():
    t_start = time.time()
    print("=== ViT-B Re-aggregation + DeLong Re-run ===\n")

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

    # Map short checkpoint keys to full names (same as run_experiments_v2.py main)
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

    # Step 2: Preserve threshold_optimization and q5_agent_weights from original
    print("\nStep 2: Loading threshold_optimization and q5_agent_weights from original results...")
    with open(ORIGINAL_RESULTS_PATH) as f:
        original = json.load(f)

    threshold_opt = original['threshold_optimization']
    q5_agent_weights = original['q5_agent_weights']
    training_info = original['training_info']
    print(f"  Threshold optimal: {threshold_opt['optimal']}")
    print(f"  Agent weights: {q5_agent_weights['agents']}")

    # Step 3: Re-run DeLong tests with seed-0 predictions
    print("\nStep 3: Re-training seed-0 models for DeLong tests...")
    feature_dim = BACKBONE_DIMS[BACKBONE]

    # Load features
    print("  Loading features...")
    id_train_f, id_train_l, id_train_d = load_features(BACKBONE, "camelyon17_id_train")
    pcam_train_f, pcam_train_l, pcam_train_d = load_features(BACKBONE, "pcam_train")
    id_val_f, id_val_l, id_val_d = load_features(BACKBONE, "camelyon17_id_val")
    ood_test_f, ood_test_l, ood_test_d = load_features(BACKBONE, "camelyon17_ood_test")
    pcam_test_f, pcam_test_l, pcam_test_d = load_features(BACKBONE, "pcam_test")

    train_f = np.concatenate([id_train_f, pcam_train_f])
    train_l = np.concatenate([id_train_l, pcam_train_l])
    train_d = np.concatenate([id_train_d, pcam_train_d])
    print(f"  Train: {train_f.shape}, ID val: {id_val_f.shape}, OOD: {ood_test_f.shape}, PCam: {pcam_test_f.shape}")

    set_seeds(0)

    # Train all 5 models with seed 0
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

    # Get predictions on all test sets
    print("  Getting predictions...")
    predictions = {}
    for name, (model, mtype) in models.items():
        predictions[name] = {
            'id_val': get_predictions(model, id_val_f.astype(np.float32), mtype),
            'ood_test': get_predictions(model, ood_test_f.astype(np.float32), mtype),
            'pcam_test': get_predictions(model, pcam_test_f.astype(np.float32), mtype),
        }

    # Run DeLong tests
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

    print(f"  DeLong id_val: {len(delong_id)} comparisons, first: {delong_id[0]}")
    print(f"  DeLong ood_test: {len(delong_ood)} comparisons, first: {delong_ood[0]}")
    print(f"  DeLong pcam_test: {len(delong_pcam)} comparisons, first: {delong_pcam[0]}")

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

    # Save (local first, then cp to S3)
    out_path = os.path.join(RESULTS_DIR, 'all_results.json')
    tmp_path = '/workspace/vit_b_results_fixed.json'
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
