#!/usr/bin/env python
"""
Run MMP-OOD Evaluation for ENIGMA Competition
================================================

End-to-end convenience script that:
1. Loads a trained model checkpoint.
2. Runs inference on the MMP-OOD test set.
3. Computes Macro-F1 and Pairwise Cliff Accuracy.
4. Prints and optionally saves a report.

Usage:
    python scripts/run_mmp_evaluation.py --checkpoint <model.pt> --model graphsage
    python scripts/run_mmp_evaluation.py --submission submissions/gcn_submission.csv
    python scripts/run_mmp_evaluation.py --submission submissions/gcn_submission.csv --output-json results/mmp_results.json
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def evaluate_from_submission(
    submission_path: str,
    pairs_path: str,
    truth_path: str,
) -> dict:
    """
    Evaluate a submission CSV against MMP-OOD pairs.

    Args:
        submission_path: Path to submission CSV (columns: id, target).
        pairs_path: Path to pairs.csv from MMP-OOD split.
        truth_path: Path to ground-truth labels CSV (columns: id, target).

    Returns:
        Dictionary of metrics.
    """
    from evaluation.mmp_ood import (
        load_pairs_csv,
        evaluate_mmp_ood,
    )

    submission_df = pd.read_csv(submission_path)
    truth_df = pd.read_csv(truth_path)
    pairs = load_pairs_csv(pairs_path)

    # Build dictionaries: mol_idx -> label
    y_true = dict(zip(truth_df['id'].values, truth_df['target'].values))
    y_pred = dict(zip(submission_df['id'].values, submission_df['target'].values))

    # Check for probability column
    y_prob = None
    if 'probability' in submission_df.columns:
        y_prob = dict(zip(submission_df['id'].values, submission_df['probability'].values))

    # Get test indices (all molecules that appear in pairs)
    test_mol_set = set()
    for p in pairs:
        test_mol_set.add(p.mol_idx_a)
        test_mol_set.add(p.mol_idx_b)
    test_idx = np.array(sorted(test_mol_set))

    metrics = evaluate_mmp_ood(
        pairs=pairs,
        test_idx=test_idx,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    return metrics


def evaluate_from_checkpoint(
    checkpoint_path: str,
    model_name: str,
    mmp_split_dir: str,
    ogb_root: str,
    hidden: int = 64,
    batch_size: int = 32,
) -> dict:
    """
    Load a model checkpoint, run inference on MMP-OOD test set, and evaluate.

    Args:
        checkpoint_path: Path to model .pt checkpoint.
        model_name: One of 'graphsage', 'gcn', 'gin', 'dmpnn', 'spectral'.
        mmp_split_dir: Path to MMP-OOD split directory.
        ogb_root: Path to OGB data root.
        hidden: Hidden dimension (must match checkpoint).
        batch_size: Batch size for inference.

    Returns:
        Dictionary of metrics.
    """
    import torch
    from torch_geometric.loader import DataLoader
    from ogb.graphproppred import PygGraphPropPredDataset

    # Fix for PyTorch 2.6+ compatibility
    import torch.serialization
    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
    except (ImportError, AttributeError):
        pass

    from evaluation.mmp_ood import load_pairs_csv, load_split_indices, evaluate_mmp_ood

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = PygGraphPropPredDataset(name='ogbg-molbace', root=ogb_root)

    # Load MMP-OOD split
    train_idx, valid_idx, test_idx = load_split_indices(mmp_split_dir)
    pairs = load_pairs_csv(os.path.join(mmp_split_dir, 'pairs.csv'))

    test_loader = DataLoader(
        dataset[torch.tensor(test_idx)],
        batch_size=batch_size,
        shuffle=False,
    )

    # Load model
    if model_name in ('graphsage', 'gcn', 'gin'):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'starter_code'))
        from baseline import GraphSAGEModel, GCNModel, GINModel
        model_classes = {
            'graphsage': GraphSAGEModel,
            'gcn': GCNModel,
            'gin': GINModel,
        }
        model = model_classes[model_name](
            in_channels=dataset.num_node_features,
            hidden_channels=hidden,
            out_channels=2,
        ).to(device)
    elif model_name == 'dmpnn':
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'advanced_baselines'))
        from dmpnn import DMPNNModel
        model = DMPNNModel(
            in_channels=dataset.num_node_features,
            edge_channels=3,
            hidden_channels=hidden,
            out_channels=2,
            num_layers=3,
        ).to(device)
    elif model_name == 'spectral':
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'advanced_baselines'))
        from spectral_gnn import SpectralGNN
        model = SpectralGNN(
            in_channels=dataset.num_node_features,
            hidden_channels=hidden,
            out_channels=2,
            num_layers=3,
            K=3,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Run inference
    y_true_dict = {}
    y_pred_dict = {}
    y_prob_dict = {}

    with torch.no_grad():
        # We need to track which mol indices correspond to which batch items
        for i, idx in enumerate(test_idx):
            data = dataset[int(idx)].to(device)
            # Add batch dimension if needed
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            out = model(data)
            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1).item()
            prob_active = probs[0, 1].item()

            y_true_dict[int(idx)] = int(data.y.item())
            y_pred_dict[int(idx)] = pred
            y_prob_dict[int(idx)] = prob_active

    metrics = evaluate_mmp_ood(
        pairs=pairs,
        test_idx=test_idx,
        y_true=y_true_dict,
        y_pred=y_pred_dict,
        y_prob=y_prob_dict,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Run MMP-OOD evaluation on a trained model or submission'
    )

    # Two modes: from submission CSV or from checkpoint
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--submission', type=str,
        help='Path to submission CSV (columns: id, target [, probability])'
    )
    group.add_argument(
        '--checkpoint', type=str,
        help='Path to model .pt checkpoint'
    )

    parser.add_argument(
        '--model', type=str, default='graphsage',
        choices=['graphsage', 'gcn', 'gin', 'dmpnn', 'spectral'],
        help='Model architecture (required with --checkpoint)'
    )
    parser.add_argument(
        '--hidden', type=int, default=64,
        help='Hidden dimension (must match checkpoint)'
    )
    parser.add_argument(
        '--pairs-csv', type=str, default=None,
        help='Path to pairs.csv (default: data/mmp_split/pairs.csv)'
    )
    parser.add_argument(
        '--truth', type=str, default=None,
        help='Path to ground-truth labels CSV (default: data/test_labels.csv)'
    )
    parser.add_argument(
        '--mmp-split-dir', type=str, default=None,
        help='Path to MMP-OOD split directory (default: data/mmp_split)'
    )
    parser.add_argument(
        '--ogb-root', type=str, default=None,
        help='Path to OGB data root (default: data/ogb)'
    )
    parser.add_argument(
        '--output-json', type=str, default=None,
        help='Save metrics to JSON file'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for inference'
    )

    args = parser.parse_args()

    # Resolve default paths
    mmp_split_dir = args.mmp_split_dir or os.path.join(PROJECT_ROOT, 'data', 'mmp_split')
    ogb_root = args.ogb_root or os.path.join(PROJECT_ROOT, 'data', 'ogb')
    pairs_csv = args.pairs_csv or os.path.join(mmp_split_dir, 'pairs.csv')
    truth_path = args.truth or os.path.join(PROJECT_ROOT, 'data', 'test_labels.csv')

    print("=" * 60)
    print("  MMP-OOD Evaluation")
    print("=" * 60)

    if args.submission:
        # --- Evaluate from submission CSV ---
        print(f"\nMode: Submission CSV")
        print(f"Submission: {args.submission}")
        print(f"Pairs CSV:  {pairs_csv}")
        print(f"Truth:      {truth_path}")

        if not os.path.exists(args.submission):
            print(f"\nError: Submission file not found: {args.submission}")
            sys.exit(1)
        if not os.path.exists(pairs_csv):
            print(f"\nError: Pairs file not found: {pairs_csv}")
            print("Run 'python scripts/generate_mmp_split.py' first.")
            sys.exit(1)
        if not os.path.exists(truth_path):
            print(f"\nError: Truth file not found: {truth_path}")
            sys.exit(1)

        metrics = evaluate_from_submission(args.submission, pairs_csv, truth_path)

    else:
        # --- Evaluate from checkpoint ---
        print(f"\nMode: Model Checkpoint")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Model:      {args.model}")
        print(f"Split dir:  {mmp_split_dir}")

        if not os.path.exists(args.checkpoint):
            print(f"\nError: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        if not os.path.exists(mmp_split_dir):
            print(f"\nError: MMP-OOD split not found: {mmp_split_dir}")
            print("Run 'python scripts/generate_mmp_split.py' first.")
            sys.exit(1)

        metrics = evaluate_from_checkpoint(
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            mmp_split_dir=mmp_split_dir,
            ogb_root=ogb_root,
            hidden=args.hidden,
            batch_size=args.batch_size,
        )

    # --- Display results ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  🎯 Macro-F1 (MMP-OOD test):    {metrics.macro_f1:.4f}")
    print(f"  🧬 Pairwise Cliff Accuracy:    {metrics.cliff_accuracy:.4f}")
    print(f"     Pairs evaluated:            {metrics.num_pairs_evaluated}")
    print(f"     Pairs correct:              {sum(metrics.per_pair_results)}")
    print(f"     Test molecules:             {metrics.num_test_molecules}")
    print("\n" + "=" * 60)

    # Machine-readable output
    print(f"MMP_MACRO_F1:{metrics.macro_f1:.6f}")
    print(f"CLIFF_ACC:{metrics.cliff_accuracy:.6f}")

    # Save JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nMetrics saved to: {args.output_json}")


if __name__ == "__main__":
    main()
