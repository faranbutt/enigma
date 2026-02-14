#!/usr/bin/env python
"""
Generate MMP-OOD Split for ENIGMA Competition
================================================

This script reads the OGB MolBACE dataset, identifies activity-cliff pairs
(structurally similar molecules with opposite activity labels), and generates
a Matched-Molecular-Pair Out-of-Distribution (MMP-OOD) split.

Outputs (written to ``data/mmp_split/``):
    - train.csv   — training indices (scaffold-excluded)
    - valid.csv   — validation indices
    - test.csv    — test indices (cliff-pair molecules)
    - pairs.csv   — activity-cliff pair metadata
    - stats.json  — split statistics

Usage:
    python scripts/generate_mmp_split.py
    python scripts/generate_mmp_split.py --threshold 0.65 --exclusion relaxed
    python scripts/generate_mmp_split.py --verbose
"""

import os
import sys
import argparse
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.mmp_ood import (
    load_smiles_from_ogb,
    build_mmp_ood_split,
    save_split_indices,
    RDKIT_AVAILABLE,
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate MMP-OOD split for the ENIGMA competition'
    )
    parser.add_argument(
        '--ogb-root', type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'ogb'),
        help='Root directory of the OGB data'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'mmp_split'),
        help='Directory to write the split files'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.7,
        help='Tanimoto similarity threshold for cliff pairs (default: 0.7)'
    )
    parser.add_argument(
        '--fallback-threshold', type=float, default=0.6,
        help='Fallback threshold if too few pairs found (default: 0.6)'
    )
    parser.add_argument(
        '--min-pairs', type=int, default=5,
        help='Minimum number of pairs before falling back (default: 5)'
    )
    parser.add_argument(
        '--exclusion', type=str, default='strict',
        choices=['strict', 'relaxed'],
        help='Scaffold exclusion mode: strict or relaxed (default: strict)'
    )
    parser.add_argument(
        '--valid-ratio', type=float, default=0.1,
        help='Fraction of non-test data for validation (default: 0.1)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print detailed progress'
    )
    args = parser.parse_args()

    if not RDKIT_AVAILABLE:
        print("Error: RDKit is required but not installed.")
        print("Install with: pip install rdkit")
        sys.exit(1)

    print("=" * 60)
    print("  MMP-OOD Split Generation")
    print("=" * 60)

    # Step 1: Load SMILES and labels from OGB
    print(f"\nLoading SMILES from OGB ({args.ogb_root})...")
    mol_df = load_smiles_from_ogb(args.ogb_root)
    print(f"  Total molecules: {len(mol_df)}")
    print(f"  Label distribution: {mol_df['label'].value_counts().to_dict()}")

    smiles_list = mol_df['smiles'].tolist()
    labels = np.asarray(mol_df['label'].values)

    # Step 2: Build the MMP-OOD split
    print(f"\nBuilding MMP-OOD split...")
    print(f"  Tanimoto threshold: {args.threshold}")
    print(f"  Fallback threshold: {args.fallback_threshold}")
    print(f"  Scaffold exclusion: {args.exclusion}")

    split = build_mmp_ood_split(
        smiles_list=smiles_list,
        labels=labels,
        tanimoto_threshold=args.threshold,
        scaffold_exclusion=args.exclusion,
        valid_ratio=args.valid_ratio,
        fallback_threshold=args.fallback_threshold,
        min_pairs=args.min_pairs,
        random_seed=args.seed,
        verbose=args.verbose or True,  # always verbose for this script
    )

    # Step 3: Save outputs
    print(f"\nSaving split to: {args.output_dir}")
    save_split_indices(split, args.output_dir)

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("  MMP-OOD Split Summary")
    print("=" * 60)
    print(f"\n  Activity-cliff pairs:  {split.stats.get('num_pairs', 0)}")
    print(f"  Tanimoto threshold:    {split.stats.get('threshold_used', 'N/A')}")
    if split.pairs:
        print(f"  Tanimoto range:        [{split.stats['tanimoto_min']:.3f}, "
              f"{split.stats['tanimoto_max']:.3f}]")
        print(f"  Tanimoto mean:         {split.stats['tanimoto_mean']:.3f}")
    print(f"\n  Train molecules:       {len(split.train_idx)}")
    print(f"  Valid molecules:       {len(split.valid_idx)}")
    print(f"  Test molecules:        {len(split.test_idx)}")
    print(f"  Excluded (scaffold):   {split.stats.get('num_excluded_by_scaffold', 0)}")
    print(f"  Unique test scaffolds: {split.stats.get('num_unique_test_scaffolds', 0)}")

    print(f"\n  Output files:")
    for fname in ['train.csv', 'valid.csv', 'test.csv', 'pairs.csv', 'stats.json']:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            print(f"    ✅ {fpath}")
        else:
            print(f"    ❌ {fpath} (missing)")

    print("\n" + "=" * 60)
    if split.pairs:
        print("  ✅ MMP-OOD split generated successfully!")
    else:
        print("  ⚠️  No activity-cliff pairs found. Try lowering --threshold.")
    print("=" * 60)


if __name__ == "__main__":
    main()
