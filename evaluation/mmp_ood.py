"""
Matched Molecular Pair Out-of-Distribution (MMP-OOD) Evaluation
================================================================

This module provides tools for constructing and evaluating Matched Molecular
Pair (MMP) OOD splits that stress-test a model's ability to handle activity
cliffs — pairs of structurally similar molecules with opposite activity labels.

Concepts:
---------

1. Activity Cliff:
   A pair of molecules (A, B) such that:
       Tanimoto(FP(A), FP(B)) ≥ τ   (structurally similar, e.g. τ = 0.7)
       y_A ≠ y_B                      (discordant labels)

   These are "counterfactual" scenarios: a tiny structural edit flips the
   activity, and a model must be sensitive enough to capture this.

2. MMP-OOD Split:
   - **Test set**: composed of activity-cliff pairs.
   - **Training set**: excludes all molecules sharing a Murcko scaffold
     with any test-set molecule, ensuring the cliff chemotypes are truly
     out-of-distribution.

3. Pairwise Cliff Accuracy:
   For each cliff pair (A_active, B_inactive), check whether the model
   assigns a higher score (or the correct hard label) to the active molecule.

   $$\\text{Cliff Acc} = \\frac{1}{|P|} \\sum_{(a,b) \\in P}
       \\mathbb{1}[\\hat{y}_a > \\hat{y}_b]$$

   where $P$ is the set of directed pairs (active, inactive).

Evaluation Metrics:
-------------------

- **Macro-F1** on the full MMP-OOD test set.
- **Pairwise Cliff Accuracy** over the activity-cliff pairs.

References:
----------
- Stumpfe & Bajorath, "Exploring Activity Cliffs in Medicinal Chemistry"
  (J. Med. Chem. 2012)
- van Tilborg et al., "Exposing the Limitations of Molecular Machine Learning
  with Activity Cliffs" (J. Chem. Inf. Model. 2022) — MoleculeACE benchmark
- Hu et al., "Open Graph Benchmark" (NeurIPS 2020)

Author: ENIGMA Competition
License: MIT
"""

import os
import gzip
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Set

from sklearn.metrics import f1_score

# RDKit imports (required for MMP-OOD)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn(
        "RDKit is not installed. MMP-OOD functionality requires rdkit>=2023.3.1. "
        "Install with: pip install rdkit"
    )

if TYPE_CHECKING:
    from rdkit import Chem as Chem
    from rdkit.Chem import AllChem as AllChem, DataStructs as DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold as MurckoScaffold


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ActivityCliffPair:
    """
    A single activity-cliff pair.

    Attributes:
        mol_idx_a: Dataset index of the **active** molecule (label = 1).
        mol_idx_b: Dataset index of the **inactive** molecule (label = 0).
        tanimoto: Tanimoto similarity between the two fingerprints.
        smiles_a: SMILES of molecule A (optional, for logging).
        smiles_b: SMILES of molecule B (optional, for logging).
    """
    mol_idx_a: int          # active
    mol_idx_b: int          # inactive
    tanimoto: float
    smiles_a: str = ""
    smiles_b: str = ""


@dataclass
class MmpOodSplit:
    """
    Container for a complete MMP-OOD split.

    Attributes:
        train_idx: Indices for training (scaffold-excluded).
        valid_idx: Indices for validation.
        test_idx: Indices for testing (contains cliff-pair molecules).
        pairs: List of ActivityCliffPair objects.
        excluded_scaffolds: Set of canonical scaffold SMILES removed from train.
        stats: Dictionary with split statistics.
    """
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    pairs: List[ActivityCliffPair] = field(default_factory=list)
    excluded_scaffolds: Set[str] = field(default_factory=set)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MmpOodMetrics:
    """
    Container for MMP-OOD evaluation results.

    Attributes:
        macro_f1: Macro-averaged F1 on the MMP-OOD test set.
        cliff_accuracy: Fraction of cliff pairs where the model correctly
            ranks the active molecule above the inactive one.
        num_pairs_evaluated: Number of cliff pairs used in evaluation.
        num_test_molecules: Total molecules in the MMP-OOD test set.
        per_pair_results: List of booleans (True = correct pair ordering).
    """
    macro_f1: float = 0.0
    cliff_accuracy: float = 0.0
    num_pairs_evaluated: int = 0
    num_test_molecules: int = 0
    per_pair_results: List[bool] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'macro_f1': round(self.macro_f1, 6),
            'cliff_accuracy': round(self.cliff_accuracy, 6),
            'num_pairs_evaluated': self.num_pairs_evaluated,
            'num_test_molecules': self.num_test_molecules,
            'pairs_correct': sum(self.per_pair_results),
            'pairs_total': len(self.per_pair_results),
        }

    def __repr__(self) -> str:
        return (
            f"MmpOodMetrics(macro_f1={self.macro_f1:.4f}, "
            f"cliff_accuracy={self.cliff_accuracy:.4f}, "
            f"pairs={self.num_pairs_evaluated})"
        )


# ---------------------------------------------------------------------------
# Molecular utilities
# ---------------------------------------------------------------------------

def load_smiles_from_ogb(ogb_root: str, dataset_name: str = 'ogbg_molbace') -> pd.DataFrame:
    """
    Load SMILES strings from the OGB mapping file.

    Args:
        ogb_root: Root directory of the OGB data (contains the dataset folder).
        dataset_name: OGB dataset folder name (default: ``ogbg_molbace``).

    Returns:
        DataFrame with columns ``['mol_id', 'smiles', 'label']``.
    """
    mapping_dir = os.path.join(ogb_root, dataset_name, 'mapping')

    # The mapping file is mol.csv.gz
    mol_csv_gz = os.path.join(mapping_dir, 'mol.csv.gz')
    if not os.path.exists(mol_csv_gz):
        raise FileNotFoundError(
            f"OGB mapping file not found at {mol_csv_gz}. "
            f"Make sure the dataset is downloaded."
        )

    df = pd.read_csv(mol_csv_gz)
    # OGB mapping CSVs have columns: [task_col(s)..., smiles, mol_id]
    # For molbace: columns are ['Class', 'smiles'] (mol_id is the row index)
    if 'smiles' not in df.columns:
        # Try alternative column names
        smiles_cols = [c for c in df.columns if 'smiles' in c.lower()]
        if smiles_cols:
            df = df.rename(columns={smiles_cols[0]: 'smiles'})
        else:
            raise ValueError(f"No SMILES column found in {mol_csv_gz}. Columns: {df.columns.tolist()}")

    # Identify the label column (first column that isn't smiles/mol_id)
    label_col = None
    for c in df.columns:
        if c not in ('smiles', 'mol_id'):
            label_col = c
            break

    result = pd.DataFrame({
        'mol_id': range(len(df)),
        'smiles': df['smiles'].values,
        'label': df[label_col].values if label_col else np.nan,
    })
    return result


def compute_ecfp_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[List[Optional[Any]], List[bool]]:
    """
    Compute ECFP (Morgan) fingerprints for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings.
        radius: Morgan radius (2 = ECFP4).
        n_bits: Number of fingerprint bits.

    Returns:
        Tuple of (fingerprint_list, valid_mask).
        ``fingerprint_list[i]`` is None if the SMILES could not be parsed.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for fingerprint computation.")

    fps = []
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                # Use MorganGenerator (rdkit >= 2024) if available
                gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)  # type: ignore[attr-defined]
                fp = gen.GetFingerprint(mol)
            except AttributeError:
                # Fallback for older rdkit versions
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)  # type: ignore[attr-defined]
            fps.append(fp)
            valid.append(True)
        else:
            fps.append(None)
            valid.append(False)
    return fps, valid


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """
    Return the canonical Murcko scaffold SMILES for a molecule.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical scaffold SMILES, or None if parsing fails.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for scaffold computation.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MMP-OOD split construction
# ---------------------------------------------------------------------------

def find_activity_cliff_pairs(
    smiles_list: List[str],
    labels: np.ndarray,
    tanimoto_threshold: float = 0.7,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    verbose: bool = True,
) -> List[ActivityCliffPair]:
    """
    Identify activity-cliff pairs: high Tanimoto similarity + discordant labels.

    For every pair (i, j) with i < j, Tanimoto(FP_i, FP_j) ≥ threshold and
    label_i ≠ label_j, create an ``ActivityCliffPair`` (active first).

    Args:
        smiles_list: SMILES for all molecules.
        labels: Binary labels (0/1) aligned with ``smiles_list``.
        tanimoto_threshold: Minimum Tanimoto similarity to qualify as a pair.
        fingerprint_radius: Morgan fingerprint radius.
        fingerprint_bits: Number of fingerprint bits.
        verbose: Print progress information.

    Returns:
        List of ``ActivityCliffPair`` objects (active molecule always first).
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for MMP-OOD.")

    n = len(smiles_list)
    labels = np.asarray(labels).flatten()
    assert len(labels) == n, "labels length must match smiles_list"

    if verbose:
        print(f"Computing ECFP{2 * fingerprint_radius} fingerprints for {n} molecules...")

    fps, valid = compute_ecfp_fingerprints(
        smiles_list, radius=fingerprint_radius, n_bits=fingerprint_bits
    )

    num_valid = sum(valid)
    if verbose:
        print(f"  Valid molecules: {num_valid}/{n}")

    # Build index lists for active (1) and inactive (0)
    active_idx = [i for i in range(n) if valid[i] and labels[i] == 1]
    inactive_idx = [i for i in range(n) if valid[i] and labels[i] == 0]

    if verbose:
        print(f"  Active: {len(active_idx)}, Inactive: {len(inactive_idx)}")
        print(f"Scanning for activity cliff pairs (Tanimoto ≥ {tanimoto_threshold})...")

    pairs: List[ActivityCliffPair] = []

    # Only compare active vs inactive (we need discordant labels)
    for ai in active_idx:
        fp_a = fps[ai]
        for bi in inactive_idx:
            fp_b = fps[bi]
            sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)  # type: ignore[arg-type]
            if sim >= tanimoto_threshold:
                pairs.append(ActivityCliffPair(
                    mol_idx_a=ai,
                    mol_idx_b=bi,
                    tanimoto=sim,
                    smiles_a=smiles_list[ai],
                    smiles_b=smiles_list[bi],
                ))

    if verbose:
        print(f"  Found {len(pairs)} activity-cliff pairs")
        if pairs:
            sims = [p.tanimoto for p in pairs]
            print(f"  Tanimoto range: [{min(sims):.3f}, {max(sims):.3f}], "
                  f"mean: {np.mean(sims):.3f}")

    return pairs


def build_mmp_ood_split(
    smiles_list: List[str],
    labels: np.ndarray,
    tanimoto_threshold: float = 0.7,
    scaffold_exclusion: str = 'strict',
    valid_ratio: float = 0.1,
    fallback_threshold: float = 0.6,
    min_pairs: int = 5,
    random_seed: int = 42,
    verbose: bool = True,
) -> MmpOodSplit:
    """
    Build a complete MMP-OOD split.

    Steps:
        1. Find activity-cliff pairs (fall back to lower threshold if needed).
        2. Collect all molecules involved in cliff pairs → test set.
        3. Compute Murcko scaffolds for test molecules.
        4. Exclude molecules sharing those scaffolds from training data.
        5. Split remaining molecules into train / valid.

    Args:
        smiles_list: SMILES for all molecules.
        labels: Binary labels aligned with ``smiles_list``.
        tanimoto_threshold: Primary Tanimoto threshold.
        scaffold_exclusion: ``'strict'`` (exclude full Murcko scaffolds) or
            ``'relaxed'`` (exclude only molecules with Tanimoto ≥ 0.5 to any
            test molecule).
        valid_ratio: Fraction of non-test data reserved for validation.
        fallback_threshold: If fewer than ``min_pairs`` are found at the
            primary threshold, retry at this threshold.
        min_pairs: Minimum acceptable number of cliff pairs.
        random_seed: Random seed for train/valid partitioning.
        verbose: Print progress.

    Returns:
        ``MmpOodSplit`` with train/valid/test indices and pair metadata.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for MMP-OOD split construction.")

    n = len(smiles_list)
    labels = np.asarray(labels).flatten()
    rng = np.random.RandomState(random_seed)

    # Step 1: Find activity-cliff pairs
    pairs = find_activity_cliff_pairs(
        smiles_list, labels,
        tanimoto_threshold=tanimoto_threshold,
        verbose=verbose,
    )

    # Fallback to lower threshold if too few pairs
    used_threshold = tanimoto_threshold
    if len(pairs) < min_pairs and fallback_threshold < tanimoto_threshold:
        if verbose:
            print(f"\nToo few pairs ({len(pairs)}). "
                  f"Retrying with threshold {fallback_threshold}...")
        pairs = find_activity_cliff_pairs(
            smiles_list, labels,
            tanimoto_threshold=fallback_threshold,
            verbose=verbose,
        )
        used_threshold = fallback_threshold

    if len(pairs) == 0:
        warnings.warn(
            "No activity-cliff pairs found. Returning empty MMP-OOD split."
        )
        return MmpOodSplit(
            train_idx=np.arange(n),
            valid_idx=np.array([], dtype=int),
            test_idx=np.array([], dtype=int),
            pairs=[],
            stats={'num_pairs': 0, 'threshold_used': used_threshold},
        )

    # Step 2: Collect test molecule indices (unique)
    test_set: Set[int] = set()
    for p in pairs:
        test_set.add(p.mol_idx_a)
        test_set.add(p.mol_idx_b)
    test_idx = np.array(sorted(test_set))

    if verbose:
        print(f"\nMMP-OOD test set: {len(test_idx)} molecules from {len(pairs)} pairs")

    # Step 3: Compute scaffolds for test molecules
    test_scaffolds: Set[str] = set()
    scaffold_map: Dict[int, Optional[str]] = {}  # idx -> scaffold SMILES
    for idx in test_idx:
        scaf = get_murcko_scaffold(smiles_list[idx])
        scaffold_map[idx] = scaf
        if scaf is not None:
            test_scaffolds.add(scaf)

    if verbose:
        print(f"Unique test scaffolds: {len(test_scaffolds)}")

    # Step 4: Exclude molecules sharing scaffolds from training
    excluded: Set[int] = set(test_set)

    if scaffold_exclusion == 'strict':
        # Compute scaffold for every molecule and exclude matches
        if verbose:
            print("Strict scaffold exclusion: computing all scaffolds...")
        for i in range(n):
            if i in excluded:
                continue
            scaf = get_murcko_scaffold(smiles_list[i])
            if scaf is not None and scaf in test_scaffolds:
                excluded.add(i)
    elif scaffold_exclusion == 'relaxed':
        # Exclude molecules with Tanimoto ≥ 0.5 to any test molecule
        if verbose:
            print("Relaxed exclusion: checking similarity to test molecules...")
        fps, valid = compute_ecfp_fingerprints(smiles_list)
        test_fps = [fps[i] for i in test_idx if fps[i] is not None]
        for i in range(n):
            if i in excluded or not valid[i]:
                continue
            fp_i = fps[i]
            if fp_i is None:
                continue
            for tfp in test_fps:
                sim = DataStructs.TanimotoSimilarity(fp_i, tfp)
                if sim >= 0.5:
                    excluded.add(i)
                    break
    else:
        raise ValueError(f"Unknown scaffold_exclusion mode: {scaffold_exclusion}")

    # Step 5: Split remaining into train/valid
    remaining = np.array(sorted(set(range(n)) - excluded))
    rng.shuffle(remaining)

    n_valid = max(1, int(len(remaining) * valid_ratio))
    valid_idx = remaining[:n_valid]
    train_idx = remaining[n_valid:]

    if verbose:
        print(f"\nSplit summary:")
        print(f"  Train:    {len(train_idx)}")
        print(f"  Valid:    {len(valid_idx)}")
        print(f"  Test:     {len(test_idx)}")
        print(f"  Excluded: {len(excluded) - len(test_idx)} (scaffold overlap)")

    stats = {
        'num_pairs': len(pairs),
        'num_test_molecules': len(test_idx),
        'num_train_molecules': len(train_idx),
        'num_valid_molecules': len(valid_idx),
        'num_excluded_by_scaffold': len(excluded) - len(test_idx),
        'num_unique_test_scaffolds': len(test_scaffolds),
        'threshold_used': used_threshold,
        'scaffold_exclusion': scaffold_exclusion,
        'tanimoto_mean': float(np.mean([p.tanimoto for p in pairs])),
        'tanimoto_min': float(np.min([p.tanimoto for p in pairs])),
        'tanimoto_max': float(np.max([p.tanimoto for p in pairs])),
    }

    return MmpOodSplit(
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        pairs=pairs,
        excluded_scaffolds=test_scaffolds,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Saving / loading pairs and splits
# ---------------------------------------------------------------------------

def save_pairs_csv(pairs: List[ActivityCliffPair], path: str) -> None:
    """
    Save activity-cliff pairs to a CSV file.

    Columns: ``mol_idx_a, mol_idx_b, tanimoto, smiles_a, smiles_b``

    Args:
        pairs: List of ``ActivityCliffPair``.
        path: Output CSV path.
    """
    rows = []
    for p in pairs:
        rows.append({
            'mol_idx_a': p.mol_idx_a,
            'mol_idx_b': p.mol_idx_b,
            'tanimoto': round(p.tanimoto, 6),
            'smiles_a': p.smiles_a,
            'smiles_b': p.smiles_b,
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path, index=False)


def load_pairs_csv(path: str) -> List[ActivityCliffPair]:
    """
    Load activity-cliff pairs from a CSV file.

    Args:
        path: Path to the pairs CSV.

    Returns:
        List of ``ActivityCliffPair``.
    """
    df = pd.read_csv(path)
    pairs = []
    for _, row in df.iterrows():
        pairs.append(ActivityCliffPair(
            mol_idx_a=int(row['mol_idx_a']),
            mol_idx_b=int(row['mol_idx_b']),
            tanimoto=float(row['tanimoto']),
            smiles_a=str(row.get('smiles_a', '')),
            smiles_b=str(row.get('smiles_b', '')),
        ))
    return pairs


def save_split_indices(split: MmpOodSplit, output_dir: str) -> None:
    """
    Save MMP-OOD split indices to CSV files.

    Creates:
        - ``<output_dir>/train.csv``  — one index per line
        - ``<output_dir>/valid.csv``
        - ``<output_dir>/test.csv``
        - ``<output_dir>/pairs.csv``
        - ``<output_dir>/stats.json``

    Args:
        split: ``MmpOodSplit`` to save.
        output_dir: Directory to write files into.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)

    for name, idx in [('train', split.train_idx),
                      ('valid', split.valid_idx),
                      ('test', split.test_idx)]:
        path = os.path.join(output_dir, f'{name}.csv')
        pd.DataFrame(idx, columns=['index']).to_csv(path, index=False)

    save_pairs_csv(split.pairs, os.path.join(output_dir, 'pairs.csv'))

    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(split.stats, f, indent=2)


def load_split_indices(split_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MMP-OOD split indices from CSV files.

    Args:
        split_dir: Directory containing train.csv, valid.csv, test.csv.

    Returns:
        Tuple of (train_idx, valid_idx, test_idx) as numpy arrays.
    """
    train = np.asarray(pd.read_csv(os.path.join(split_dir, 'train.csv'))['index'].values)
    valid = np.asarray(pd.read_csv(os.path.join(split_dir, 'valid.csv'))['index'].values)
    test = np.asarray(pd.read_csv(os.path.join(split_dir, 'test.csv'))['index'].values)
    return train, valid, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_cliff_accuracy_hard(
    pairs: List[ActivityCliffPair],
    predictions: Dict[int, int],
) -> Tuple[float, List[bool]]:
    """
    Compute pairwise cliff accuracy from **hard labels** (0/1).

    A pair is "correct" if the active molecule is predicted as 1 AND the
    inactive molecule is predicted as 0.

    Args:
        pairs: Activity-cliff pairs (mol_idx_a = active, mol_idx_b = inactive).
        predictions: Mapping from molecule index → predicted label (0 or 1).

    Returns:
        Tuple of (cliff_accuracy, per_pair_results).
    """
    results: List[bool] = []
    for p in pairs:
        pred_a = predictions.get(p.mol_idx_a)
        pred_b = predictions.get(p.mol_idx_b)
        if pred_a is None or pred_b is None:
            continue  # skip pairs where a molecule has no prediction
        correct = (pred_a == 1 and pred_b == 0)
        results.append(correct)

    if not results:
        return 0.0, results

    acc = sum(results) / len(results)
    return acc, results


def compute_cliff_accuracy_prob(
    pairs: List[ActivityCliffPair],
    probabilities: Dict[int, float],
) -> Tuple[float, List[bool]]:
    """
    Compute pairwise cliff accuracy from **predicted probabilities**.

    A pair is "correct" if the active molecule has a higher predicted
    probability (of being active) than the inactive molecule:
        P(active | mol_a) > P(active | mol_b).

    Args:
        pairs: Activity-cliff pairs (mol_idx_a = active, mol_idx_b = inactive).
        probabilities: Mapping from molecule index → P(active).

    Returns:
        Tuple of (cliff_accuracy, per_pair_results).
    """
    results: List[bool] = []
    for p in pairs:
        prob_a = probabilities.get(p.mol_idx_a)
        prob_b = probabilities.get(p.mol_idx_b)
        if prob_a is None or prob_b is None:
            continue
        correct = (prob_a > prob_b)
        results.append(correct)

    if not results:
        return 0.0, results

    acc = sum(results) / len(results)
    return acc, results


def evaluate_mmp_ood(
    pairs: List[ActivityCliffPair],
    test_idx: np.ndarray,
    y_true: Dict[int, int],
    y_pred: Dict[int, int],
    y_prob: Optional[Dict[int, float]] = None,
) -> MmpOodMetrics:
    """
    Full MMP-OOD evaluation: Macro-F1 + pairwise cliff accuracy.

    Args:
        pairs: Activity-cliff pairs.
        test_idx: Array of all test molecule indices.
        y_true: Ground-truth labels ``{mol_idx: label}``.
        y_pred: Predicted labels ``{mol_idx: label}``.
        y_prob: (Optional) Predicted P(active) ``{mol_idx: prob}``.

    Returns:
        ``MmpOodMetrics`` with all results.
    """
    # --- Macro-F1 on the full test set ---
    true_labels = []
    pred_labels = []
    for idx in test_idx:
        if idx in y_true and idx in y_pred:
            true_labels.append(y_true[idx])
            pred_labels.append(y_pred[idx])

    if true_labels:
        macro_f1 = float(f1_score(true_labels, pred_labels, average='macro'))
    else:
        macro_f1 = 0.0

    # --- Pairwise cliff accuracy ---
    if y_prob is not None:
        cliff_acc, per_pair = compute_cliff_accuracy_prob(pairs, y_prob)
    else:
        cliff_acc, per_pair = compute_cliff_accuracy_hard(pairs, y_pred)

    return MmpOodMetrics(
        macro_f1=macro_f1,
        cliff_accuracy=cliff_acc,
        num_pairs_evaluated=len(per_pair),
        num_test_molecules=len(test_idx),
        per_pair_results=per_pair,
    )
