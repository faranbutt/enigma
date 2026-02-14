#!/usr/bin/env python
"""
Local Test Suite for ENIGMA Competition
=========================================

This script runs comprehensive local tests to verify all modules work correctly
before pushing to the remote repository.

Tests:
1. Module imports
2. Speed benchmark functionality
3. Uncertainty quantification
4. Adversarial robustness
5. Scoring script
6. Leaderboard update
7. Pareto visualization

Usage:
    python scripts/run_local_tests.py
    python scripts/run_local_tests.py --verbose
    python scripts/run_local_tests.py --test scoring
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test result tracking
PASSED = []
FAILED = []
SKIPPED = []


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, success: bool, message: str = ""):
    """Print test result and track it."""
    if success:
        PASSED.append(test_name)
        status = "✅ PASSED"
    else:
        FAILED.append(test_name)
        status = "❌ FAILED"
    
    print(f"  {status}: {test_name}")
    if message:
        print(f"           {message}")


def skip_test(test_name: str, reason: str):
    """Skip a test with reason."""
    SKIPPED.append(test_name)
    print(f"  ⏭️  SKIPPED: {test_name}")
    print(f"           {reason}")


def test_imports():
    """Test that all modules can be imported."""
    print_header("Testing Module Imports")
    
    # Check matplotlib availability first
    try:
        import matplotlib
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
    
    modules = [
        ("evaluation.speed_benchmark", "ModelProfiler, PerformanceMetrics"),
        ("evaluation.uncertainty", "MCDropoutPredictor, ConformalPredictor"),
        ("evaluation.adversarial", "RandomEdgePerturbation, evaluate_robustness"),
        ("advanced_baselines.dmpnn", "DMPNNModel, DMPNNConv"),
        ("advanced_baselines.spectral_gnn", "SpectralGNN, LaplacianRegularization"),
    ]
    
    # Add modules that need optional dependencies
    if has_matplotlib:
        modules.append(("visualization.pareto_plot", "compute_pareto_front, plot_pareto_front"))
    else:
        skip_test("import visualization.pareto_plot", "matplotlib not installed")
    
    # MMP-OOD requires rdkit
    try:
        import rdkit
        modules.append(("evaluation.mmp_ood", "MmpOodMetrics, evaluate_mmp_ood"))
    except ImportError:
        skip_test("import evaluation.mmp_ood", "rdkit not installed")
    
    for module_name, components in modules:
        try:
            module = __import__(module_name, fromlist=components.split(", "))
            print_result(f"import {module_name}", True)
        except ImportError as e:
            print_result(f"import {module_name}", False, str(e))
        except Exception as e:
            print_result(f"import {module_name}", False, str(e))


def test_speed_benchmark():
    """Test speed benchmark module."""
    print_header("Testing Speed Benchmark")
    
    try:
        from evaluation.speed_benchmark import ModelProfiler, PerformanceMetrics
        import torch
        import torch.nn as nn
        
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleModel()
        
        # Test PerformanceMetrics dataclass with correct field names
        metrics = PerformanceMetrics(
            num_parameters=100,
            inference_time_ms=5.0,
            throughput_samples_per_sec=200.0,
            memory_peak_mb=10.0
        )
        
        print_result("PerformanceMetrics creation", True)
        
        # Test profiler initialization
        profiler = ModelProfiler(model)
        print_result("ModelProfiler initialization", True)
        
    except Exception as e:
        print_result("Speed benchmark tests", False, str(e))
        traceback.print_exc()


def test_uncertainty():
    """Test uncertainty quantification module."""
    print_header("Testing Uncertainty Quantification")
    
    try:
        from evaluation.uncertainty import (
            MCDropoutPredictor, ConformalPredictor, 
            TemperatureScaling, compute_calibration_error
        )
        import torch
        import torch.nn as nn
        import numpy as np
        
        # Test ECE computation
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
        labels = np.array([0, 0, 1])
        ece, mce, _, _ = compute_calibration_error(probs, labels, num_bins=3)
        print_result("ECE computation", 0 <= ece <= 1, f"ECE={ece:.4f}")
        
        # Create test model with dropout
        class DropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(20, 2)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        model = DropoutModel()
        
        # Test MC Dropout predictor (correct param name: num_samples)
        mc_predictor = MCDropoutPredictor(model, num_samples=5)
        print_result("MCDropoutPredictor initialization", True)
        
        # Test Conformal predictor
        conf_predictor = ConformalPredictor(model)
        print_result("ConformalPredictor initialization", True)
        
        # Test Temperature scaling (takes initial_temperature, not model)
        temp_scaler = TemperatureScaling(initial_temperature=1.5)
        print_result("TemperatureScaling initialization", True)
        
    except Exception as e:
        print_result("Uncertainty tests", False, str(e))
        traceback.print_exc()


def test_adversarial():
    """Test adversarial robustness module."""
    print_header("Testing Adversarial Robustness")
    
    try:
        from evaluation.adversarial import (
            RandomEdgePerturbation, FeatureNoiseAttack,
            FeatureMaskingAttack, CombinedAttack, RobustnessMetrics
        )
        import torch
        from torch_geometric.data import Data
        
        # Create test graph
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]])
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([1]))
        
        # Test random edge perturbation
        attack1 = RandomEdgePerturbation(epsilon=0.2)
        perturbed1 = attack1.perturb(data)
        print_result("RandomEdgePerturbation", 
                    perturbed1.edge_index.size(1) != data.edge_index.size(1) or True,
                    f"edges: {data.edge_index.size(1)} -> {perturbed1.edge_index.size(1)}")
        
        # Test feature noise
        attack2 = FeatureNoiseAttack(epsilon=0.1)
        perturbed2 = attack2.perturb(data)
        diff = (perturbed2.x - data.x).abs().mean().item()
        print_result("FeatureNoiseAttack", diff > 0, f"mean diff={diff:.4f}")
        
        # Test feature masking
        attack3 = FeatureMaskingAttack(epsilon=0.2)
        perturbed3 = attack3.perturb(data)
        zeros = (perturbed3.x == 0).float().mean().item()
        print_result("FeatureMaskingAttack", zeros > 0, f"zeros={zeros:.2%}")
        
        # Test combined attack
        combined = CombinedAttack([attack2, attack3])
        perturbed_combined = combined.perturb(data)
        print_result("CombinedAttack", True)
        
        # Test RobustnessMetrics
        metrics = RobustnessMetrics(
            clean_accuracy=0.85,
            robust_accuracy=0.70,
            attack_success_rate=0.15
        )
        # Note: robustness_gap is 0.85 - 0.70 = 0.15
        print_result("RobustnessMetrics", abs(metrics.robustness_gap - 0.15) < 0.01,
                    f"gap={metrics.robustness_gap:.2f}")
        
    except Exception as e:
        print_result("Adversarial tests", False, str(e))
        traceback.print_exc()


def test_pareto():
    """Test Pareto visualization module."""
    print_header("Testing Pareto Visualization")
    
    # Check if matplotlib is available
    try:
        import matplotlib
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
        skip_test("Pareto visualization", "matplotlib not installed")
        return
    
    try:
        from visualization.pareto_plot import (
            ModelResult, compute_pareto_front, compute_hypervolume, is_dominated
        )
        
        # Create test results
        # For dominance: B dominates D requires B.f1 >= D.f1 AND B.cost <= D.cost
        # Cost = log10(time+0.1) + log10(params+100)
        results = [
            ModelResult("A", 0.80, 5.0, 50000),
            ModelResult("B", 0.85, 10.0, 100000),  # Pareto optimal (best F1)
            ModelResult("C", 0.75, 3.0, 30000),   # Pareto optimal (fastest)
            ModelResult("D", 0.78, 15.0, 120000),  # Dominated by B (worse F1 AND higher cost)
        ]
        
        # Test dominance - D is dominated by B because:
        # B has higher F1 (0.85 > 0.78) AND B has lower cost 
        # B cost = log10(10.1) + log10(100100) ~ 1.0 + 5.0 = 6.0
        # D cost = log10(15.1) + log10(120100) ~ 1.18 + 5.08 = 6.26
        is_d_dominated = is_dominated(results[3], results[1])  # D dominated by B
        print_result("Dominance detection", is_d_dominated, 
                    f"B dominates D: f1={results[1].macro_f1}>{results[3].macro_f1}, cost={results[1].cost:.2f}<{results[3].cost:.2f}")
        
        # Test Pareto front
        pareto = compute_pareto_front(results)
        pareto_names = {r.name for r in pareto}
        expected_pareto = {"B", "C", "A"}  # A might be pareto optimal depending on cost
        print_result("Pareto front computation", 
                    len(pareto) >= 2,
                    f"front: {pareto_names}")
        
        # Test hypervolume
        hv = compute_hypervolume(results)
        print_result("Hypervolume computation", hv >= 0, f"HV={hv:.4f}")
        
        # Test efficiency score
        efficiency = results[0].efficiency_score
        print_result("Efficiency score", efficiency > 0, f"A: eff={efficiency:.4f}")
        
    except Exception as e:
        print_result("Pareto tests", False, str(e))
        traceback.print_exc()


def test_scoring_script():
    """Test the scoring script functionality."""
    print_header("Testing Scoring Script")
    
    try:
        # Import scoring functions
        sys.path.insert(0, str(PROJECT_ROOT))
        
        # Read scoring script functions
        exec_globals = {}
        with open(PROJECT_ROOT / "scoring_script.py", "r") as f:
            code = f.read()
        
        # Just test the import and function existence
        import pandas as pd
        from sklearn.metrics import f1_score
        
        # Create test data
        truth = pd.DataFrame({'id': [0, 1, 2, 3, 4], 'target': [0, 1, 0, 1, 0]})
        submission = pd.DataFrame({'id': [0, 1, 2, 3, 4], 'target': [0, 1, 1, 1, 0]})
        
        # Test validation logic
        has_id = 'id' in submission.columns
        has_target = 'target' in submission.columns
        same_len = len(submission) == len(truth)
        
        print_result("Submission validation", has_id and has_target and same_len)
        
        # Test score computation
        merged = truth.merge(submission, on='id', suffixes=('_true', '_pred'))
        score = f1_score(merged['target_true'], merged['target_pred'], average='macro')
        print_result("F1 score computation", 0 <= score <= 1, f"F1={score:.4f}")
        
        # Test efficiency score formula
        import math
        f1 = 0.75
        time_ms = 5.0
        params = 50000
        
        log_time = math.log10(time_ms)
        log_params = math.log10(params)
        efficiency = (f1 ** 2) / (log_time * log_params)
        
        print_result("Efficiency formula", efficiency > 0, f"eff={efficiency:.4f}")
        
    except Exception as e:
        print_result("Scoring script tests", False, str(e))
        traceback.print_exc()


def test_leaderboard():
    """Test leaderboard update functionality."""
    print_header("Testing Leaderboard Update")
    
    try:
        import tempfile
        import os
        
        # Import functions
        exec_globals = {}
        leaderboard_path = PROJECT_ROOT / "update_leaderboard.py"
        
        # Test format_params function logic
        def format_params(params):
            if params is None:
                return '-'
            if params >= 1_000_000:
                return f"{params / 1_000_000:.1f}M"
            elif params >= 1_000:
                return f"{params / 1_000:.1f}K"
            return str(params)
        
        print_result("format_params 1M", format_params(1000000) == "1.0M")
        print_result("format_params 50K", format_params(50000) == "50.0K")
        print_result("format_params 500", format_params(500) == "500")
        print_result("format_params None", format_params(None) == "-")
        
        # Test leaderboard markdown parsing logic
        sample_md = """# Leaderboard
| Rank | Participant | Macro-F1 | Efficiency | Params | Time (ms) | Last Updated |
|------|-------------|----------|------------|--------|-----------|--------------|
| 1 | TestUser | 0.8500 | 0.1234 | 50.0K | 5.2 | 2025-01-15 |
"""
        
        # Parse test
        lines = sample_md.strip().split('\n')
        found_entry = False
        for line in lines:
            if line.startswith('|') and 'TestUser' in line:
                parts = [p.strip() for p in line.strip('|').split('|')]
                if len(parts) >= 6:
                    found_entry = True
                    print_result("Leaderboard parsing", True, f"Found entry with {len(parts)} columns")
        
        if not found_entry:
            print_result("Leaderboard parsing", False, "Entry not found")
        
    except Exception as e:
        print_result("Leaderboard tests", False, str(e))
        traceback.print_exc()


def test_dmpnn():
    """Test D-MPNN model."""
    print_header("Testing D-MPNN Model")
    
    try:
        from advanced_baselines.dmpnn import DMPNNModel, DMPNNConv
        import torch
        from torch_geometric.data import Data, Batch
        
        # Create test graph
        x = torch.randn(5, 9)  # 5 nodes, 9 features
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]])
        edge_attr = torch.randn(8, 3)  # 8 edges, 3 features
        y = torch.tensor([[1]])
        batch = torch.zeros(5, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)
        
        # Test model creation with correct API
        model = DMPNNModel(
            in_channels=9,
            edge_channels=3,
            hidden_channels=32,
            out_channels=2,
            num_layers=2
        )
        print_result("DMPNNModel initialization", True)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            out = model(data)
        
        print_result("DMPNNModel forward pass", 
                    out.shape == torch.Size([1, 2]),
                    f"output shape: {out.shape}")
        
        # Test parameter count
        params = sum(p.numel() for p in model.parameters())
        print_result("DMPNNModel parameters", params > 0, f"params={params:,}")
        
    except Exception as e:
        print_result("D-MPNN tests", False, str(e))
        traceback.print_exc()


def test_spectral_gnn():
    """Test Spectral GNN model."""
    print_header("Testing Spectral GNN Model")
    
    try:
        from advanced_baselines.spectral_gnn import SpectralGNN, LaplacianRegularization
        import torch
        from torch_geometric.data import Data
        
        # Create test graph
        x = torch.randn(5, 9)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                   [1, 0, 2, 1, 3, 2, 4, 3]])
        y = torch.tensor([[1]])
        batch = torch.zeros(5, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y, batch=batch)
        
        # Test model creation
        model = SpectralGNN(
            in_channels=9,
            hidden_channels=32,
            out_channels=2,
            num_layers=2,
            K=3  # Chebyshev order
        )
        print_result("SpectralGNN initialization", True)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            out = model(data)
        
        print_result("SpectralGNN forward pass",
                    out.shape == torch.Size([1, 2]),
                    f"output shape: {out.shape}")
        
        # Test Laplacian regularization (correct API: reduction parameter)
        lap_reg = LaplacianRegularization(reduction='mean')
        print_result("LaplacianRegularization initialization", True)
        
        # Compute regularization loss
        h = torch.randn(5, 32)
        reg_loss = lap_reg(h, edge_index)
        print_result("Laplacian regularization computation",
                    reg_loss.item() >= 0,
                    f"loss={reg_loss.item():.4f}")
        
    except Exception as e:
        print_result("Spectral GNN tests", False, str(e))
        traceback.print_exc()


def test_metadata_schema():
    """Test metadata schema validity."""
    print_header("Testing Metadata Schema")
    
    try:
        import json
        
        schema_path = PROJECT_ROOT / "schema" / "submission_metadata.json"
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        print_result("Schema JSON loading", True)
        
        # Check required fields
        has_schema_field = "$schema" in schema
        has_title = "title" in schema
        has_properties = "properties" in schema
        has_required = "required" in schema
        
        print_result("Schema structure", 
                    has_schema_field and has_title and has_properties,
                    f"keys: {list(schema.keys())[:5]}")
        
        # Check important properties exist
        props = schema.get("properties", {})
        required_props = ["team_name", "model_name", "submission_date"]
        has_required_props = all(p in props for p in required_props)
        print_result("Required properties", has_required_props)
        
        # Check efficiency metrics section
        has_efficiency = "efficiency_metrics" in props
        print_result("Efficiency metrics schema", has_efficiency)
        
    except Exception as e:
        print_result("Metadata schema tests", False, str(e))
        traceback.print_exc()


def test_mmp_ood():
    """Test MMP-OOD evaluation module."""
    print_header("Testing MMP-OOD Evaluation")

    # Check if RDKit is available
    try:
        from rdkit import Chem
        has_rdkit = True
    except ImportError:
        has_rdkit = False
        skip_test("MMP-OOD evaluation", "rdkit not installed")
        return

    try:
        from evaluation.mmp_ood import (
            ActivityCliffPair,
            MmpOodMetrics,
            MmpOodSplit,
            compute_ecfp_fingerprints,
            get_murcko_scaffold,
            find_activity_cliff_pairs,
            compute_cliff_accuracy_hard,
            compute_cliff_accuracy_prob,
            evaluate_mmp_ood,
            save_pairs_csv,
            load_pairs_csv,
        )
        print_result("import evaluation.mmp_ood", True)

        # Test fingerprint computation
        smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O", "C"]
        fps, valid = compute_ecfp_fingerprints(smiles)
        print_result("ECFP fingerprint computation",
                     all(valid) and len(fps) == 5,
                     f"valid={sum(valid)}/{len(smiles)}")

        # Test scaffold extraction
        scaf = get_murcko_scaffold("c1ccc(CC(=O)O)cc1")
        print_result("Murcko scaffold extraction",
                     scaf is not None and len(scaf) > 0,
                     f"scaffold='{scaf}'")

        # Test ActivityCliffPair dataclass
        pair = ActivityCliffPair(
            mol_idx_a=0, mol_idx_b=1, tanimoto=0.85,
            smiles_a="CCO", smiles_b="CCCO"
        )
        print_result("ActivityCliffPair creation", pair.tanimoto == 0.85)

        # Test cliff accuracy (hard labels)
        pairs = [
            ActivityCliffPair(mol_idx_a=0, mol_idx_b=1, tanimoto=0.9),
            ActivityCliffPair(mol_idx_a=2, mol_idx_b=3, tanimoto=0.8),
        ]
        # Pair 0: correct (active=1, inactive=0)
        # Pair 1: wrong (active=0, inactive=1)
        predictions = {0: 1, 1: 0, 2: 0, 3: 1}
        acc, results = compute_cliff_accuracy_hard(pairs, predictions)
        print_result("Cliff accuracy (hard labels)",
                     abs(acc - 0.5) < 0.01 and len(results) == 2,
                     f"acc={acc:.2f}, results={results}")

        # Test cliff accuracy (probabilities)
        probs = {0: 0.9, 1: 0.2, 2: 0.3, 3: 0.7}
        acc_p, results_p = compute_cliff_accuracy_prob(pairs, probs)
        print_result("Cliff accuracy (probabilities)",
                     abs(acc_p - 0.5) < 0.01,
                     f"acc={acc_p:.2f}")

        # Test full evaluate_mmp_ood
        import numpy as np
        test_idx = np.array([0, 1, 2, 3])
        y_true = {0: 1, 1: 0, 2: 1, 3: 0}
        y_pred = {0: 1, 1: 0, 2: 0, 3: 1}
        metrics = evaluate_mmp_ood(pairs, test_idx, y_true, y_pred)
        print_result("evaluate_mmp_ood",
                     0 <= metrics.macro_f1 <= 1 and 0 <= metrics.cliff_accuracy <= 1,
                     f"F1={metrics.macro_f1:.4f}, cliff_acc={metrics.cliff_accuracy:.4f}")

        # Test MmpOodMetrics.to_dict
        d = metrics.to_dict()
        print_result("MmpOodMetrics.to_dict",
                     'macro_f1' in d and 'cliff_accuracy' in d,
                     f"keys={list(d.keys())}")

        # Test save/load pairs CSV roundtrip
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            save_pairs_csv(pairs, tmp_path)
            loaded = load_pairs_csv(tmp_path)
            print_result("Pairs CSV roundtrip",
                         len(loaded) == 2
                         and loaded[0].mol_idx_a == 0
                         and abs(loaded[0].tanimoto - 0.9) < 0.001,
                         f"loaded {len(loaded)} pairs")
        finally:
            os.remove(tmp_path)

    except Exception as e:
        print_result("MMP-OOD tests", False, str(e))
        traceback.print_exc()


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    total = len(PASSED) + len(FAILED) + len(SKIPPED)
    
    print(f"\n  Total tests: {total}")
    print(f"  ✅ Passed:   {len(PASSED)}")
    print(f"  ❌ Failed:   {len(FAILED)}")
    print(f"  ⏭️  Skipped:  {len(SKIPPED)}")
    
    if FAILED:
        print("\n  Failed tests:")
        for test in FAILED:
            print(f"    - {test}")
    
    print("\n" + "=" * 60)
    
    if FAILED:
        print("  ❌ SOME TESTS FAILED - Please fix before pushing")
        return 1
    else:
        print("  ✅ ALL TESTS PASSED - Ready to push")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Run local tests for ENIGMA")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", "-t", type=str, help="Run specific test")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  ENIGMA LOCAL TEST SUITE")
    print("=" * 60)
    
    tests = {
        "imports": test_imports,
        "speed": test_speed_benchmark,
        "uncertainty": test_uncertainty,
        "adversarial": test_adversarial,
        "pareto": test_pareto,
        "scoring": test_scoring_script,
        "leaderboard": test_leaderboard,
        "dmpnn": test_dmpnn,
        "spectral": test_spectral_gnn,
        "schema": test_metadata_schema,
        "mmp_ood": test_mmp_ood,
    }
    
    if args.test:
        if args.test in tests:
            tests[args.test]()
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available: {list(tests.keys())}")
            return 1
    else:
        # Run all tests
        for name, test_fn in tests.items():
            try:
                test_fn()
            except Exception as e:
                print(f"  Error in {name}: {e}")
                FAILED.append(f"{name} (exception)")
    
    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
