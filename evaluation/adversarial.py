"""
Adversarial Robustness Evaluation for Graph Neural Networks
============================================================

This module provides tools for evaluating the robustness of GNN models
against adversarial perturbations on molecular graphs.

Attack Types:
------------

1. Structure Perturbations (Topology Attacks):
   - Edge addition: Adding fake bonds
   - Edge deletion: Removing existing bonds
   - Edge rewiring: Redirect bonds between atoms
   
   Mathematical formulation:
   Find δ_A (perturbation to adjacency) such that:
       ||δ_A||_0 ≤ ε  (budget constraint)
       f(G + δ_A) ≠ f(G)  (misclassification)

2. Feature Perturbations:
   - Node feature noise: Gaussian/uniform noise on atom features
   - Feature masking: Zeroing out features
   - Feature shuffling: Permuting features across nodes
   
   Mathematical formulation:
   Find δ_X (perturbation to features) such that:
       ||δ_X||_p ≤ ε  (Lp norm constraint)
       f(X + δ_X, A) ≠ f(X, A)

3. Combined Attacks:
   - Joint structure + feature perturbations
   - Adaptive attacks based on gradient information

Robustness Metrics:
------------------

1. Robust Accuracy:
   Acc_robust = min_{||δ|| ≤ ε} Acc(X + δ)
   
   Accuracy under worst-case perturbation.

2. Certified Robustness Radius:
   r* = max{r : f(x + δ) = f(x) ∀ ||δ|| ≤ r}
   
   Largest perturbation radius with guaranteed correct prediction.

3. Attack Success Rate (ASR):
   ASR = |{x : f(x + δ) ≠ y}| / |{x : f(x) = y}|
   
   Fraction of correctly classified samples that are flipped.

4. Perturbation Sensitivity:
   S(x) = ||∇_x L(f(x), y)||
   
   Gradient magnitude w.r.t. input.

References:
----------
- Zügner et al., "Adversarial Attacks on Neural Networks for Graph Data" 
  (KDD 2018)
- Xu et al., "Topology Attack and Defense for Graph Neural Networks: 
  An Optimization Perspective" (IJCAI 2019)
- Jin et al., "Adversarial Attacks and Defenses on Graphs" (2020)

Author: ENIGMA Competition
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, remove_self_loops
import copy
import warnings


@dataclass
class RobustnessMetrics:
    """
    Container for adversarial robustness metrics.
    
    Attributes:
        clean_accuracy: Accuracy on unperturbed data
        robust_accuracy: Accuracy under attack
        attack_success_rate: Fraction of successful attacks
        avg_perturbation_size: Average size of perturbations
        certified_radius: Certified robustness radius (if computed)
        sensitivity: Average gradient sensitivity
    """
    clean_accuracy: float = 0.0
    robust_accuracy: float = 0.0
    attack_success_rate: float = 0.0
    avg_perturbation_size: float = 0.0
    certified_radius: float = 0.0
    sensitivity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'clean_accuracy': round(self.clean_accuracy, 4),
            'robust_accuracy': round(self.robust_accuracy, 4),
            'attack_success_rate': round(self.attack_success_rate, 4),
            'avg_perturbation_size': round(self.avg_perturbation_size, 4),
            'certified_radius': round(self.certified_radius, 4),
            'sensitivity': round(self.sensitivity, 4)
        }
    
    @property
    def robustness_gap(self) -> float:
        """Compute gap between clean and robust accuracy."""
        return self.clean_accuracy - self.robust_accuracy


class GraphPerturbation:
    """
    Base class for graph perturbation attacks.
    
    Subclasses implement specific attack strategies for
    molecular graph adversarial examples.
    """
    
    def __init__(self, epsilon: float = 0.1):
        """
        Args:
            epsilon: Perturbation budget (meaning depends on attack type)
        """
        self.epsilon = epsilon
    
    def perturb(self, data: Data) -> Data:
        """
        Apply perturbation to graph.
        
        Args:
            data: PyG Data object
            
        Returns:
            Perturbed Data object
        """
        raise NotImplementedError


class RandomEdgePerturbation(GraphPerturbation):
    """
    Random edge perturbation attack.
    
    Randomly adds or removes edges from the graph.
    This simulates random noise in the molecular structure.
    
    For molecular graphs, this represents:
    - Edge addition: Adding spurious bonds
    - Edge deletion: Missing bonds in the representation
    
    Args:
        epsilon: Fraction of edges to perturb
        mode: 'add', 'delete', or 'both'
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        mode: str = 'both'
    ):
        super().__init__(epsilon)
        self.mode = mode
    
    def perturb(self, data: Data) -> Data:
        """Apply random edge perturbation."""
        data = copy.deepcopy(data)
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_edges = edge_index.size(1)
        
        # Number of edges to perturb
        num_perturb = max(1, int(num_edges * self.epsilon))
        
        if self.mode in ['delete', 'both']:
            # Delete random edges
            num_delete = num_perturb // 2 if self.mode == 'both' else num_perturb
            if num_delete > 0 and num_edges > num_delete:
                keep_mask = torch.ones(num_edges, dtype=torch.bool)
                delete_idx = torch.randperm(num_edges)[:num_delete]
                keep_mask[delete_idx] = False
                edge_index = edge_index[:, keep_mask]
        
        if self.mode in ['add', 'both']:
            # Add random edges
            num_add = num_perturb // 2 if self.mode == 'both' else num_perturb
            if num_add > 0:
                # Generate random new edges
                new_src = torch.randint(0, num_nodes, (num_add,))
                new_dst = torch.randint(0, num_nodes, (num_add,))
                # Remove self-loops
                mask = new_src != new_dst
                new_edges = torch.stack([new_src[mask], new_dst[mask]], dim=0)
                # Add reverse edges for undirected graph
                new_edges = torch.cat([new_edges, new_edges.flip(0)], dim=1)
                edge_index = torch.cat([edge_index, new_edges.to(edge_index.device)], dim=1)
        
        data.edge_index = edge_index
        return data


class GradientEdgeAttack(GraphPerturbation):
    """
    Gradient-based edge perturbation attack.
    
    Uses gradient information to identify the most important edges
    and perturb them to maximize loss.
    
    Mathematical formulation:
    For each edge (i,j), compute importance score:
        s_{ij} = |∂L/∂A_{ij}|
    
    Then remove/add edges with highest scores.
    
    This is a simplified version of the Meta-Attack from Zügner et al.
    
    Args:
        model: Target model to attack
        epsilon: Number of edges to perturb
        criterion: Loss function
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: int = 5,
        criterion: Optional[nn.Module] = None
    ):
        super().__init__(epsilon)
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
    
    def _compute_edge_importance(
        self,
        data: Data,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute importance scores for edges using gradients.
        
        Args:
            data: Graph data
            device: Device to compute on
            
        Returns:
            Edge importance scores [num_edges]
        """
        self.model.train()  # Enable gradients
        
        # Make edge_index differentiable via adjacency
        data = data.to(device)
        
        # Forward pass
        out = self.model(data)
        
        # Compute loss
        target = data.y.squeeze().long()
        loss = self.criterion(out, target)
        
        # Backpropagate
        loss.backward()
        
        # Get gradients w.r.t. node features as proxy for edge importance
        if data.x.grad is not None:
            node_importance = data.x.grad.abs().sum(dim=-1)
            src, dst = data.edge_index
            edge_importance = node_importance[src] + node_importance[dst]
        else:
            # Uniform importance if no gradient available
            edge_importance = torch.ones(data.edge_index.size(1), device=device)
        
        return edge_importance
    
    def perturb(
        self,
        data: Data,
        device: torch.device = torch.device('cpu')
    ) -> Data:
        """
        Apply gradient-based edge attack.
        
        Args:
            data: Graph data
            device: Device to compute on
            
        Returns:
            Perturbed graph
        """
        data_copy = copy.deepcopy(data)
        data_copy.x.requires_grad_(True)
        
        importance = self._compute_edge_importance(data_copy, device)
        
        # Remove most important edges
        num_remove = min(int(self.epsilon), data_copy.edge_index.size(1) // 4)
        
        if num_remove > 0:
            _, top_indices = importance.topk(num_remove * 2)  # *2 for undirected
            mask = torch.ones(data_copy.edge_index.size(1), dtype=torch.bool)
            mask[top_indices[:num_remove]] = False
            
            perturbed_data = copy.deepcopy(data)
            perturbed_data.edge_index = data_copy.edge_index[:, mask]
            return perturbed_data
        
        return data


class FeatureNoiseAttack(GraphPerturbation):
    """
    Feature noise injection attack.
    
    Adds Gaussian or uniform noise to node features.
    
    Mathematical formulation:
        X' = X + ε * δ
    where δ ~ N(0, I) or δ ~ U(-1, 1)
    
    This simulates measurement noise or uncertainty in
    molecular feature extraction.
    
    Args:
        epsilon: Noise scale
        noise_type: 'gaussian' or 'uniform'
        relative: If True, scale by feature magnitude
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        noise_type: str = 'gaussian',
        relative: bool = True
    ):
        super().__init__(epsilon)
        self.noise_type = noise_type
        self.relative = relative
    
    def perturb(self, data: Data) -> Data:
        """Apply feature noise."""
        data = copy.deepcopy(data)
        x = data.x.float()
        
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(x)
        else:  # uniform
            noise = torch.rand_like(x) * 2 - 1
        
        if self.relative:
            # Scale noise by feature magnitude
            scale = x.abs().mean(dim=0, keepdim=True).clamp(min=1e-6)
            noise = noise * scale
        
        data.x = x + self.epsilon * noise
        return data


class FeatureMaskingAttack(GraphPerturbation):
    """
    Feature masking attack.
    
    Randomly zeros out node features, simulating missing
    or corrupted molecular attributes.
    
    Args:
        epsilon: Fraction of features to mask
        mask_type: 'random' or 'structured' (entire feature dimension)
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        mask_type: str = 'random'
    ):
        super().__init__(epsilon)
        self.mask_type = mask_type
    
    def perturb(self, data: Data) -> Data:
        """Apply feature masking."""
        data = copy.deepcopy(data)
        x = data.x.float()
        
        if self.mask_type == 'structured':
            # Mask entire feature dimensions
            num_mask = max(1, int(x.size(1) * self.epsilon))
            mask_dims = torch.randperm(x.size(1))[:num_mask]
            x[:, mask_dims] = 0
        else:
            # Random masking
            mask = torch.rand_like(x) < self.epsilon
            x[mask] = 0
        
        data.x = x
        return data


class CombinedAttack(GraphPerturbation):
    """
    Combined structure and feature attack.
    
    Applies multiple perturbation types simultaneously
    for more realistic robustness evaluation.
    
    Args:
        attacks: List of attack instances
    """
    
    def __init__(self, attacks: List[GraphPerturbation]):
        super().__init__(epsilon=0)
        self.attacks = attacks
    
    def perturb(self, data: Data) -> Data:
        """Apply all attacks sequentially."""
        for attack in self.attacks:
            data = attack.perturb(data)
        return data


def evaluate_robustness(
    model: nn.Module,
    loader: DataLoader,
    attacks: List[GraphPerturbation],
    device: torch.device,
    verbose: bool = False
) -> Dict[str, RobustnessMetrics]:
    """
    Evaluate model robustness against multiple attacks.
    
    Args:
        model: Model to evaluate
        loader: Test data loader
        attacks: List of attacks to evaluate
        device: Device to compute on
        verbose: Print progress
        
    Returns:
        Dictionary mapping attack names to robustness metrics
    """
    model.eval()
    results = {}
    
    # First, compute clean accuracy
    clean_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            clean_correct += (pred == batch.y.squeeze()).sum().item()
            total += batch.num_graphs
    
    clean_accuracy = clean_correct / total
    
    if verbose:
        print(f"Clean accuracy: {clean_accuracy:.4f}")
    
    # Evaluate each attack
    for attack in attacks:
        attack_name = attack.__class__.__name__
        
        robust_correct = 0
        attack_success = 0
        total_correctly_classified = 0
        perturbation_sizes = []
        
        for batch in loader:
            batch = batch.to(device)
            
            # Get clean predictions
            with torch.no_grad():
                clean_out = model(batch)
                clean_pred = clean_out.argmax(dim=1)
            
            # Apply attack to each graph in batch
            for i in range(batch.num_graphs):
                # Extract single graph
                start_idx = (batch.batch == i).nonzero().min().item()
                end_idx = (batch.batch == i).nonzero().max().item() + 1
                
                single_data = Data(
                    x=batch.x[start_idx:end_idx],
                    edge_index=batch.edge_index[:, (batch.edge_index[0] >= start_idx) & 
                                                  (batch.edge_index[0] < end_idx)] - start_idx,
                    y=batch.y[i:i+1]
                )
                single_data.batch = torch.zeros(end_idx - start_idx, dtype=torch.long)
                
                # Apply perturbation
                if hasattr(attack, 'model'):
                    perturbed = attack.perturb(single_data, device)
                else:
                    perturbed = attack.perturb(single_data)
                perturbed = perturbed.to(device)
                
                # Get perturbed prediction
                with torch.no_grad():
                    perturbed_out = model(perturbed)
                    perturbed_pred = perturbed_out.argmax(dim=1)
                
                true_label = batch.y[i].item()
                clean_label = clean_pred[i].item()
                perturbed_label = perturbed_pred.item()
                
                # Count robust correct
                if perturbed_label == true_label:
                    robust_correct += 1
                
                # Count attack success (only on correctly classified samples)
                if clean_label == true_label:
                    total_correctly_classified += 1
                    if perturbed_label != true_label:
                        attack_success += 1
                
                # Track perturbation size
                orig_edges = single_data.edge_index.size(1)
                pert_edges = perturbed.edge_index.size(1)
                perturbation_sizes.append(abs(orig_edges - pert_edges))
        
        robust_accuracy = robust_correct / total if total > 0 else 0
        asr = attack_success / total_correctly_classified if total_correctly_classified > 0 else 0
        avg_pert = np.mean(perturbation_sizes) if perturbation_sizes else 0
        
        results[attack_name] = RobustnessMetrics(
            clean_accuracy=clean_accuracy,
            robust_accuracy=robust_accuracy,
            attack_success_rate=asr,
            avg_perturbation_size=avg_pert
        )
        
        if verbose:
            print(f"{attack_name}: Robust acc = {robust_accuracy:.4f}, ASR = {asr:.4f}")
    
    return results


def compute_sensitivity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> float:
    """
    Compute average gradient sensitivity.
    
    Sensitivity = E[||∇_x L(f(x), y)||]
    
    Higher sensitivity indicates the model is more susceptible
    to small input perturbations.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to compute on
        
    Returns:
        Average sensitivity
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    sensitivities = []
    
    for batch in loader:
        batch = batch.to(device)
        batch.x.requires_grad_(True)
        
        out = model(batch)
        loss = criterion(out, batch.y.squeeze().long())
        
        loss.backward()
        
        if batch.x.grad is not None:
            grad_norm = batch.x.grad.norm(dim=-1).mean().item()
            sensitivities.append(grad_norm)
        
        batch.x.requires_grad_(False)
    
    model.eval()
    
    return np.mean(sensitivities) if sensitivities else 0.0


def quick_robustness_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float = 0.1
) -> RobustnessMetrics:
    """
    Quick robustness evaluation with default attacks.
    
    Args:
        model: Model to evaluate
        loader: Test data loader
        device: Device to compute on
        epsilon: Perturbation budget
        
    Returns:
        Aggregated robustness metrics
    """
    attacks = [
        RandomEdgePerturbation(epsilon=epsilon, mode='delete'),
        FeatureNoiseAttack(epsilon=epsilon),
        FeatureMaskingAttack(epsilon=epsilon)
    ]
    
    results = evaluate_robustness(model, loader, attacks, device, verbose=False)
    
    # Aggregate results
    avg_robust_acc = np.mean([r.robust_accuracy for r in results.values()])
    avg_asr = np.mean([r.attack_success_rate for r in results.values()])
    
    sensitivity = compute_sensitivity(model, loader, device)
    
    return RobustnessMetrics(
        clean_accuracy=list(results.values())[0].clean_accuracy,
        robust_accuracy=avg_robust_acc,
        attack_success_rate=avg_asr,
        sensitivity=sensitivity
    )


if __name__ == "__main__":
    print("Adversarial Robustness Evaluation")
    print("=" * 50)
    print("\nAttack types available:")
    print("  - RandomEdgePerturbation: Random edge add/delete")
    print("  - GradientEdgeAttack: Gradient-based edge attacks")
    print("  - FeatureNoiseAttack: Gaussian/uniform noise")
    print("  - FeatureMaskingAttack: Feature zeroing")
    print("  - CombinedAttack: Multiple attack combination")
    print("\nMetrics computed:")
    print("  - Clean accuracy")
    print("  - Robust accuracy")
    print("  - Attack Success Rate (ASR)")
    print("  - Gradient sensitivity")
