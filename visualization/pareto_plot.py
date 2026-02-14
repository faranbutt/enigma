"""
Pareto Front Visualization for GNN Competition
==============================================

This module provides visualization tools for analyzing the trade-off
between model accuracy (Macro F1) and computational efficiency.

Mathematical Background:
-----------------------

A solution x is Pareto optimal (non-dominated) if there exists no
other solution y such that:
    - y is at least as good as x in all objectives
    - y is strictly better than x in at least one objective

For our bi-objective optimization (maximize F1, minimize cost):
    x dominates y iff:
        F1(x) ≥ F1(y) AND Cost(x) ≤ Cost(y)
        with at least one strict inequality

The Pareto front is the set of all non-dominated solutions.

Efficiency Metric:
    Efficiency = F1² / (log₁₀(time_ms) × log₁₀(params))

This creates a single scalar combining both objectives.

Hypervolume Indicator:
    HV(S) = volume of space dominated by solution set S
    
    Higher hypervolume indicates better overall Pareto front.

Usage:
    python visualization/pareto_plot.py

Author: ENIGMA Competition
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os


@dataclass
class ModelResult:
    """Container for model performance results."""
    name: str
    macro_f1: float
    inference_time_ms: float
    total_params: int
    is_baseline: bool = False
    
    @property
    def efficiency_score(self) -> float:
        """Compute efficiency score."""
        if self.macro_f1 <= 0:
            return 0.0
        time_ms = max(self.inference_time_ms, 0.1)
        params = max(self.total_params, 100)
        log_time = np.log10(time_ms)
        log_params = np.log10(params)
        denominator = log_time * log_params
        if denominator <= 0:
            denominator = max(log_params, 1.0)
        return (self.macro_f1 ** 2) / denominator
    
    @property
    def cost(self) -> float:
        """Compute computational cost (for Pareto analysis)."""
        return np.log10(self.inference_time_ms + 0.1) + np.log10(self.total_params + 100)


def is_dominated(p1: ModelResult, p2: ModelResult) -> bool:
    """
    Check if p1 is dominated by p2.
    
    p2 dominates p1 iff:
        - p2.f1 >= p1.f1 AND p2.cost <= p1.cost
        - At least one inequality is strict
    
    Args:
        p1: First model result
        p2: Second model result
        
    Returns:
        True if p1 is dominated by p2
    """
    f1_better = p2.macro_f1 >= p1.macro_f1
    cost_better = p2.cost <= p1.cost
    f1_strictly = p2.macro_f1 > p1.macro_f1
    cost_strictly = p2.cost < p1.cost
    
    return f1_better and cost_better and (f1_strictly or cost_strictly)


def compute_pareto_front(results: List[ModelResult]) -> List[ModelResult]:
    """
    Compute the Pareto front from a set of results.
    
    A solution is on the Pareto front if no other solution dominates it.
    
    Args:
        results: List of model results
        
    Returns:
        List of non-dominated results (Pareto front)
    """
    pareto_front = []
    
    for p1 in results:
        dominated = False
        for p2 in results:
            if p1 != p2 and is_dominated(p1, p2):
                dominated = True
                break
        if not dominated:
            pareto_front.append(p1)
    
    # Sort by F1 score for visualization
    pareto_front.sort(key=lambda x: x.macro_f1)
    
    return pareto_front


def compute_hypervolume(
    results: List[ModelResult],
    reference_point: Tuple[float, float] = (0.0, 10.0)
) -> float:
    """
    Compute the hypervolume indicator.
    
    Hypervolume = area dominated by the Pareto front above reference point.
    
    We use objectives:
        - Maximize F1 (x-axis)
        - Minimize cost (y-axis, lower is better)
    
    Args:
        results: List of model results
        reference_point: (min_f1, max_cost) reference point
        
    Returns:
        Hypervolume value
    """
    pareto = compute_pareto_front(results)
    
    if not pareto:
        return 0.0
    
    # Sort by F1 ascending
    pareto.sort(key=lambda x: x.macro_f1)
    
    # Compute hypervolume using simple algorithm
    ref_f1, ref_cost = reference_point
    hypervolume = 0.0
    
    prev_f1 = ref_f1
    prev_cost = ref_cost
    
    for model in pareto:
        f1 = model.macro_f1
        cost = model.cost
        
        # Only count if within reference bounds
        if f1 > prev_f1 and cost < ref_cost:
            # Add rectangle contribution
            width = f1 - prev_f1
            height = ref_cost - cost
            hypervolume += width * height
            
            prev_f1 = f1
    
    return hypervolume


def plot_pareto_front(
    results: List[ModelResult],
    save_path: Optional[str] = None,
    show_efficiency_contours: bool = True,
    title: str = "Accuracy vs. Efficiency Trade-off"
) -> plt.Figure:
    """
    Create a Pareto front visualization.
    
    Args:
        results: List of model results
        save_path: Path to save figure (optional)
        show_efficiency_contours: Whether to show efficiency iso-lines
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Extract data
    f1_scores = np.array([r.macro_f1 for r in results])
    costs = np.array([r.cost for r in results])
    names = [r.name for r in results]
    is_baseline = [r.is_baseline for r in results]
    efficiency = [r.efficiency_score for r in results]
    
    # Compute Pareto front
    pareto = compute_pareto_front(results)
    pareto_names = {r.name for r in pareto}
    
    # Color map based on efficiency
    norm_eff = (np.array(efficiency) - min(efficiency)) / (max(efficiency) - min(efficiency) + 1e-6)
    colors = plt.cm.viridis(norm_eff)
    
    # Plot efficiency contours
    if show_efficiency_contours:
        f1_range = np.linspace(0.3, 1.0, 100)
        cost_range = np.linspace(2, 12, 100)
        F1_grid, Cost_grid = np.meshgrid(f1_range, cost_range)
        
        # Efficiency = F1² / cost (simplified for visualization)
        Eff_grid = F1_grid ** 2 / np.maximum(Cost_grid * 0.5, 0.1)
        
        contour = ax.contourf(F1_grid, Cost_grid, Eff_grid, levels=20, 
                              cmap='Blues', alpha=0.3)
        cbar = plt.colorbar(contour, ax=ax, label='Efficiency (approx.)')
    
    # Plot non-Pareto points
    for i, result in enumerate(results):
        if result.name not in pareto_names:
            marker = 's' if is_baseline[i] else 'o'
            ax.scatter(f1_scores[i], costs[i], c=[colors[i]], 
                      s=100, marker=marker, alpha=0.6, 
                      edgecolors='gray', linewidths=1)
    
    # Plot Pareto front points (highlighted)
    pareto_f1 = [r.macro_f1 for r in pareto]
    pareto_cost = [r.cost for r in pareto]
    pareto_eff = [r.efficiency_score for r in pareto]
    pareto_colors = plt.cm.viridis(
        (np.array(pareto_eff) - min(efficiency)) / (max(efficiency) - min(efficiency) + 1e-6)
    )
    
    for i, result in enumerate(pareto):
        marker = 's' if result.is_baseline else '*'
        ax.scatter(pareto_f1[i], pareto_cost[i], c=[pareto_colors[i]], 
                  s=250 if marker == '*' else 150, marker=marker,
                  edgecolors='gold', linewidths=2, zorder=5)
    
    # Draw Pareto front line
    if len(pareto) > 1:
        pareto_sorted = sorted(pareto, key=lambda x: x.macro_f1)
        front_f1 = [r.macro_f1 for r in pareto_sorted]
        front_cost = [r.cost for r in pareto_sorted]
        ax.plot(front_f1, front_cost, 'g--', linewidth=2, alpha=0.7, 
               label='Pareto Front')
    
    # Add labels
    for i, result in enumerate(results):
        offset = (5, 5) if i % 2 == 0 else (5, -12)
        fontweight = 'bold' if result.name in pareto_names else 'normal'
        ax.annotate(result.name, (f1_scores[i], costs[i]), 
                   xytext=offset, textcoords='offset points',
                   fontsize=9, fontweight=fontweight, alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Macro F1 Score (higher is better)', fontsize=12)
    ax.set_ylabel('Computational Cost (lower is better)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='gold', linewidth=2, 
                      label='Pareto Optimal'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                  markersize=15, label='Participant Model'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                  markersize=10, label='Baseline Model'),
        plt.Line2D([0], [0], linestyle='--', color='green', linewidth=2,
                  label='Pareto Front')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Compute and display hypervolume
    hv = compute_hypervolume(results)
    ax.text(0.02, 0.02, f'Hypervolume: {hv:.4f}', transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_efficiency_comparison(
    results: List[ModelResult],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar chart comparing efficiency scores.
    
    Args:
        results: List of model results
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sort by efficiency
    sorted_results = sorted(results, key=lambda x: x.efficiency_score, reverse=True)
    names = [r.name for r in sorted_results]
    
    # Define colors
    colors = ['#2ecc71' if not r.is_baseline else '#3498db' for r in sorted_results]
    
    # Plot 1: Efficiency Score
    ax1 = axes[0]
    eff_scores = [r.efficiency_score for r in sorted_results]
    bars1 = ax1.barh(names, eff_scores, color=colors)
    ax1.set_xlabel('Efficiency Score')
    ax1.set_title('Overall Efficiency\n(F1² / log(time) × log(params))', fontsize=11)
    ax1.invert_yaxis()
    
    # Plot 2: Macro F1
    ax2 = axes[1]
    f1_scores = [r.macro_f1 for r in sorted_results]
    bars2 = ax2.barh(names, f1_scores, color=colors)
    ax2.set_xlabel('Macro F1 Score')
    ax2.set_title('Prediction Quality', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    
    # Plot 3: Parameters (log scale)
    ax3 = axes[2]
    params = [r.total_params for r in sorted_results]
    bars3 = ax3.barh(names, params, color=colors)
    ax3.set_xlabel('Parameters')
    ax3.set_xscale('log')
    ax3.set_title('Model Size', fontsize=11)
    ax3.invert_yaxis()
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='Participant'),
        mpatches.Patch(facecolor='#3498db', label='Baseline')
    ]
    fig.legend(handles=legend_elements, loc='upper right', ncol=2, 
              bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def load_results_from_leaderboard(leaderboard_path: str) -> List[ModelResult]:
    """
    Load model results from leaderboard.md file.
    
    Args:
        leaderboard_path: Path to leaderboard.md
        
    Returns:
        List of ModelResult objects
    """
    results = []
    
    with open(leaderboard_path, 'r') as f:
        lines = f.readlines()
    
    in_table = False
    for line in lines:
        line = line.strip()
        
        if line.startswith('| Rank'):
            in_table = True
            continue
        
        if line.startswith('|---'):
            continue
        
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= 6:
                try:
                    name = parts[1].strip().strip('*')
                    f1 = float(parts[2])
                    
                    # Parse params
                    params_str = parts[4].replace(',', '').replace('K', '000').replace('M', '000000')
                    params = int(float(params_str)) if params_str != '-' else 10000
                    
                    # Parse time
                    time_str = parts[5]
                    time_ms = float(time_str) if time_str != '-' else 10.0
                    
                    is_baseline = '*' in parts[1] or 'baseline' in name.lower()
                    
                    results.append(ModelResult(
                        name=name,
                        macro_f1=f1,
                        inference_time_ms=time_ms,
                        total_params=params,
                        is_baseline=is_baseline
                    ))
                except (ValueError, IndexError):
                    continue
    
    return results


# Example data for demonstration
DEMO_RESULTS = [
    ModelResult("GCN_baseline", 0.75, 5.2, 45000, is_baseline=True),
    ModelResult("GIN_baseline", 0.78, 6.1, 52000, is_baseline=True),
    ModelResult("GraphSAGE_baseline", 0.76, 4.8, 48000, is_baseline=True),
    ModelResult("DMPNN", 0.82, 8.5, 85000, is_baseline=True),
    ModelResult("SpectralGNN", 0.79, 12.3, 120000, is_baseline=True),
    ModelResult("Team_Alpha", 0.84, 15.0, 150000),
    ModelResult("Team_Beta", 0.81, 7.2, 60000),
    ModelResult("Team_Gamma", 0.77, 3.5, 35000),
    ModelResult("Team_Delta", 0.85, 25.0, 200000),
    ModelResult("Team_Epsilon", 0.80, 5.0, 55000),
]


if __name__ == "__main__":
    print("Pareto Front Visualization")
    print("=" * 50)
    
    # Use demo data
    results = DEMO_RESULTS
    
    print(f"\nAnalyzing {len(results)} models...")
    
    # Compute Pareto front
    pareto = compute_pareto_front(results)
    print(f"\nPareto optimal models ({len(pareto)}):")
    for model in sorted(pareto, key=lambda x: -x.macro_f1):
        print(f"  - {model.name}: F1={model.macro_f1:.3f}, "
              f"Eff={model.efficiency_score:.4f}")
    
    # Compute hypervolume
    hv = compute_hypervolume(results)
    print(f"\nHypervolume indicator: {hv:.4f}")
    
    # Create visualizations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\nGenerating Pareto front plot...")
    fig1 = plot_pareto_front(
        results,
        save_path=os.path.join(script_dir, 'pareto_front.png'),
        title="GNN Competition: Accuracy vs. Efficiency"
    )
    
    print("\nGenerating efficiency comparison...")
    fig2 = plot_efficiency_comparison(
        results,
        save_path=os.path.join(script_dir, 'efficiency_comparison.png')
    )
    
    print("\nVisualization complete!")
    plt.show()
