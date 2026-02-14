"""
Speed Benchmark Module for GNN Molecular Graph Classification Challenge
========================================================================

This module provides comprehensive performance profiling for GNN models,
measuring inference time, memory usage, parameter count, and FLOPs.

The efficiency score combines accuracy with computational cost:

    Efficiency = F1² / (log₁₀(Time_ms) * log₁₀(Params))

This rewards both high accuracy AND computational efficiency.

Usage:
    from evaluation.speed_benchmark import ModelProfiler
    
    profiler = ModelProfiler(model, device='cuda')
    metrics = profiler.profile(test_loader)
    
Author: ENIGMA Competition
License: MIT
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from contextlib import contextmanager
import gc
import json
import os


@dataclass
class PerformanceMetrics:
    """
    Container for comprehensive model performance metrics.
    
    Attributes:
        inference_time_ms: Average inference time per batch in milliseconds
        inference_time_per_sample_ms: Average inference time per molecule
        total_inference_time_ms: Total time for all predictions
        memory_peak_mb: Peak GPU/CPU memory usage in megabytes
        memory_allocated_mb: Memory allocated for model
        num_parameters: Total trainable parameters
        num_parameters_non_trainable: Non-trainable parameters
        flops_estimate: Estimated FLOPs per forward pass
        throughput_samples_per_sec: Samples processed per second
        efficiency_score: Combined efficiency metric
        relative_speed: Speed relative to baseline (1.0 = baseline)
    """
    inference_time_ms: float = 0.0
    inference_time_per_sample_ms: float = 0.0
    total_inference_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    num_parameters: int = 0
    num_parameters_non_trainable: int = 0
    flops_estimate: int = 0
    throughput_samples_per_sec: float = 0.0
    efficiency_score: float = 0.0
    relative_speed: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'inference_time_ms': round(self.inference_time_ms, 4),
            'inference_time_per_sample_ms': round(self.inference_time_per_sample_ms, 4),
            'total_inference_time_ms': round(self.total_inference_time_ms, 4),
            'memory_peak_mb': round(self.memory_peak_mb, 2),
            'memory_allocated_mb': round(self.memory_allocated_mb, 2),
            'num_parameters': self.num_parameters,
            'num_parameters_non_trainable': self.num_parameters_non_trainable,
            'flops_estimate': self.flops_estimate,
            'throughput_samples_per_sec': round(self.throughput_samples_per_sec, 2),
            'efficiency_score': round(self.efficiency_score, 4),
            'relative_speed': round(self.relative_speed, 2)
        }
    
    def format_speed(self) -> str:
        """Format relative speed for display."""
        if self.relative_speed == 1.0:
            return "1.00x (baseline)"
        elif self.relative_speed < 1.0:
            return f"{self.relative_speed:.2f}x faster"
        else:
            return f"{self.relative_speed:.2f}x slower"


class ModelProfiler:
    """
    Comprehensive model profiler for GNN architectures.
    
    Measures:
    - Inference latency (with warmup)
    - Memory consumption
    - Parameter count
    - FLOPs estimation
    - Throughput
    
    Example:
        >>> model = GCNModel(9, 64, 2)
        >>> profiler = ModelProfiler(model, device='cuda')
        >>> metrics = profiler.profile(test_loader, num_runs=10)
        >>> print(f"Time per sample: {metrics.inference_time_per_sample_ms:.2f}ms")
    """
    
    # Baseline times for relative speed calculation (in ms per sample)
    BASELINE_TIMES = {
        'gcn': 0.5,      # GCN baseline
        'graphsage': 0.6, # GraphSAGE baseline
        'gin': 0.7       # GIN baseline
    }
    DEFAULT_BASELINE = 0.5  # Default baseline time
    
    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = 'cpu',
        baseline_name: str = 'gcn'
    ):
        """
        Initialize the profiler.
        
        Args:
            model: PyTorch model to profile
            device: Device to run profiling on
            baseline_name: Name of baseline model for relative speed
        """
        self.model = model
        self.device = torch.device(device)
        self.baseline_time = self.BASELINE_TIMES.get(
            baseline_name.lower(), 
            self.DEFAULT_BASELINE
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @contextmanager
    def _timer(self):
        """Context manager for precise timing."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))[2]
        else:
            start = time.perf_counter()
            yield lambda: (time.perf_counter() - start) * 1000  # Convert to ms
    
    def _count_parameters(self) -> Tuple[int, int]:
        """
        Count model parameters.
        
        Returns:
            Tuple of (trainable_params, non_trainable_params)
        """
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        return trainable, non_trainable
    
    def _estimate_flops(self, sample_batch) -> int:
        """
        Estimate FLOPs for a forward pass.
        
        This is an approximation based on:
        - Linear layers: 2 * input_dim * output_dim
        - Graph convolutions: ~2 * edges * hidden_dim
        
        Args:
            sample_batch: Sample batch for shape inference
            
        Returns:
            Estimated FLOPs
        """
        flops = 0
        
        # Count parameters as rough FLOP estimate
        # Each parameter roughly corresponds to 2 FLOPs (multiply + add)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                flops += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.BatchNorm1d):
                flops += 2 * module.num_features
        
        # Account for graph structure
        if hasattr(sample_batch, 'edge_index'):
            num_edges = sample_batch.edge_index.size(1)
            # Rough estimate: each edge involves hidden_dim operations
            hidden_dim = 64  # Default estimate
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    hidden_dim = module.out_features
                    break
            flops += num_edges * hidden_dim * 2
        
        return flops
    
    def _measure_memory(self) -> Tuple[float, float]:
        """
        Measure memory usage.
        
        Returns:
            Tuple of (peak_memory_mb, allocated_memory_mb)
        """
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return peak, allocated
        else:
            # CPU memory estimation using model size
            param_memory = sum(
                p.numel() * p.element_size() 
                for p in self.model.parameters()
            ) / (1024 ** 2)
            return param_memory, param_memory
    
    @torch.no_grad()
    def profile(
        self,
        dataloader: DataLoader,
        num_warmup: int = 5,
        num_runs: int = 10,
        f1_score: Optional[float] = None
    ) -> PerformanceMetrics:
        """
        Profile the model on a dataset.
        
        Args:
            dataloader: DataLoader with test data
            num_warmup: Number of warmup iterations
            num_runs: Number of profiling runs to average
            f1_score: Optional F1 score for efficiency calculation
            
        Returns:
            PerformanceMetrics with all measurements
        """
        self.model.eval()
        
        # Get sample batch for FLOPs estimation
        sample_batch = next(iter(dataloader)).to(self.device)
        
        # Count parameters
        trainable_params, non_trainable_params = self._count_parameters()
        
        # Estimate FLOPs
        flops = self._estimate_flops(sample_batch)
        
        # Warmup runs
        for _ in range(num_warmup):
            for batch in dataloader:
                batch = batch.to(self.device)
                _ = self.model(batch)
        
        # Clear memory stats before profiling
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        total_samples = 0
        
        for _ in range(num_runs):
            run_time = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)
                
                with self._timer() as get_time:
                    _ = self.model(batch)
                
                run_time += get_time()
                if _ == 0:  # Count samples only once
                    total_samples += batch.num_graphs
            
            times.append(run_time)
            total_samples = total_samples // num_runs * num_runs  # Normalize
        
        # Calculate timing metrics
        total_samples = sum(1 for batch in dataloader for _ in range(batch.num_graphs))
        avg_time = np.mean(times)
        time_per_sample = avg_time / total_samples if total_samples > 0 else 0
        
        # Measure memory
        peak_memory, allocated_memory = self._measure_memory()
        
        # Calculate throughput
        throughput = (total_samples / avg_time * 1000) if avg_time > 0 else 0
        
        # Calculate efficiency score
        efficiency = 0.0
        if f1_score is not None and time_per_sample > 0 and trainable_params > 0:
            # Efficiency = F1² / (log₁₀(Time_ms) * log₁₀(Params))
            log_time = np.log10(max(time_per_sample, 0.01))
            log_params = np.log10(max(trainable_params, 1))
            efficiency = (f1_score ** 2) / (log_time * log_params) if log_time * log_params > 0 else 0
        
        # Calculate relative speed
        relative_speed = time_per_sample / self.baseline_time if self.baseline_time > 0 else 1.0
        
        return PerformanceMetrics(
            inference_time_ms=avg_time / len(dataloader) if len(dataloader) > 0 else 0,
            inference_time_per_sample_ms=time_per_sample,
            total_inference_time_ms=avg_time,
            memory_peak_mb=peak_memory,
            memory_allocated_mb=allocated_memory,
            num_parameters=trainable_params,
            num_parameters_non_trainable=non_trainable_params,
            flops_estimate=flops,
            throughput_samples_per_sec=throughput,
            efficiency_score=efficiency,
            relative_speed=relative_speed
        )


def benchmark_submission(
    model: nn.Module,
    test_loader: DataLoader,
    f1_score: float,
    device: str = 'cpu',
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Benchmark a submission and optionally save results.
    
    Args:
        model: Trained model to benchmark
        test_loader: Test data loader
        f1_score: Achieved F1 score
        device: Device to run on
        output_path: Optional path to save JSON results
        
    Returns:
        Dictionary with performance metrics
    """
    profiler = ModelProfiler(model, device=device)
    metrics = profiler.profile(test_loader, f1_score=f1_score)
    
    results = {
        'f1_score': f1_score,
        'performance': metrics.to_dict(),
        'relative_speed_formatted': metrics.format_speed()
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def format_parameters(num_params: int) -> str:
    """
    Format parameter count for human readability.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string (e.g., "1.2M", "125K")
    """
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.0f}K"
    else:
        return str(num_params)


if __name__ == "__main__":
    # Example usage
    print("Speed Benchmark Module")
    print("=" * 50)
    print("\nUsage:")
    print("  from evaluation.speed_benchmark import ModelProfiler")
    print("  profiler = ModelProfiler(model, device='cuda')")
    print("  metrics = profiler.profile(test_loader)")
    print("\nMetrics available:")
    print("  - inference_time_ms: Time per batch")
    print("  - inference_time_per_sample_ms: Time per molecule")
    print("  - memory_peak_mb: Peak memory usage")
    print("  - num_parameters: Trainable parameters")
    print("  - efficiency_score: Combined F1/speed metric")
    print("  - relative_speed: Speed vs baseline")
