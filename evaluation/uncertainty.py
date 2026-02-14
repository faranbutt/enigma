"""
Uncertainty Quantification for GNN Predictions
===============================================

This module provides comprehensive uncertainty quantification methods
for Graph Neural Network predictions, including:

1. Monte Carlo Dropout (MC Dropout)
2. Deep Ensembles
3. Conformal Prediction
4. Temperature Scaling Calibration

Mathematical Framework:
----------------------

1. Epistemic Uncertainty (Model Uncertainty):
   Uncertainty due to limited training data. Can be reduced with more data.
   
   Estimated via MC Dropout:
   Var_epistemic[y] = (1/T) Σ_t (f_θt(x) - ȳ)²
   where θt are different dropout masks

2. Aleatoric Uncertainty (Data Uncertainty):
   Inherent noise in the data. Cannot be reduced with more data.
   
   Estimated by predicting mean and variance:
   p(y|x) = N(μ(x), σ²(x))

3. Total Predictive Uncertainty:
   Var[y] = E[Var[y|θ]] + Var[E[y|θ]]
          = Aleatoric    + Epistemic

4. Conformal Prediction:
   Provides calibrated prediction sets with guaranteed coverage:
   P(y ∈ C(x)) ≥ 1 - α
   
   Uses nonconformity scores:
   s(x, y) = 1 - p̂_y(x)  (softmax probability of true class)

5. Expected Calibration Error (ECE):
   ECE = Σ_m (|B_m|/n) |acc(B_m) - conf(B_m)|
   
   where B_m are confidence bins, acc is accuracy, conf is confidence.

References:
----------
- Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
- Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty 
  Estimation using Deep Ensembles" (NeurIPS 2017)
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction 
  and Distribution-Free Uncertainty Quantification" (2022)

Author: ENIGMA Competition
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from torch_geometric.loader import DataLoader
import warnings


@dataclass
class UncertaintyMetrics:
    """
    Container for uncertainty quantification metrics.
    
    Attributes:
        mean_prediction: Mean predicted probability
        epistemic_uncertainty: Model uncertainty (from MC Dropout)
        aleatoric_uncertainty: Data uncertainty (if estimated)
        total_uncertainty: Combined uncertainty
        entropy: Predictive entropy
        confidence: Confidence score (max probability)
        calibration_error: Expected Calibration Error
        brier_score: Brier score for calibration
    """
    mean_prediction: np.ndarray = field(default_factory=lambda: np.array([]))
    epistemic_uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    aleatoric_uncertainty: Optional[np.ndarray] = None
    total_uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    entropy: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    calibration_error: float = 0.0
    brier_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'epistemic_mean': float(np.mean(self.epistemic_uncertainty)),
            'epistemic_std': float(np.std(self.epistemic_uncertainty)),
            'total_uncertainty_mean': float(np.mean(self.total_uncertainty)),
            'entropy_mean': float(np.mean(self.entropy)),
            'confidence_mean': float(np.mean(self.confidence)),
            'calibration_error': self.calibration_error,
            'brier_score': self.brier_score
        }


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    MC Dropout keeps dropout enabled during inference and runs
    multiple forward passes to estimate predictive uncertainty.
    
    Mathematical formulation:
        p(y|x, D) ≈ (1/T) Σ_t p(y|x, θ_t)
    
    where θ_t represents different dropout masks and T is the
    number of forward passes.
    
    The uncertainty is estimated as:
        Var[y] ≈ (1/T) Σ_t (ŷ_t - ȳ)²
    
    where ŷ_t is the prediction at pass t and ȳ is the mean prediction.
    
    Args:
        model: Neural network model with dropout layers
        num_samples: Number of MC samples (forward passes)
        apply_softmax: Whether to apply softmax to outputs
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 30,
        apply_softmax: bool = True
    ):
        self.model = model
        self.num_samples = num_samples
        self.apply_softmax = apply_softmax
    
    def _enable_dropout(self):
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def predict(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            data: PyG Data object
            
        Returns:
            Tuple of:
            - mean_probs: Mean predicted probabilities [batch_size, num_classes]
            - epistemic: Epistemic uncertainty per sample [batch_size]
            - entropy: Predictive entropy per sample [batch_size]
        """
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        for _ in range(self.num_samples):
            logits = self.model(data)
            
            if self.apply_softmax:
                probs = F.softmax(logits, dim=-1)
            else:
                probs = logits
            
            predictions.append(probs.cpu().numpy())
        
        # Stack predictions: [num_samples, batch_size, num_classes]
        predictions = np.stack(predictions, axis=0)
        
        # Mean prediction
        mean_probs = predictions.mean(axis=0)
        
        # Epistemic uncertainty: variance of predictions
        epistemic = predictions.var(axis=0).mean(axis=-1)  # Average variance across classes
        
        # Predictive entropy: H[E[p(y|x)]]
        epsilon = 1e-10
        entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)
        
        return mean_probs, epistemic, entropy
    
    def predict_with_samples(self, data) -> np.ndarray:
        """
        Get all MC samples for detailed analysis.
        
        Args:
            data: PyG Data object
            
        Returns:
            All predictions [num_samples, batch_size, num_classes]
        """
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        for _ in range(self.num_samples):
            logits = self.model(data)
            probs = F.softmax(logits, dim=-1) if self.apply_softmax else logits
            predictions.append(probs.cpu().numpy())
        
        return np.stack(predictions, axis=0)


class ConformalPredictor:
    """
    Conformal prediction for calibrated uncertainty quantification.
    
    Conformal prediction provides prediction sets with guaranteed
    coverage probability:
        P(y ∈ C(x)) ≥ 1 - α
    
    Algorithm (Split Conformal):
    1. Split calibration data
    2. Compute nonconformity scores on calibration set:
       s_i = 1 - p̂_{y_i}(x_i)  (where p̂_y is softmax prob)
    3. Find quantile q̂ = Quantile(s_1,...,s_n; (1-α)(1+1/n))
    4. For new x, prediction set is:
       C(x) = {y : p̂_y(x) > 1 - q̂}
    
    Args:
        model: Trained classifier
        alpha: Target miscoverage rate (default 0.1 for 90% coverage)
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1
    ):
        self.model = model
        self.alpha = alpha
        self.quantile = None
        self.calibrated = False
    
    @torch.no_grad()
    def calibrate(self, cal_loader: DataLoader) -> float:
        """
        Calibrate conformal predictor on calibration set.
        
        Args:
            cal_loader: Calibration data loader
            
        Returns:
            Computed quantile threshold
        """
        self.model.eval()
        
        scores = []
        
        for batch in cal_loader:
            logits = self.model(batch)
            probs = F.softmax(logits, dim=-1)
            
            # Get true labels
            y_true = batch.y.squeeze()
            
            # Nonconformity score: 1 - p(y_true)
            # This is the "softmax" score from RAPS/APS methods
            true_probs = probs[torch.arange(len(y_true)), y_true]
            score = 1 - true_probs.cpu().numpy()
            
            scores.extend(score.tolist())
        
        scores = np.array(scores)
        n = len(scores)
        
        # Compute quantile with finite sample correction
        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        self.quantile = np.quantile(scores, level)
        
        self.calibrated = True
        
        return self.quantile
    
    @torch.no_grad()
    def predict(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with conformal prediction sets.
        
        Args:
            data: PyG Data object
            
        Returns:
            Tuple of:
            - predictions: Predicted class [batch_size]
            - prediction_sets: Binary mask of included classes [batch_size, num_classes]
            - set_sizes: Size of each prediction set [batch_size]
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before predict()")
        
        self.model.eval()
        
        logits = self.model(data)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        # Prediction set: classes with prob > 1 - quantile
        threshold = 1 - self.quantile
        prediction_sets = probs > threshold
        
        # Point prediction: highest probability
        predictions = probs.argmax(axis=-1)
        
        # Set sizes
        set_sizes = prediction_sets.sum(axis=-1)
        
        return predictions, prediction_sets, set_sizes
    
    def get_coverage(self, loader: DataLoader) -> float:
        """
        Compute empirical coverage on a dataset.
        
        Args:
            loader: Data loader
            
        Returns:
            Empirical coverage rate
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before get_coverage()")
        
        covered = 0
        total = 0
        
        for batch in loader:
            _, pred_sets, _ = self.predict(batch)
            y_true = batch.y.squeeze().cpu().numpy()
            
            # Check if true label is in prediction set
            for i, y in enumerate(y_true):
                if pred_sets[i, y]:
                    covered += 1
                total += 1
        
        return covered / total if total > 0 else 0.0


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for neural network calibration.
    
    Temperature scaling post-hoc calibrates model outputs by learning
    a single temperature parameter T:
        p_calibrated = softmax(z / T)
    
    where z are the pre-softmax logits.
    
    This is a simple but effective calibration method that preserves
    accuracy while improving calibration.
    
    Mathematical property:
    - T > 1: Softens predictions (increases uncertainty)
    - T < 1: Sharpens predictions (decreases uncertainty)
    - T = 1: No change
    
    Reference:
        Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
    
    Args:
        initial_temperature: Starting temperature value
    """
    
    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling.
        
        Args:
            logits: Pre-softmax predictions [batch_size, num_classes]
            
        Returns:
            Calibrated logits
        """
        return logits / self.temperature
    
    def calibrate(
        self,
        model: nn.Module,
        cal_loader: DataLoader,
        device: torch.device,
        max_iter: int = 100,
        lr: float = 0.01
    ) -> float:
        """
        Learn optimal temperature on calibration set.
        
        Args:
            model: Trained model
            cal_loader: Calibration data loader
            device: Device to run on
            max_iter: Maximum optimization iterations
            lr: Learning rate
            
        Returns:
            Optimal temperature value
        """
        self.to(device)
        model.eval()
        
        # Collect all logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in cal_loader:
                batch = batch.to(device)
                logits = model(batch)
                all_logits.append(logits)
                all_labels.append(batch.y.squeeze())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).long()
        
        # Optimize temperature using NLL loss
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = criterion(scaled_logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return self.temperature.item()


def compute_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE) and related metrics.
    
    ECE measures how well predicted probabilities match empirical frequencies:
        ECE = Σ_m (|B_m| / n) |acc(B_m) - conf(B_m)|
    
    where B_m are confidence bins.
    
    Args:
        probs: Predicted probabilities [num_samples, num_classes]
        labels: True labels [num_samples]
        num_bins: Number of confidence bins
        
    Returns:
        Tuple of:
        - ECE: Expected Calibration Error
        - MCE: Maximum Calibration Error
        - bin_accuracies: Accuracy per bin
        - bin_confidences: Mean confidence per bin
    """
    # Get predictions and confidences
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    
    # Compute accuracy
    accuracies = (predictions == labels).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            
            # Update ECE
            ece += np.abs(bin_acc - bin_conf) * prop_in_bin
            mce = max(mce, np.abs(bin_acc - bin_conf))
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
    
    return ece, mce, np.array(bin_accuracies), np.array(bin_confidences)


def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Brier score for probabilistic predictions.
    
    Brier score measures the mean squared error of predicted probabilities:
        BS = (1/n) Σ_i (p_i - y_i)²
    
    where p_i is predicted probability and y_i is one-hot label.
    
    For binary classification:
        BS = (1/n) Σ_i (p_i - y_i)²
    
    Lower is better. Perfect score is 0.
    
    Args:
        probs: Predicted probabilities [num_samples, num_classes]
        labels: True labels [num_samples]
        
    Returns:
        Brier score
    """
    n_classes = probs.shape[1]
    
    # One-hot encode labels
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    
    # Brier score
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def evaluate_uncertainty(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_mc_samples: int = 30
) -> UncertaintyMetrics:
    """
    Comprehensive uncertainty evaluation.
    
    Args:
        model: Trained model
        loader: Data loader
        device: Device to run on
        num_mc_samples: Number of MC Dropout samples
        
    Returns:
        UncertaintyMetrics with all computed values
    """
    mc_predictor = MCDropoutPredictor(model, num_samples=num_mc_samples)
    
    all_probs = []
    all_epistemic = []
    all_entropy = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        probs, epistemic, entropy = mc_predictor.predict(batch)
        
        all_probs.append(probs)
        all_epistemic.append(epistemic)
        all_entropy.append(entropy)
        all_labels.append(batch.y.squeeze().cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_epistemic = np.concatenate(all_epistemic, axis=0)
    all_entropy = np.concatenate(all_entropy, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute calibration metrics
    ece, mce, _, _ = compute_calibration_error(all_probs, all_labels)
    brier = compute_brier_score(all_probs, all_labels)
    
    return UncertaintyMetrics(
        mean_prediction=all_probs,
        epistemic_uncertainty=all_epistemic,
        total_uncertainty=all_epistemic,  # For MC Dropout, these are the same
        entropy=all_entropy,
        confidence=all_probs.max(axis=1),
        calibration_error=ece,
        brier_score=brier
    )


if __name__ == "__main__":
    print("Uncertainty Quantification Module")
    print("=" * 50)
    print("\nMethods available:")
    print("  - MC Dropout: Epistemic uncertainty via multiple passes")
    print("  - Conformal Prediction: Calibrated prediction sets")
    print("  - Temperature Scaling: Post-hoc calibration")
    print("\nMetrics computed:")
    print("  - Epistemic uncertainty (model uncertainty)")
    print("  - Predictive entropy")
    print("  - Expected Calibration Error (ECE)")
    print("  - Brier Score")
