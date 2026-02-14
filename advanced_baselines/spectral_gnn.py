"""
Spectral Graph Neural Networks with Laplacian Regularization
=============================================================

This module implements spectral-based GNN methods that leverage the 
eigenstructure of the graph Laplacian matrix for message passing.

Mathematical Foundation:
-----------------------

1. Graph Laplacian:
   L = D - A
   where D is the degree matrix and A is the adjacency matrix.
   
   Normalized Laplacian:
   L_norm = I - D^{-1/2} A D^{-1/2}

2. Spectral Convolution (exact):
   h' = U g_θ(Λ) U^T h
   where U, Λ are eigenvectors/eigenvalues of L.
   
   This is computationally expensive O(n³), so we use approximations.

3. Chebyshev Approximation (ChebNet):
   g_θ(L) ≈ Σ_{k=0}^{K} θ_k T_k(L̃)
   where T_k are Chebyshev polynomials and L̃ = 2L/λ_max - I

4. Laplacian Regularization:
   R = Tr(H^T L H) = Σ_{(i,j)∈E} ||h_i - h_j||²
   
   This encourages smooth representations where connected 
   nodes have similar embeddings.

5. Dirichlet Energy:
   E(f) = f^T L f = (1/2) Σ_{(i,j)∈E} w_{ij}(f_i - f_j)²
   
   Minimizing this makes node features vary smoothly across edges.

References:
----------
- Defferrard et al., "Convolutional Neural Networks on Graphs with Fast 
  Localized Spectral Filtering" (NeurIPS 2016)
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
  Networks" (ICLR 2017)
- Li et al., "Deeper Insights into Graph Convolutional Networks for 
  Semi-Supervised Learning" (AAAI 2018)

Author: ENIGMA Competition
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool, global_add_pool
from torch_geometric.utils import get_laplacian, to_dense_adj
from typing import Optional, Tuple, List
import math
import numpy as np


def compute_laplacian_eigendecomposition(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalization: str = 'sym',
    k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eigendecomposition of the graph Laplacian.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        normalization: 'sym' for symmetric, 'rw' for random walk
        k: Number of eigenvectors to compute
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # Get Laplacian
    edge_index_lap, edge_weight = get_laplacian(
        edge_index, 
        num_nodes=num_nodes,
        normalization=normalization
    )
    
    # Convert to dense for eigendecomposition
    L = to_dense_adj(edge_index_lap, edge_attr=edge_weight, max_num_nodes=num_nodes)
    L = L.squeeze(0)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    # Return top-k eigenvectors (smallest eigenvalues)
    return eigenvalues[:k], eigenvectors[:, :k]


class LaplacianRegularization(nn.Module):
    """
    Laplacian regularization module for encouraging smooth representations.
    
    Computes the Dirichlet energy:
        E(H) = Tr(H^T L H) = Σ_{(i,j)∈E} ||h_i - h_j||²
    
    This regularization term encourages connected nodes to have 
    similar representations.
    
    Args:
        reduction: How to reduce over the batch ('mean' or 'sum')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Laplacian regularization term.
        
        Args:
            h: Node representations [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Regularization loss scalar
        """
        src, dst = edge_index
        
        # Compute squared differences for connected nodes
        diff = h[src] - h[dst]
        energy = (diff ** 2).sum(dim=-1)  # [num_edges]
        
        # Apply edge weights if provided
        if edge_weight is not None:
            energy = energy * edge_weight
        
        # Reduce
        if self.reduction == 'mean':
            return energy.mean()
        else:
            return energy.sum()


class SpectralConvolution(nn.Module):
    """
    Spectral convolution layer using Chebyshev polynomial approximation.
    
    Mathematical formulation:
        h' = Σ_{k=0}^{K} θ_k T_k(L̃) h
    
    where:
    - T_k are Chebyshev polynomials
    - L̃ = 2L/λ_max - I is the rescaled Laplacian
    - θ_k are learnable parameters
    
    Chebyshev polynomials satisfy the recurrence:
        T_0(x) = 1
        T_1(x) = x
        T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        K: Order of Chebyshev polynomials (filter size)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        
        # Use PyG's optimized ChebConv
        self.conv = ChebConv(in_channels, out_channels, K)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        lambda_max: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
            lambda_max: Maximum eigenvalue for normalization
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        return self.conv(x, edge_index, edge_weight, lambda_max=lambda_max)


class SpectralGNN(nn.Module):
    """
    Spectral Graph Neural Network with Laplacian regularization.
    
    This model combines:
    1. Chebyshev spectral convolutions for efficient spectral filtering
    2. Laplacian regularization to encourage smooth representations
    3. Optional positional encodings from Laplacian eigenvectors
    
    The total loss is:
        L_total = L_classification + α * L_laplacian
    
    where α controls the smoothness regularization strength.
    
    Architecture:
    - Node embedding layer
    - K spectral convolution layers
    - Graph-level pooling
    - Classification head
    
    Args:
        in_channels: Number of input node features
        hidden_channels: Hidden dimension
        out_channels: Number of output classes
        num_layers: Number of spectral convolution layers
        K: Chebyshev polynomial order
        dropout: Dropout probability
        laplacian_weight: Weight for Laplacian regularization (α)
        use_positional_encoding: Whether to add Laplacian eigenvector features
        num_eigenvectors: Number of eigenvectors for positional encoding
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        K: int = 3,
        dropout: float = 0.2,
        laplacian_weight: float = 0.01,
        use_positional_encoding: bool = False,
        num_eigenvectors: int = 8
    ):
        super().__init__()
        
        self.laplacian_weight = laplacian_weight
        self.use_positional_encoding = use_positional_encoding
        self.num_eigenvectors = num_eigenvectors
        
        # Adjust input channels if using positional encoding
        actual_in_channels = in_channels
        if use_positional_encoding:
            actual_in_channels += num_eigenvectors
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(actual_in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spectral convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                ChebConv(hidden_channels, hidden_channels, K)
            )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        # Laplacian regularization
        self.laplacian_reg = LaplacianRegularization(reduction='mean')
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.dropout = dropout
        
        # Store regularization loss for training
        self.reg_loss = 0.0
    
    def _add_positional_encoding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Add Laplacian eigenvector positional encodings.
        
        This adds the k smallest eigenvectors of the Laplacian as 
        additional node features, providing positional information.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices
            batch: Batch indices
            
        Returns:
            Node features with positional encoding [num_nodes, in_channels + k]
        """
        # For simplicity, compute per-graph and concatenate
        # In practice, you might want to batch this more efficiently
        device = x.device
        num_nodes = x.size(0)
        
        # Initialize positional encoding tensor
        pos_enc = torch.zeros(num_nodes, self.num_eigenvectors, device=device)
        
        # Get unique graphs in batch
        unique_graphs = batch.unique()
        
        for graph_id in unique_graphs:
            mask = batch == graph_id
            graph_nodes = mask.sum().item()
            
            if graph_nodes < self.num_eigenvectors:
                # Not enough nodes, use zeros
                continue
            
            # Get subgraph edge indices
            node_map = torch.where(mask)[0]
            node_to_new = torch.full((num_nodes,), -1, device=device)
            node_to_new[node_map] = torch.arange(graph_nodes, device=device)
            
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            sub_edge_index = edge_index[:, edge_mask]
            sub_edge_index = node_to_new[sub_edge_index]
            
            # Compute Laplacian eigenvectors
            try:
                _, eigvecs = compute_laplacian_eigendecomposition(
                    sub_edge_index, graph_nodes, k=self.num_eigenvectors
                )
                pos_enc[mask, :eigvecs.size(1)] = eigvecs[:, :self.num_eigenvectors]
            except:
                # Fall back to zeros if eigendecomposition fails
                pass
        
        return torch.cat([x, pos_enc], dim=-1)
    
    def forward(self, data, return_reg_loss: bool = False):
        """
        Forward pass.
        
        Args:
            data: PyG Data object
            return_reg_loss: Whether to return regularization loss
            
        Returns:
            Logits [batch_size, out_channels]
            (Optional) Regularization loss
        """
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self._add_positional_encoding(x, edge_index, batch)
        
        # Node embedding
        x = self.node_embed(x)
        
        # Store intermediate representations for regularization
        hidden_states = [x]
        
        # Spectral convolutions
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden_states.append(x)
        
        # Compute Laplacian regularization on all hidden states
        self.reg_loss = 0.0
        for h in hidden_states:
            self.reg_loss += self.laplacian_reg(h, edge_index)
        self.reg_loss /= len(hidden_states)
        
        # Graph-level pooling
        graph_embed = global_mean_pool(x, batch)
        
        # Classification
        out = self.classifier(graph_embed)
        
        if return_reg_loss:
            return out, self.reg_loss
        return out
    
    def get_loss(
        self,
        data,
        criterion: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss including regularization.
        
        Args:
            data: PyG Data object
            criterion: Classification loss function
            
        Returns:
            Tuple of (total_loss, classification_loss, regularization_loss)
        """
        out, reg_loss = self.forward(data, return_reg_loss=True)
        cls_loss = criterion(out, data.y.squeeze().long())
        total_loss = cls_loss + self.laplacian_weight * reg_loss
        
        return total_loss, cls_loss, reg_loss


class GraphDiffusionConvolution(nn.Module):
    """
    Graph Diffusion Convolution based on the heat equation.
    
    Mathematical formulation:
    
    The heat equation on graphs:
        ∂h/∂t = -Lh
    
    has the solution:
        h(t) = exp(-tL) h(0)
    
    We approximate this using truncated Taylor expansion:
        exp(-tL) ≈ Σ_{k=0}^{K} (-tL)^k / k!
    
    This allows information to diffuse smoothly across the graph,
    with the diffusion time t controlling the spread.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        K: Number of diffusion steps (Taylor expansion order)
        diffusion_time: Diffusion time parameter t
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 10,
        diffusion_time: float = 1.0
    ):
        super().__init__()
        
        self.K = K
        self.t = diffusion_time
        
        # Learnable weights for each diffusion step
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels) / math.sqrt(in_channels))
            for _ in range(K + 1)
        ])
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Forward pass with diffusion-based convolution.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            
        Returns:
            Diffused node features [num_nodes, out_channels]
        """
        # Get normalized Laplacian
        edge_index_lap, edge_weight = get_laplacian(
            edge_index,
            num_nodes=num_nodes,
            normalization='sym'
        )
        
        # Build sparse Laplacian matrix
        L = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight,
            (num_nodes, num_nodes)
        ).to_dense()
        
        # Compute diffusion: exp(-tL) ≈ Σ_{k=0}^{K} (-tL)^k / k!
        out = torch.zeros(num_nodes, self.weights[0].size(1), device=x.device)
        
        h_k = x  # (-tL)^0 h = h
        factorial = 1
        
        for k in range(self.K + 1):
            # Weight this diffusion step
            out += (h_k @ self.weights[k]) / factorial
            
            # Compute next power: (-tL)^{k+1} h
            if k < self.K:
                h_k = -self.t * (L @ h_k)
                factorial *= (k + 1)
        
        return out + self.bias


def train_spectral_gnn(
    model: SpectralGNN,
    train_loader,
    valid_loader,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None
):
    """
    Train Spectral GNN with Laplacian regularization.
    
    Args:
        model: SpectralGNN instance
        train_loader: Training data loader
        valid_loader: Validation data loader
        device: Device to train on
        num_epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization
        class_weights: Optional class weights
        
    Returns:
        Best validation F1 score
    """
    from sklearn.metrics import f1_score
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    best_state = None
    patience = 15
    no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            loss, cls_loss, reg_loss = model.get_loss(batch, criterion)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            total_cls_loss += cls_loss.item() * batch.num_graphs
            total_reg_loss += reg_loss.item() * batch.num_graphs
        
        n = len(train_loader.dataset)
        train_loss = total_loss / n
        cls_loss_avg = total_cls_loss / n
        reg_loss_avg = total_reg_loss / n
        
        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1).cpu().numpy()
                y_true.extend(batch.y.cpu().numpy().flatten())
                y_pred.extend(pred)
        
        val_f1 = f1_score(y_true, y_pred, average='macro')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} (cls: {cls_loss_avg:.4f}, reg: {reg_loss_avg:.4f}) | Val F1: {val_f1:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_val_f1


if __name__ == "__main__":
    print("Spectral Graph Neural Networks")
    print("=" * 50)
    print("\nFeatures:")
    print("  - Chebyshev spectral convolutions")
    print("  - Laplacian regularization for smoothness")
    print("  - Optional positional encodings")
    print("  - Graph diffusion convolutions")
    print("\nMathematical foundation:")
    print("  - Spectral graph theory")
    print("  - Heat equation on graphs")
    print("  - Dirichlet energy minimization")
