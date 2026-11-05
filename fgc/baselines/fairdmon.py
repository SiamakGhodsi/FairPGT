"""
FairDMoN: Modularity-Fair Deep Community Detection
Reference: Gkartzios et al., ICDM 2025
Combines DMoN's modularity optimization with fairness constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict


class FairDMoNGCN(nn.Module):
    """GCN encoder for FairDMoN"""
    def __init__(self, n_features: int, hidden_dim: int, n_clusters: int, dropout: float = 0.0):
        super().__init__()
        self.gcn1 = GCNConv(n_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, n_clusters)
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index, edge_weight)
        return F.softmax(x, dim=1)


class FairDMoN(nn.Module):
    """
    FairDMoN: Modularity optimization + Fairness constraints
    """
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        hidden_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.encoder = FairDMoNGCN(n_features, hidden_dim, n_clusters, dropout)
        self.n_clusters = n_clusters
        
    def forward(self, x, edge_index, edge_weight=None):
        """Get soft cluster assignments"""
        return self.encoder(x, edge_index, edge_weight)
    
    def cluster(self, x, edge_index, edge_weight=None):
        """Get hard cluster assignments"""
        with torch.no_grad():
            assignments = self.forward(x, edge_index, edge_weight)
            return assignments.argmax(dim=1)


class FairDMoNExperiment(BaseExperiment):
    """FairDMoN Experiment Wrapper"""
    
    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        dropout: float = 0.0,
        lambda_modularity: float = 1.0,
        lambda_collapse: float = 1.0,
        lambda_fairness: float = 0.1,
        fairness_type: str = 'statistical_parity',
        **kwargs
    ):
        super().__init__(
            model_name='FairDMoN',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.lambda_modularity = lambda_modularity
        self.lambda_collapse = lambda_collapse
        self.lambda_fairness = lambda_fairness
        self.fairness_type = fairness_type
        
    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        """Build FairDMoN model"""
        model = FairDMoN(
            n_features=n_features,
            n_clusters=self.n_clusters,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        return model
    
    def compute_modularity_loss(
        self,
        assignments: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_nodes: int
    ) -> torch.Tensor:
        """
        Compute differentiable modularity loss
        
        L_mod = -Tr(S^T (A - dd^T/(2m)) S) / (2m)
        """
        # Get adjacency matrix from edge_index
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)
        m2 = edge_weight.sum()
    
        K = assignments.shape[1]
        degree = torch.zeros(n_nodes, device=assignments.device)
        degree.scatter_add_(0, src, edge_weight)
    
        # Sparse modularity: -(1/2m) Σ_e (S_src^T S_dst) + (1/2m) (d^T S)^2 / (2m)
        neighbor_S = torch.zeros(n_nodes, K, device=assignments.device)
        neighbor_S.scatter_add_(0, src.unsqueeze(1).expand(-1, K), 
                            assignments[dst] * edge_weight.unsqueeze(1))
    
        edge_term = (neighbor_S * assignments).sum() / m2
        degree_term = ((degree.unsqueeze(1) * assignments).sum(dim=0) ** 2).sum() / (m2 ** 2)
    
        modularity = edge_term - degree_term
        return -modularity
    
    def compute_collapse_loss(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        Prevent all nodes from collapsing into one cluster
        """
        K = assignments.shape[1]
        cluster_sizes = assignments.sum(dim=0) / assignments.shape[0]
        
        sqrt_K = torch.sqrt(torch.tensor(K, dtype=torch.float32, device=assignments.device))
        #norm = torch.norm(torch.sqrt(cluster_sizes + 1e-10))
        norm = torch.norm(cluster_sizes + 1e-10)
        
        return sqrt_K * norm - 1.0
    
    def compute_fairness_loss(
        self,
        assignments: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """
        Fairness constraint based on selected type
        """
        if self.fairness_type == 'statistical_parity':
            return self._statistical_parity_loss(assignments, S)
        elif self.fairness_type == 'equal_opportunity':
            return self._equal_opportunity_loss(assignments, S)
        else:
            return self._statistical_parity_loss(assignments, S)
    
    def _statistical_parity_loss(
        self,
        assignments: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """
        Statistical Parity: p_k(g) ≈ p*(g)
        """
        n = len(assignments)
        groups = torch.unique(S)
        
        # Global proportions
        p_star = torch.stack([
            (S == g).float().sum() / n for g in groups
        ])
        
        # Per-cluster deviations
        fairness_loss = 0.0
        for k in range(self.n_clusters):
            cluster_probs = assignments[:, k]
            cluster_size = cluster_probs.sum() + 1e-8
            
            p_k = torch.stack([
                (cluster_probs * (S == g).float()).sum() / cluster_size
                for g in groups
            ])
            
            # L2 deviation
            fairness_loss += ((p_k - p_star) ** 2).sum()
        
        return fairness_loss / self.n_clusters
    
    def _equal_opportunity_loss(
        self,
        assignments: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """
        Equal Opportunity: P(cluster|group) similar across groups
        """
        groups = torch.unique(S)
        
        # Assignment rates per group
        assignment_rates = []
        for g in groups:
            mask = (S == g).float()
            group_size = mask.sum() + 1e-8
            
            # Average assignment probability for this group
            group_assignments = (assignments * mask.unsqueeze(1)).sum(dim=0) / group_size
            assignment_rates.append(group_assignments)
        
        # Minimize variance across groups
        assignment_rates = torch.stack(assignment_rates)
        fairness_loss = assignment_rates.var(dim=0).sum()
        
        return fairness_loss
    
    def train_epoch(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        S: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Convert to edge_index
        edge_index, edge_weight = self._extract_edge_info(A)
        #if A.is_sparse:
        #    edge_index = A._indices()
        #    edge_weight = A._values()
        #else:
        #    edge_index = A.nonzero().t()
        #    edge_weight = A[edge_index[0], edge_index[1]]
        
        # Forward pass
        assignments = self.model(X, edge_index, edge_weight)
        
        # Compute losses
        loss_mod = self.compute_modularity_loss(
            assignments, edge_index, edge_weight, X.shape[0]
        )
        loss_collapse = self.compute_collapse_loss(assignments)
        loss_fair = self.compute_fairness_loss(assignments, S)
        
        total_loss = (
            self.lambda_modularity * loss_mod +
            self.lambda_collapse * loss_collapse +
            self.lambda_fairness * loss_fair
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'loss_modularity': loss_mod.item(),
            'loss_collapse': loss_collapse.item(),
            'loss_fairness': loss_fair.item()
        }
    
    def predict(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        **kwargs
    ) -> np.ndarray:
        """Predict cluster assignments"""
        self.model.eval()
        
        # Convert to edge_index
        edge_index, edge_weight = self._extract_edge_info(A)
        #if A.is_sparse:
        #    edge_index = A._indices()
        #    edge_weight = A._values()
        #else:
        #    edge_index = A.nonzero().t()
        #    edge_weight = A[edge_index[0], edge_index[1]]
        
        with torch.no_grad():
            assignments = self.model(X, edge_index, edge_weight)
            hard_assignments = assignments.argmax(dim=1)
        
        return hard_assignments.cpu().numpy()
