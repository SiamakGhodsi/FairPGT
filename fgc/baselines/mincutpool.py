"""
MinCutPool: Spectral Clustering with Graph Neural Networks
Reference: Bianchi et al., ICML 2020
Using PyG's native MinCutPool implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, dense_mincut_pool
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict


class MinCutPoolGNN(nn.Module):
    """GCN encoder for MinCutPool"""
    def __init__(self, n_features: int, hidden_dim: int, n_clusters: int):
        super().__init__()
        self.gcn1 = GCNConv(n_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, n_clusters)
        
    def forward(self, x, edge_index, edge_weight=None):
        """Returns embeddings and soft assignments"""
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        z = self.gcn2(h, edge_index, edge_weight)
        s = self.linear(z)
        return z, s


class MinCutPool(nn.Module):
    """MinCutPool for graph clustering"""
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.gnn = MinCutPoolGNN(n_features, hidden_dim, n_clusters)
        self.n_clusters = n_clusters
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Returns:
            s: Soft assignment matrix (N, K)
            mincut_loss: MinCut loss
            ortho_loss: Orthogonality loss
        """
        # Get embeddings and logits
        z, logits = self.gnn(x, edge_index, edge_weight)
        
        # Soft assignments
        s = F.softmax(logits, dim=1)
        
        # Convert edge_index to dense adjacency for mincut_pool
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        if edge_weight is not None:
            adj[edge_index[0], edge_index[1]] = edge_weight
        
        # Compute MinCut loss
        _, _, mincut_loss, ortho_loss = dense_mincut_pool(
            z.unsqueeze(0), adj.unsqueeze(0), s.unsqueeze(0)
        )
        
        return s, mincut_loss, ortho_loss
    
    def cluster(self, x, edge_index, edge_weight=None):
        """Get hard cluster assignments"""
        with torch.no_grad():
            s, _, _ = self.forward(x, edge_index, edge_weight)
            return s.argmax(dim=1)


class MinCutPoolExperiment(BaseExperiment):
    """MinCutPool Experiment Wrapper"""
    
    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        lambda_mincut: float = 1.0,
        lambda_ortho: float = 1.0,
        **kwargs
    ):
        super().__init__(
            model_name='MinCutPool',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.lambda_mincut = lambda_mincut
        self.lambda_ortho = lambda_ortho
        
    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        """Build MinCutPool model"""
        model = MinCutPool(
            n_features=n_features,
            n_clusters=self.n_clusters,
            hidden_dim=self.hidden_dim
        )
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        return model
    
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
        s, mincut_loss, ortho_loss = self.model(X, edge_index, edge_weight)
        
        # Total loss
        total_loss = (
            self.lambda_mincut * mincut_loss +
            self.lambda_ortho * ortho_loss
        )

        # DEC clustering signal (MinCutPool regularizers alone don't cluster)
        p = (s ** 2 / (s.sum(dim=0, keepdim=True) + 1e-8))
        p = (p / (p.sum(dim=1, keepdim=True) + 1e-8)).detach()
        loss_cluster = F.kl_div((s + 1e-8).log(), p, reduction='batchmean')
        total_loss = total_loss + loss_cluster
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'loss_mincut': mincut_loss.item(),
            'loss_ortho': ortho_loss.item()
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
            s, _, _ = self.model(X, edge_index, edge_weight)
            hard_assignments = s.argmax(dim=1)
        
        return hard_assignments.cpu().numpy()
