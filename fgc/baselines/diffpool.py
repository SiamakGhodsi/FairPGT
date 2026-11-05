"""
DiffPool: Differentiable Pooling for Graph Classification
Reference: Ying et al., NeurIPS 2018
Adapted for clustering using PyG's native DiffPool implementation
"""

import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict


class DiffPoolGNN(nn.Module):
    """GCN layers for DiffPool"""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = DenseGCNConv(in_dim, hidden_dim)
        self.gcn2 = DenseGCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, adj, mask=None):
        x = F.relu(self.gcn1(x, adj, mask))
        x = self.gcn2(x, adj, mask)
        return x


class DiffPool(nn.Module):
    """DiffPool for graph clustering"""
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        # Embedding GNN
        self.embed_gnn = DiffPoolGNN(n_features, hidden_dim)
        
        # Pooling GNN (produces soft assignments)
        self.pool_gnn = DiffPoolGNN(n_features, n_clusters)
        
        self.n_clusters = n_clusters
        
    def forward(self, x, adj):
        """
        Args:
            x: Node features (N, D)
            adj: Dense adjacency matrix (N, N)
        
        Returns:
            s: Soft assignment matrix (N, K)
            x_pooled: Pooled node features (K, hidden_dim)
            adj_pooled: Pooled adjacency (K, K)
            link_loss: Link prediction loss
            ent_loss: Entropy loss
        """
        # Embeddings
        z = self.embed_gnn(x.unsqueeze(0), adj.unsqueeze(0))
        
        # Soft assignments
        s = self.pool_gnn(x.unsqueeze(0), adj.unsqueeze(0))
        s = F.softmax(s, dim=-1)
        
        # Pool
        x_pooled, adj_pooled, link_loss, ent_loss = dense_diff_pool(
            z, adj.unsqueeze(0), s
        )
        
        return s.squeeze(0), x_pooled, adj_pooled, link_loss, ent_loss
    
    def cluster(self, x, adj):
        """Get hard cluster assignments"""
        with torch.no_grad():
            s, _, _, _, _ = self.forward(x, adj)
            return s.argmax(dim=1)


class DiffPoolExperiment(BaseExperiment):
    """DiffPool Experiment Wrapper"""
    
    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
        lambda_link: float = 1.0,
        lambda_ent: float = 1.0,
        **kwargs
    ):
        super().__init__(
            model_name='DiffPool',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.lambda_link = lambda_link
        self.lambda_ent = lambda_ent
        
    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        """Build DiffPool model"""
        model = DiffPool(
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
        
        # Convert to dense (DiffPool requires dense adjacency)
        if sp.issparse(A):
            if A.shape[0] > 50000:
                raise RuntimeError(f"DiffPool requires dense A, too large: {A.shape[0]} nodes")
            A = torch.from_numpy(A.toarray()).float().to(self.device)
            
        if A.is_sparse:
            A_dense = A.to_dense()
        else:
            A_dense = A
        
        # Forward pass
        s, x_pooled, adj_pooled, link_loss, ent_loss = self.model(X, A_dense)
        
        # Total loss
        total_loss = self.lambda_link * link_loss + self.lambda_ent * ent_loss

        # DEC clustering signal (DiffPool has no clustering objective by default)
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
            'loss_link': link_loss.item(),
            'loss_ent': ent_loss.item()
        }
    
    def predict(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        **kwargs
    ) -> np.ndarray:
        """Predict cluster assignments"""
        self.model.eval()
        
        # Convert to dense
        if A.is_sparse:
            A_dense = A.to_dense()
        else:
            A_dense = A
        
        with torch.no_grad():
            s, _, _, _, _ = self.model(X, A_dense)
            hard_assignments = s.argmax(dim=1)
        
        return hard_assignments.cpu().numpy()
