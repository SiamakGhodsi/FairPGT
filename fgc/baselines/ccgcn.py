"""
CCGCN: Cluster-Guided Contrastive Graph Clustering Network
Reference: https://github.com/xihongyang1999/CCGC (AAAI 2023)
Optimized implementation with PyG acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict


class GCNEncoder(nn.Module):
    """GCN encoder for CCGCN"""
    def __init__(self, n_features: int, hidden_dim: int, n_clusters: int):
        super().__init__()
        self.gcn1 = GCNConv(n_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, n_clusters)
        self.prelu = nn.PReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        h = self.gcn1(x, edge_index, edge_weight)
        h = self.prelu(h)
        z = self.gcn2(h, edge_index, edge_weight)
        return z


class CCGCN(nn.Module):
    """CCGCN model with cluster-guided contrastive learning"""
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        hidden_dim: int = 256,
        temperature: float = 0.5
    ):
        super().__init__()
        self.encoder = GCNEncoder(n_features, hidden_dim, n_clusters)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, n_clusters))
        self.temperature = temperature
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Returns:
            z: Node embeddings
            q: Soft cluster assignments
        """
        z = self.encoder(x, edge_index, edge_weight)
        z_norm = F.normalize(z, dim=1)
        
        # Soft assignments via similarity to cluster centers
        centers_norm = F.normalize(self.cluster_centers, dim=1)
        q = torch.mm(z_norm, centers_norm.t()) / self.temperature
        q = F.softmax(q, dim=1)
        
        return z, q
    
    def cluster(self, x, edge_index, edge_weight=None):
        """Get hard cluster assignments"""
        with torch.no_grad():
            _, q = self.forward(x, edge_index, edge_weight)
            return q.argmax(dim=1)


class CCGCNExperiment(BaseExperiment):
    """CCGCN Experiment Wrapper"""
    
    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 256,
        lr: float = 0.0001,
        temperature: float = 0.5,
        lambda_cluster: float = 1.0,
        lambda_contrastive: float = 1.0,
        **kwargs
    ):
        super().__init__(
            model_name='CCGCN',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.temperature = temperature
        self.lambda_cluster = lambda_cluster
        self.lambda_contrastive = lambda_contrastive
        
    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        """Build CCGCN model"""
        model = CCGCN(
            n_features=n_features,
            n_clusters=self.n_clusters,
            hidden_dim=self.hidden_dim,
            temperature=self.temperature
        )
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        return model
    
    def compute_contrastive_loss(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Cluster-guided contrastive loss
        Positive pairs: nodes in same cluster
        Negative pairs: nodes in different clusters
        """
        # Get cluster assignments
        cluster_ids = q.argmax(dim=1)
        
        # Normalize embeddings
        z_norm = F.normalize(z, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature
        
        # Create positive mask (same cluster)
        pos_mask = (cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # exclude self
        
        # Create negative mask (different cluster)
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)
        
        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        
        # Positive similarities
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        
        # All similarities (excluding self)
        all_sim = (exp_sim * (1 - torch.eye(len(z), device=z.device))).sum(dim=1)
        
        # Loss: -log(pos / all)
        loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8))
        
        return loss.mean()
    
    def compute_cluster_loss(self, q: torch.Tensor) -> torch.Tensor:
        """
        Clustering objective: maximize confidence and balance
        """
        # Confidence: encourage peaky distributions
        confidence = -(q * torch.log(q + 1e-8)).sum(dim=1).mean()
        
        # Balance: encourage uniform cluster sizes
        cluster_sizes = q.sum(dim=0)
        target_size = len(q) / self.n_clusters
        balance = ((cluster_sizes - target_size) ** 2).mean()
        
        return -confidence + 0.1 * balance
    
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
        z, q = self.model(X, edge_index, edge_weight)
        
        # Compute losses
        loss_contrastive = self.compute_contrastive_loss(z, q)
        loss_cluster = self.compute_cluster_loss(q)
        
        total_loss = (
            self.lambda_contrastive * loss_contrastive +
            self.lambda_cluster * loss_cluster
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_cluster': loss_cluster.item()
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
            _, q = self.model(X, edge_index, edge_weight)
            hard_assignments = q.argmax(dim=1)
        
        return hard_assignments.cpu().numpy()
