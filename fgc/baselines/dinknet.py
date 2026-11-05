"""
DinkNet: Neural Clustering on Large Graphs
Reference: https://github.com/yueliu1999/Dink-Net (ICML 2023)
Scalable with dilation-shrink mechanism for mini-batch training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict


class DinkNetEncoder(nn.Module):
    """GCN encoder with contrastive learning"""
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = GCNConv(n_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.activation = nn.PReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        h = self.activation(self.gcn1(x, edge_index, edge_weight))
        z = self.gcn2(h, edge_index, edge_weight)
        return F.normalize(z, dim=1)


class DinkNet(nn.Module):
    """DinkNet: Dilation and Shrink for Scalable Clustering"""
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        hidden_dim: int = 256,
        tradeoff: float = 0.5
    ):
        super().__init__()
        self.encoder = DinkNetEncoder(n_features, hidden_dim)
        
        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(
            F.normalize(torch.randn(n_clusters, hidden_dim), dim=1)
        )
        
        self.tradeoff = tradeoff
        self.n_clusters = n_clusters
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Returns:
            z: Normalized embeddings
            q: Soft cluster assignments
        """
        z = self.encoder(x, edge_index, edge_weight)
        
        # Compute similarity to cluster centers
        centers_norm = F.normalize(self.cluster_centers, dim=1)
        sim = torch.mm(z, centers_norm.t())
        q = F.softmax(sim, dim=1)
        
        return z, q
    
    def cluster(self, x, edge_index, edge_weight=None):
        """Get hard cluster assignments"""
        with torch.no_grad():
            _, q = self.forward(x, edge_index, edge_weight)
            return q.argmax(dim=1)


class DinkNetExperiment(BaseExperiment):
    """DinkNet Experiment Wrapper"""
    
    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 256,
        lr: float = 0.0001,
        tradeoff: float = 0.5,
        temperature: float = 0.5,
        **kwargs
    ):
        super().__init__(
            model_name='DinkNet',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.tradeoff = tradeoff
        self.temperature = temperature
        
    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        """Build DinkNet model"""
        model = DinkNet(
            n_features=n_features,
            n_clusters=self.n_clusters,
            hidden_dim=self.hidden_dim,
            tradeoff=self.tradeoff
        )
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        return model
    
    def compute_dilation_loss(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Dilation loss: maximize distance between different clusters
        Push cluster centers apart
        """
        # Get cluster assignments
        cluster_ids = q.argmax(dim=1)
        
        # Compute cluster centroids from data
        centroids = []
        for k in range(self.n_clusters):
            mask = (cluster_ids == k)
            if mask.sum() > 0:
                centroids.append(z[mask].mean(dim=0))
            else:
                centroids.append(z.mean(dim=0))  # fallback
        centroids = torch.stack(centroids)
        centroids = F.normalize(centroids, dim=1)
        
        # Pairwise distances between centroids
        centroid_sim = torch.mm(centroids, centroids.t())
        
        # Dilation: minimize similarity between different clusters
        mask = 1 - torch.eye(self.n_clusters, device=z.device)
        dilation = (centroid_sim * mask).sum() / (self.n_clusters * (self.n_clusters - 1))
        
        return dilation
    
    def compute_shrink_loss(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Shrink loss: minimize distance within clusters
        Pull nodes closer to their cluster centers
        """
        # Similarity to assigned cluster centers
        centers_norm = F.normalize(self.model.cluster_centers, dim=1)
        sim_to_centers = torch.mm(z, centers_norm.t())
        
        # Get assigned cluster for each node
        cluster_ids = q.argmax(dim=1)
        
        # Shrink: maximize similarity to assigned cluster
        assigned_sim = sim_to_centers[torch.arange(len(z)), cluster_ids]
        shrink = -assigned_sim.mean()
        
        return shrink
    
    def compute_contrastive_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Self-supervised contrastive loss
        Distinguish original vs. augmented views
        """
        # Simple augmentation: add noise
        z_aug = z + 0.1 * torch.randn_like(z)
        z_aug = F.normalize(z_aug, dim=1)
        
        # Positive pairs: (z_i, z_i_aug)
        pos_sim = (z * z_aug).sum(dim=1) / self.temperature
        
        # Negative pairs: all other nodes
        neg_sim = torch.mm(z, z_aug.t()) / self.temperature
        
        # InfoNCE
        exp_pos = torch.exp(pos_sim)
        exp_neg = torch.exp(neg_sim).sum(dim=1) - exp_pos  # exclude self
        
        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8))
        return loss.mean()
    
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
        loss_dilation = self.compute_dilation_loss(z, q)
        loss_shrink = self.compute_shrink_loss(z, q)
        loss_contrastive = self.compute_contrastive_loss(z)
        
        # Combined loss with tradeoff
        total_loss = (
            self.tradeoff * loss_dilation +
            (1 - self.tradeoff) * loss_shrink +
            0.1 * loss_contrastive
        )

        # DEC clustering signal
        p = (q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8))
        p = (p / (p.sum(dim=1, keepdim=True) + 1e-8)).detach()
        loss_dec = F.kl_div((q + 1e-8).log(), p, reduction='batchmean')
        total_loss = total_loss + loss_dec
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'loss_dilation': loss_dilation.item(),
            'loss_shrink': loss_shrink.item(),
            'loss_contrastive': loss_contrastive.item()
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
