"""
SDCN: Structural Deep Clustering Network (2020)
Reference: https://dl.acm.org/doi/10.1145/3366423.3380214
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict


class GCNAutoEncoder(nn.Module):
    """Graph Convolutional Autoencoder"""
    def __init__(self, n_features: int, hidden_dims: list):
        super().__init__()
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        dims = [n_features] + hidden_dims
        for i in range(len(dims) - 1):
            self.encoder_layers.append(GCNConv(dims[i], dims[i+1]))
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        rev_dims = hidden_dims[::-1] + [n_features]
        for i in range(len(rev_dims) - 1):
            self.decoder_layers.append(GCNConv(rev_dims[i], rev_dims[i+1]))
    
    def encode(self, x, edge_index, edge_weight=None):
        """Encode to latent space"""
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.encoder_layers) - 1:
                x = F.relu(x)
        return x
    
    def decode(self, z, edge_index, edge_weight=None):
        """Decode from latent space"""
        x = z
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.decoder_layers) - 1:
                x = F.relu(x)
        return x
    
    def forward(self, x, edge_index, edge_weight=None):
        """Full forward pass"""
        z = self.encode(x, edge_index, edge_weight)
        x_recon = self.decode(z, edge_index, edge_weight)
        return z, x_recon


class ClusteringLayer(nn.Module):
    """Clustering layer with learnable cluster centers"""
    def __init__(self, n_clusters: int, hidden_dim: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Initialize cluster centers
        self.cluster_centers = nn.Parameter(
            torch.randn(n_clusters, hidden_dim)
        )
        
    def forward(self, z):
        """
        Compute soft assignments using Student's t-distribution
        
        q_ij = (1 + ||z_i - μ_j||^2 / α)^(-(α+1)/2) / Σ_j' ...
        """
        # Compute pairwise distances
        dist = torch.cdist(z, self.cluster_centers, p=2)  # (N, K)
        
        # Student's t-distribution
        q = 1.0 / (1.0 + dist ** 2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        
        return q
    
    @staticmethod
    def target_distribution(q):
        """
        Compute target distribution p
        
        p_ij = q_ij^2 / Σ_i q_ij / Σ_j' (q_ij'^2 / Σ_i q_ij')
        """
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        p = weight / weight.sum(dim=1, keepdim=True)
        return p


class SDCN(nn.Module):
    """SDCN: Structural Deep Clustering Network"""
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        hidden_dims: list = [500, 500, 2000, 10],
        alpha: float = 1.0
    ):
        super().__init__()
        
        self.autoencoder = GCNAutoEncoder(n_features, hidden_dims)
        self.clustering_layer = ClusteringLayer(n_clusters, hidden_dims[-1], alpha)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Returns:
            z: Latent embeddings
            x_recon: Reconstructed features
            q: Soft cluster assignments
        """
        z, x_recon = self.autoencoder(x, edge_index, edge_weight)
        q = self.clustering_layer(z)
        return z, x_recon, q
    
    def cluster(self, x, edge_index, edge_weight=None):
        """Get hard cluster assignments"""
        with torch.no_grad():
            z, _, q = self.forward(x, edge_index, edge_weight)
            return q.argmax(dim=1)


class SDCNExperiment(BaseExperiment):
    """SDCN Experiment Wrapper"""
    
    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dims: list = [500, 500, 2000, 10],
        lr: float = 0.001,
        alpha: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_kl: float = 0.1,
        pretrain_epochs: int = 50,
        **kwargs
    ):
        super().__init__(
            model_name='SDCN',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )
        
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.alpha = alpha
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.pretrain_epochs = pretrain_epochs
        self.is_pretrained = False
        
    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        """Build SDCN model"""
        model = SDCN(
            n_features=n_features,
            n_clusters=self.n_clusters,
            hidden_dims=self.hidden_dims,
            alpha=self.alpha
        )
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        return model
    
    def pretrain(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        **kwargs
    ):
        """Pretrain autoencoder"""
        self.logger.info("Pretraining autoencoder...")
        
        # Convert to edge_index
        edge_index, edge_weight = self._extract_edge_info(A)
        #if A.is_sparse:
        #    edge_index = A._indices()
        #    edge_weight = A._values()
        #else:
        #    edge_index = A.nonzero().t()
        #    edge_weight = A[edge_index[0], edge_index[1]]
        
        for epoch in range(self.pretrain_epochs):
            self.model.train()
            
            z, x_recon = self.model.autoencoder(X, edge_index, edge_weight)
            
            # Reconstruction loss
            loss = F.mse_loss(x_recon, X)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                self.logger.info(f"Pretrain Epoch {epoch:03d} | Loss: {loss.item():.4f}")
        
        self.is_pretrained = True
    
    def train_epoch(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        S: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """Train for one epoch"""
        # Pretrain if not done
        if not self.is_pretrained:
            self.pretrain(A, X)
        
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
        z, x_recon, q = self.model(X, edge_index, edge_weight)
        
        # Reconstruction loss
        loss_recon = F.mse_loss(x_recon, X)
        
        # KL divergence loss with target distribution
        p = ClusteringLayer.target_distribution(q)
        p = p.detach()  # Detach target distribution to prevent gradients
        loss_kl = F.kl_div(
            q.log(),
            p,
            reduction='batchmean'
        )
        
        # Total loss
        total_loss = (
            self.lambda_recon * loss_recon +
            self.lambda_kl * loss_kl
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_kl': loss_kl.item()
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
            _, _, q = self.model(X, edge_index, edge_weight)
            hard_assignments = q.argmax(dim=1)
        
        return hard_assignments.cpu().numpy()
