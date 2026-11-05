"""
FairGT: Fairness-aware Graph Transformer
Reference: IJCAI 2024
Global multi-head attention with sensitive attribute encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict


class FairGTEncoder(nn.Module):
    """Graph Transformer with fairness-aware encodings"""
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Initial projection
        self.proj = nn.Linear(n_features, hidden_dim)
        
        # Transformer layers (PyG's TransformerConv)
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                hidden_dim,
                hidden_dim // n_heads,
                heads=n_heads,
                dropout=dropout,
                edge_dim=None
            )
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (N, D)
            edge_index: Graph edges (2, E)
        Returns:
            z: Embeddings (N, hidden_dim)
        """
        x = self.proj(x)
        
        for transformer, ln in zip(self.transformer_layers, self.layer_norms):
            # Transformer conv
            h = transformer(x, edge_index)
            h = self.dropout(h)
            
            # Residual + layer norm
            x = ln(x + h)
        
        return x


class SensitiveAttributeEncoder(nn.Module):
    """Encode sensitive attributes for fairness"""
    def __init__(self, n_groups: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_groups, embed_dim)
        
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: Sensitive attributes (N,)
        Returns:
            Sensitive embeddings (N, embed_dim)
        """
        return self.embedding(S)


class FairGT(nn.Module):
    """Fairness-aware Graph Transformer"""
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        n_groups: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        use_sensitive_encoding: bool = True
    ):
        super().__init__()
        
        self.use_sensitive_encoding = use_sensitive_encoding
        
        # Main encoder
        self.encoder = FairGTEncoder(n_features, hidden_dim, n_heads)
        
        # Sensitive attribute encoder
        if use_sensitive_encoding:
            self.sensitive_encoder = SensitiveAttributeEncoder(n_groups, hidden_dim // 4)
            self.fusion = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        # Clustering head
        self.cluster_head = nn.Linear(hidden_dim, n_clusters)
        
    def forward(self, x, edge_index, S=None):
        """
        Args:
            x: Node features
            edge_index: Graph edges
            S: Sensitive attributes (optional)
        Returns:
            z: Embeddings
            q: Soft cluster assignments
        """
        # Encode graph structure
        z = self.encoder(x, edge_index)
        
        # Fuse with sensitive encoding
        if self.use_sensitive_encoding and S is not None:
            s_embed = self.sensitive_encoder(S)
            z = self.fusion(torch.cat([z, s_embed], dim=1))
        
        # Cluster assignments
        logits = self.cluster_head(z)
        q = F.softmax(logits, dim=1)
        
        return z, q
    
    def cluster(self, x, edge_index, S=None):
        """Get hard cluster assignments"""
        with torch.no_grad():
            _, q = self.forward(x, edge_index, S)
            return q.argmax(dim=1)


class FairGTExperiment(BaseExperiment):
    """FairGT Experiment Wrapper"""
    
    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        lr: float = 0.001,
        lambda_fair: float = 0.1,
        lambda_cluster: float = 1.0,
        lambda_separation: float = 0.5,
        **kwargs
    ):
        super().__init__(
            model_name='FairGT',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.lr = lr
        self.lambda_fair = lambda_fair
        self.lambda_cluster = lambda_cluster
        self.lambda_separation = lambda_separation
        self.n_groups = None  # Will be set during model building
        
    def build_model(self, n_features: int, n_nodes: int, **kwargs) -> nn.Module:
        """Build FairGT model"""
        # We'll infer n_groups from data during training
        # For now, use a default
        self.n_groups = kwargs.get('n_groups', 10)
        
        model = FairGT(
            n_features=n_features,
            n_clusters=self.n_clusters,
            n_groups=self.n_groups,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads
        )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        return model
    
    def compute_fairness_loss(self, q: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Fairness constraint: minimize ΔSP"""
        n = len(q)
        groups = torch.unique(S)
        
        # Update n_groups if needed
        if len(groups) > self.n_groups:
            self.n_groups = len(groups)
        
        # Global proportions
        p_star = torch.stack([
            (S == g).float().mean() for g in groups
        ])
        
        # Per-cluster fairness
        fairness_loss = 0.0
        for k in range(self.n_clusters):
            cluster_probs = q[:, k]
            cluster_size = cluster_probs.sum() + 1e-8
            
            p_k = torch.stack([
                (cluster_probs * (S == g).float()).sum() / cluster_size
                for g in groups
            ])
            
            # KL divergence
            kl = (p_k * torch.log((p_k + 1e-8) / (p_star + 1e-8))).sum()
            fairness_loss += kl
        
        return fairness_loss / self.n_clusters
    
    def compute_cluster_separation_loss(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Encourage cluster separation in embedding space
        """
        # Compute cluster centroids
        centroids = []
        for k in range(self.n_clusters):
            weights = q[:, k].unsqueeze(1)
            centroid = (z * weights).sum(dim=0) / (weights.sum() + 1e-8)
            centroids.append(centroid)
        centroids = torch.stack(centroids)  # (K, D)
        
        # Inter-cluster distance (maximize)
        centroid_sim = F.cosine_similarity(
            centroids.unsqueeze(1),
            centroids.unsqueeze(0),
            dim=2
        )
        
        # Remove diagonal
        mask = 1 - torch.eye(self.n_clusters, device=z.device)
        inter_sim = (centroid_sim * mask).sum() / (self.n_clusters * (self.n_clusters - 1))
        
        # Maximize separation = minimize similarity
        return inter_sim
    
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
        self._S = S                     # FairGT takes S as an input: keep it for inference as well
        z, q = self.model(X, edge_index, S)
        
        # Losses
        # Replace entropy loss with DEC KL
        p = (q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8))
        p = (p / (p.sum(dim=1, keepdim=True) + 1e-8)).detach()
        loss_cluster = F.kl_div((q + 1e-8).log(), p, reduction='batchmean')
        
        #loss_cluster = -(q * torch.log(q + 1e-8)).sum(dim=1).mean()
        loss_fair = self.compute_fairness_loss(q, S)
        loss_sep = self.compute_cluster_separation_loss(z, q)
        
        total_loss = (
            self.lambda_cluster * loss_cluster +
            self.lambda_fair * loss_fair +
            self.lambda_separation * loss_sep
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'loss_cluster': loss_cluster.item(),
            'loss_fair': loss_fair.item(),
            'loss_separation': loss_sep.item()
        }
    
    def predict(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        S: torch.Tensor = None,
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
            S = S if S is not None else getattr(self, "_S", None)
            _, q = self.model(X, edge_index, S)
            hard_assignments = q.argmax(dim=1)
        
        return hard_assignments.cpu().numpy()

    @torch.no_grad()
    def embed(self, A, X, **kwargs):
        """Encoder output z (input of the clustering head), with the same S input as in training."""
        self.model.eval()
        edge_index, _ = self._extract_edge_info(A)
        z, _ = self.model(X, edge_index, getattr(self, "_S", None))
        return z

