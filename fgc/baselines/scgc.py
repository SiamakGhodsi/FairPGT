"""
SCGC: Simple Contrastive Graph Clustering
Reference: https://github.com/yueliu1999/SCGC (TNNLS 2024)
KEY: Low-pass filter preprocessing + Only 2 MLPs (NO GCN!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict


class LowPassFilter(nn.Module):
    """
    Graph Laplacian low-pass filter (SCGC preprocessing):  X_s = (I - alpha * L)^t X,
    L = I - D^{-1/2} A D^{-1/2}. Computed with sparse products and cached: the graph and the
    input features are fixed, so the result is identical every epoch. (The earlier version
    built dense n x n matrices and dense diag products on every forward pass, O(n^3).)
    """
    def __init__(self, t: int = 2, alpha: float = 0.5):
        super().__init__()
        self.t = t
        self.alpha = alpha
        self._cache = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:
        key = (x.data_ptr(), x.shape, edge_index.data_ptr(), edge_index.shape)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        n = x.shape[0]
        w = torch.ones(edge_index.shape[1], device=x.device) if edge_weight is None else edge_weight.float()
        deg = torch.zeros(n, device=x.device).index_add_(0, edge_index[0], w)
        dis = torch.pow(deg + 1e-8, -0.5)
        A_hat = torch.sparse_coo_tensor(edge_index, dis[edge_index[0]] * w * dis[edge_index[1]], (n, n)).coalesce()
        xs = x
        for _ in range(self.t):                    # (I - aL) x = (1 - a) x + a A_hat x
            xs = (1 - self.alpha) * xs + self.alpha * torch.sparse.mm(A_hat, xs)
        self._cache = (key, xs)
        return xs


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP encoder"""
    def __init__(self, n_features: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.prelu(self.fc1(x))
        z = self.fc2(h)
        return F.normalize(z, dim=1)


class SCGC(nn.Module):
    """
    Simple Contrastive Graph Clustering
    CRITICAL: Uses ONLY MLPs, NO GCN!
    Structure incorporated via low-pass filtering + contrastive loss
    """
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        hidden_dim: int = 256,
        temperature: float = 0.5,
        filter_layers: int = 2
    ):
        super().__init__()

        # Low-pass filter (independent preprocessing)
        self.low_pass_filter = LowPassFilter(t=filter_layers)

        # Two parameter-UNSHARED MLP encoders (siamese)
        self.encoder1 = SimpleMLP(n_features, hidden_dim, hidden_dim)
        self.encoder2 = SimpleMLP(n_features, hidden_dim, hidden_dim)

        # Clustering head
        self.cluster_layer = nn.Linear(hidden_dim, n_clusters, bias=False)
        self.temperature = temperature

    def forward(self, x, edge_index, edge_weight=None):
        """
        Returns:
            z: L2-normalized embeddings
            q: Soft cluster assignments
        """
        # Low-pass filtering (preprocessing)
        x_smooth = self.low_pass_filter(x, edge_index, edge_weight)

        # Two views with parameter-unshared encoders
        z1 = self.encoder1(x_smooth)
        z2 = self.encoder2(x_smooth)

        # Fuse views (average)
        z = (z1 + z2) / 2

        # Cluster assignments
        logits = self.cluster_layer(z)
        q = F.softmax(logits / self.temperature, dim=1)

        return z, q, z1, z2

    def cluster(self, x, edge_index, edge_weight=None):
        """Get hard cluster assignments"""
        with torch.no_grad():
            z, q, _, _ = self.forward(x, edge_index, edge_weight)
            return q.argmax(dim=1)


class SCGCExperiment(BaseExperiment):
    """SCGC Experiment Wrapper"""

    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 256,
        lr: float = 0.0001,
        temperature: float = 0.5,
        filter_layers: int = 2,
        lambda_instance: float = 1.0,
        lambda_cluster: float = 1.0,
        **kwargs
    ):
        super().__init__(
            model_name='SCGC',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.temperature = temperature
        self.filter_layers = filter_layers
        self.lambda_instance = lambda_instance
        self.lambda_cluster = lambda_cluster

    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        """Build SCGC model"""
        model = SCGC(
            n_features=n_features,
            n_clusters=self.n_clusters,
            hidden_dim=self.hidden_dim,
            temperature=self.temperature,
            filter_layers=self.filter_layers
        )

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )

        return model

    def compute_instance_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Neighbor-oriented contrastive loss between two views
        Cross-view structural consistency
        """
        # Similarity matrices
        sim_z1_z2 = torch.mm(z1, z2.t()) / self.temperature
        sim_z2_z1 = torch.mm(z2, z1.t()) / self.temperature

        # Construct positive masks (top-k similar nodes)
        k = min(10, len(z1) // 10)

        with torch.no_grad():
            # Find k-nearest neighbors in each view
            _, topk_idx_z1 = torch.topk(torch.mm(z1, z1.t()), k=k+1, dim=1)
            _, topk_idx_z2 = torch.topk(torch.mm(z2, z2.t()), k=k+1, dim=1)

            # Create masks
            pos_mask_z1 = torch.zeros_like(sim_z1_z2)
            pos_mask_z2 = torch.zeros_like(sim_z2_z1)

            for i in range(len(z1)):
                pos_mask_z1[i, topk_idx_z1[i, 1:]] = 1  # exclude self
                pos_mask_z2[i, topk_idx_z2[i, 1:]] = 1

        # InfoNCE loss
        exp_sim_z1_z2 = torch.exp(sim_z1_z2)
        exp_sim_z2_z1 = torch.exp(sim_z2_z1)

        pos_sim_z1 = (exp_sim_z1_z2 * pos_mask_z1).sum(dim=1)
        all_sim_z1 = exp_sim_z1_z2.sum(dim=1)

        pos_sim_z2 = (exp_sim_z2_z1 * pos_mask_z2).sum(dim=1)
        all_sim_z2 = exp_sim_z2_z1.sum(dim=1)

        loss_z1 = -torch.log((pos_sim_z1 + 1e-8) / (all_sim_z1 + 1e-8))
        loss_z2 = -torch.log((pos_sim_z2 + 1e-8) / (all_sim_z2 + 1e-8))

        return (loss_z1.mean() + loss_z2.mean()) / 2

    def compute_cluster_contrastive_loss(
        self,
        z: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        Cluster-level contrastive loss
        Nodes should be similar to their assigned cluster centroid
        """
        # Get cluster assignments
        cluster_ids = q.argmax(dim=1)

        # Compute cluster centroids
        centroids = []
        for k in range(self.n_clusters):
            mask = (cluster_ids == k)
            if mask.sum() > 0:
                centroids.append(z[mask].mean(dim=0))
            else:
                centroids.append(torch.zeros_like(z[0]))
        centroids = torch.stack(centroids)
        centroids = F.normalize(centroids, dim=1)

        # Similarity to centroids
        sim_to_centroids = torch.mm(z, centroids.t()) / self.temperature

        # Loss: nodes should be similar to their assigned cluster
        loss = F.cross_entropy(sim_to_centroids, cluster_ids)

        return loss

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
        z, q, z1, z2 = self.model(X, edge_index, edge_weight)

        # Compute losses
        loss_instance = self.compute_instance_contrastive_loss(z1, z2)
        loss_cluster = self.compute_cluster_contrastive_loss(z, q)

        total_loss = (
            self.lambda_instance * loss_instance +
            self.lambda_cluster * loss_cluster
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'loss_instance': loss_instance.item(),
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
            _, q, _, _ = self.model(X, edge_index, edge_weight)
            hard_assignments = q.argmax(dim=1)

        return hard_assignments.cpu().numpy()