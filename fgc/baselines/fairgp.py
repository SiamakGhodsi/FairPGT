"""
FairGP: Fair Graph Partitioning with Shifted Window Attention
Reference: AAAI 2025
KEY MECHANISM: Eliminates inter-cluster attention ENTIRELY (sets to 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fgc.baselines.base import BaseExperiment
import numpy as np
from typing import Dict, Tuple

try:
    from fgc.utils import metis_compat as metis
    HAS_METIS = True
except ImportError:
    HAS_METIS = False
    print("Warning: METIS not available. Using random partitioning.")


class FairPartitionedAttention(nn.Module):
    """
    CRITICAL: Partitioned attention with INTER-CLUSTER MASKING
    This is the KEY mechanism that differentiates FairGP from generic partitioned attention
    """
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, cluster_assignments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, D)
            cluster_assignments: Cluster ID for each node (N,)
        Returns:
            Attended features with MASKED inter-cluster attention (N, D)
        """
        N, D = x.shape
        out = torch.zeros_like(x)

        for c in torch.unique(cluster_assignments):
            mask = (cluster_assignments == c)
            idx = mask.nonzero(as_tuple=True)[0]
            nc = len(idx)
            if nc == 0:
                continue

        # Multi-head QKV
        x_c = x[idx]
        qkv = self.qkv(x_c).reshape(nc, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (H, nc, nc)
        attn = F.softmax(attn, dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(nc, D)
        out[idx] = self.proj(o)

        return out


class FairGPEncoder(nn.Module):
    """Fair Graph Partitioning Encoder with GCN + Fair Partitioned Attention"""
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_layers: int = 2
    ):
        super().__init__()

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(n_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])

        # Fair partitioned attention layers
        self.attn_layers = nn.ModuleList([
            FairPartitionedAttention(hidden_dim, n_heads)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, x, edge_index, cluster_assignments, edge_weight=None):
        """
        Args:
            cluster_assignments: Cluster ID for each node (N,) - REQUIRED for masking
        """
        for gcn, attn, ln in zip(self.gcn_layers, self.attn_layers, self.layer_norm):
            # GCN
            h = gcn(x, edge_index, edge_weight)
            h = F.relu(h)

            # Fair partitioned attention with inter-cluster masking
            h = h + attn(h, cluster_assignments)

            # Layer norm
            x = ln(h)

        return x


class FairGP(nn.Module):
    """Fair Graph Partitioning with Inter-Cluster Attention Elimination"""
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_partitions: int = 10
    ):
        super().__init__()

        self.n_partitions = n_partitions
        self.encoder = FairGPEncoder(n_features, hidden_dim, n_heads)
        self.cluster_head = nn.Linear(hidden_dim, n_clusters)

    def partition_graph(self, edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """
        Partition graph using METIS
        Returns cluster assignments for each node
        """
        if not HAS_METIS:
            # Fallback: random partitioning
            return torch.randint(0, self.n_partitions, (n_nodes,), device=edge_index.device)

        # Convert to adjacency list for METIS
        adj_list = [[] for _ in range(n_nodes)]
        edge_index_cpu = edge_index.cpu()
        for i in range(edge_index_cpu.shape[1]):
            src, dst = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
            if src != dst:
                adj_list[src].append(dst)

        try:
            # METIS partitioning
            _, parts = metis.part_graph(adj_list, nparts=self.n_partitions)
            return torch.tensor(parts, device=edge_index.device)
        except:
            # Fallback if METIS fails
            return torch.randint(0, self.n_partitions, (n_nodes,), device=edge_index.device)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Returns:
            z: Embeddings
            q: Soft cluster assignments
        """
        # Get cluster assignments for attention masking (computed once and cached;
        # the graph does not change during training)
        if getattr(self, '_parts', None) is None or self._parts.shape[0] != x.shape[0]:
            self._parts = self.partition_graph(edge_index, x.shape[0])
        cluster_assignments = self._parts

        # Encode with fair attention
        z = self.encoder(x, edge_index, cluster_assignments, edge_weight)

        # Cluster assignments
        logits = self.cluster_head(z)
        q = F.softmax(logits, dim=1)
        return z, q

    def cluster(self, x, edge_index, edge_weight=None):
        """Get hard cluster assignments"""
        with torch.no_grad():
            _, q = self.forward(x, edge_index, edge_weight)
            return q.argmax(dim=1)


class FairGPExperiment(BaseExperiment):
    """FairGP Experiment Wrapper"""

    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_partitions: int = 10,
        lr: float = 0.001,
        lambda_fair: float = 0.1,
        lambda_cluster: float = 1.0,
        **kwargs
    ):
        super().__init__(
            model_name='FairGP',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_partitions = n_partitions
        self.lr = lr
        self.lambda_fair = lambda_fair
        self.lambda_cluster = lambda_cluster

    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        """Build FairGP model"""
        model = FairGP(
            n_features=n_features,
            n_clusters=self.n_clusters,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_partitions=self.n_partitions
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )

        return model

    def compute_fairness_loss(
        self,
        q: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """
        Group fairness loss: Statistical Parity
        Minimize deviation from global group proportions
        """
        n = len(q)
        groups = torch.unique(S)

        # Global proportions
        p_star = torch.stack([
            (S == g).float().sum() / n for g in groups
        ])

        # Per-cluster proportions
        fairness_loss = 0.0
        for k in range(self.n_clusters):
            # Soft cluster membership
            cluster_weight = q[:, k]
            cluster_size = cluster_weight.sum() + 1e-8

            # Group proportions in this cluster
            p_k = torch.stack([
                (cluster_weight * (S == g).float()).sum() / cluster_size
                for g in groups
            ])

            # Statistical parity: minimize |p_k - p*|
            fairness_loss += ((p_k - p_star) ** 2).sum()

        return fairness_loss / self.n_clusters

    def compute_cluster_loss(self, q):
        """DEC-style self-training (replaces broken entropy loss)"""
        weight = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
        p = (weight / (weight.sum(dim=1, keepdim=True) + 1e-8)).detach()
        return F.kl_div((q + 1e-8).log(), p, reduction='batchmean')

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
        loss_cluster = self.compute_cluster_loss(q)
        loss_fair = self.compute_fairness_loss(q, S)

        total_loss = (
            self.lambda_cluster * loss_cluster +
            self.lambda_fair * loss_fair
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'loss_cluster': loss_cluster.item(),
            'loss_fair': loss_fair.item()
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

    @torch.no_grad()
    def embed(self, A, X, **kwargs):
        """Encoder output z (input of the clustering head)."""
        self.model.eval()
        edge_index, edge_weight = self._extract_edge_info(A)
        z, _ = self.model(X, edge_index, edge_weight)
        return z
