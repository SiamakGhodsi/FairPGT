import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans
import scipy.sparse as sp
from fgc.baselines.base import BaseExperiment

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class FairGCZhangModel(nn.Module):
    def __init__(self, in_dim, num_nodes, hidden_dim=256, embed_dim=128):
        super().__init__()

        # attribute view
        self.attr_enc1 = MLP(in_dim, hidden_dim, embed_dim)
        self.attr_enc2 = MLP(in_dim, hidden_dim, embed_dim)

        # structure view
        self.struct_enc1 = MLP(num_nodes, hidden_dim, embed_dim)
        self.struct_enc2 = MLP(num_nodes, hidden_dim, embed_dim)

        # cross-view
        self.g12 = nn.Linear(embed_dim, embed_dim)
        self.g21 = nn.Linear(embed_dim, embed_dim)

        # discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, X, A):
        # attribute
        Z1 = self.attr_enc1(X)
        Z2 = self.attr_enc2(X)

        # structure
        E1 = self.struct_enc1(A)
        E2 = self.struct_enc2(A)

        Z1 = F.normalize(Z1 + E1, dim=1)
        Z2 = F.normalize(Z2 + E2, dim=1)

        Z_1_to_2 = self.g12(Z1)
        Z_2_to_1 = self.g21(Z2)

        Z = 0.5 * (Z1 + Z2)

        return Z, Z1, Z2, Z_1_to_2, Z_2_to_1

class FairGCZhangLoss(nn.Module):
    def __init__(self, beta=0.1, gamma=0.5, tau=0.5,
                 rw_steps=3, alpha=0.5):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.rw_steps = rw_steps
        self.alpha = alpha

    def forward(self, model, outputs, S):
        Z, Z1, Z2, Z_1_to_2, Z_2_to_1 = outputs
        N = Z.size(0)

        # ---------------------------
        # 1. RANDOM WALK TARGET (CORE)
        # ---------------------------
        R = compute_affinity(Z1, Z2)
        T = random_walk_matrix(R, self.rw_steps, self.alpha)

        # normalize target rows
        T = T / (T.sum(dim=1, keepdim=True) + 1e-8)

        # ---------------------------
        # 2. CROSS-VIEW DISTRIBUTIONS
        # ---------------------------
        sim_12 = torch.mm(Z_2_to_1, Z1.t()) / self.tau
        sim_21 = torch.mm(Z_1_to_2, Z2.t()) / self.tau

        P12 = F.softmax(sim_12, dim=1)
        P21 = F.softmax(sim_21, dim=1)

        # ---------------------------
        # 3. INDIVIDUAL FAIRNESS LOSS
        # ---------------------------
        L_I = F.kl_div(P12.log(), T, reduction='batchmean') + \
              F.kl_div(P21.log(), T, reduction='batchmean')

        # ---------------------------
        # 4. GROUP FAIRNESS (ADVERSARIAL)
        # ---------------------------
        logits = model.discriminator(Z.detach())
        L_G = F.cross_entropy(logits, S)

        # ---------------------------
        # 5. COVARIANCE LOSS
        # ---------------------------
        S_centered = S.float() - S.float().mean()
        Z_centered = Z - Z.mean(dim=0)

        cov = torch.matmul(S_centered.unsqueeze(1).t(), Z_centered) / N
        L_R = cov.abs().mean()

        # ---------------------------
        # FINAL LOSS
        # ---------------------------
        total = L_I - self.beta * L_G + self.gamma * L_R

        return total, {
            'L_I': L_I.item(),
            'L_G': L_G.item(),
            'L_R': L_R.item(),
            'total': total.item()
        }


class FairGCZhangExperiment(BaseExperiment):
    def __init__(
        self,
        dataset_name: str,
        n_clusters: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        lr: float = 0.001,
        beta: float = 0.1,
        gamma: float = 0.5,
        tau: float = 0.5,
        rw_steps: int = 3,
        alpha: float = 0.5,
        **kwargs
    ):
        super().__init__(
            model_name='fairgc_zhang',
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            **kwargs
        )

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.rw_steps = rw_steps
        self.alpha = alpha

    def build_model(self, n_features: int, **kwargs) -> nn.Module:
        n_nodes = kwargs.get("n_nodes")
        if n_nodes is None:
            raise ValueError("n_nodes must be provided to build_model for fairgc_zhang")

        model = FairGCZhangModel(
            in_dim=n_features,
            num_nodes=n_nodes,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim
        )

        self.criterion = FairGCZhangLoss(
            beta=self.beta,
            gamma=self.gamma,
            tau=self.tau,
            rw_steps=self.rw_steps,
            alpha=self.alpha
        ).to(self.device)

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
    ):
        self.model.train()

        # A, X, S are already tensors on the right device by BaseExperiment
        if A.is_sparse:
            A_dense = A.to_dense()
        else:
            A_dense = A.float()

        X = X.float()
        S = S.long()

        outputs = self.model(X, A_dense)
        total_loss, loss_dict = self.criterion(self.model, outputs, S)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'L_I': loss_dict['L_I'],
            'L_G': loss_dict['L_G'],
            'L_R': loss_dict['L_R'],
        }

    def predict(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        **kwargs
    ) -> np.ndarray:
        self.model.eval()

        if A.is_sparse:
            A_dense = A.to_dense()
        else:
            A_dense = A.float()

        X = X.float()

        with torch.no_grad():
            Z, *_ = self.model(X, A_dense)

        return KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit_predict(
            Z.detach().cpu().numpy()
        )

    def fit(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        S: torch.Tensor,
        Y=None,
        n_epochs: int = 200,
        eval_freq: int = 10,
        patience: int = 20,
        **kwargs
    ):
        if self.model is None:
            n_nodes = A.shape[0] if hasattr(A, "shape") else A.size(0)
            n_feat = X.shape[1] if hasattr(X, "shape") else X.size(1)
            self.model = self.build_model(n_features=n_feat, n_nodes=n_nodes)
            self.model = self.model.to(self.device)

        return super().fit(A, X, S, Y, n_epochs, eval_freq, patience, **kwargs)


def normalize_adj(A):
    """Row-normalize adjacency matrix"""
    deg = A.sum(dim=1, keepdim=True) + 1e-8
    return A / deg


def compute_affinity(Z1, Z2, rho=0.5):
    """
    R = rho * (Z1 Z1^T) + (1-rho)*(Z2 Z2^T)
    """
    R1 = torch.mm(Z1, Z1.t())
    R2 = torch.mm(Z2, Z2.t())
    return rho * R1 + (1 - rho) * R2


def random_walk_matrix(R, steps=3, alpha=0.5):
    """
    T = alpha I + (1-alpha) R^t
    """
    R = F.relu(R)  # ensure non-negative
    R = normalize_adj(R)

    Rt = R
    for _ in range(steps - 1):
        Rt = torch.mm(Rt, R)

    N = R.size(0)
    I = torch.eye(N, device=R.device)

    T = alpha * I + (1 - alpha) * Rt
    return T