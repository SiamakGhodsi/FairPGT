"""Losses of FairPGT.

All terms are computed exactly from the model's own outputs.

* L_att    nCGAR target + KL self-distillation of the partition attention (Eqs. 4-6)
* L_grp    I(C; S): soft cluster vs. protected group (= statistical parity, Eq. 11)
* L_info   mean_i H(C_i) - H(mean_i C_i): confident and non-degenerate clusters (Eq. 12)
* graph-aware target P (held fixed, refreshed every T_P steps, Eqs. 8-10)
* diagnostics: nCGAR of the trained attention, and I(group; routing token) for the
  routing branch
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-10


# ----------------------------------------------------------------- mutual information
def mi_from_joint(J: torch.Tensor) -> torch.Tensor:
    """Mutual information of a joint distribution J[..., a, b] (sums to 1 over a, b)."""
    pa = J.sum(-1, keepdim=True)
    pb = J.sum(-2, keepdim=True)
    return (J * (torch.log(J + EPS) - torch.log(pa + EPS) - torch.log(pb + EPS))).sum((-1, -2))


def eligible_queries(E: torch.Tensor) -> torch.Tensor:
    """Mask of query rows whose group has at least two members in the same set.

    E: (..., n, G) one-hot (zero rows for padding). A group with a single member
    cannot attend to its own group once self-attention is excluded, so it is left out
    of the query side (it stays a valid key).
    """
    counts = E.sum(-2, keepdim=True)                       # (..., 1, G)
    ok = (counts >= 2).to(E.dtype)
    return (E * ok).sum(-1)                                # (..., n) in {0, 1}


def routing_group_mi(B: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Diagnostic: I(group; routing token) of the node->token attention B: (H, N, P), averaged over heads."""
    q = eligible_queries(E)
    Eq = E * q.unsqueeze(-1)
    J = torch.einsum("ng,hnt->hgt", Eq, B) / q.sum().clamp_min(1.0)
    return mi_from_joint(J).mean()


def ncgar_selfdistill(A: torch.Tensor, E: torch.Tensor, alpha: float, eps_band: float) -> torch.Tensor:
    """L_att: nCGAR target (Eq. 5, held fixed) and KL self-distillation of the attention (Eq. 6)."""
    cross = 1.0 - torch.einsum("png,pmg->pnm", E, E)                      # (P, n, n)
    valid = torch.einsum("pn,pm->pnm", (E.sum(-1) > 0).float(), (E.sum(-1) > 0).float())
    valid = valid * (1 - torch.eye(A.shape[-1], device=A.device))
    cross = cross * valid
    within = valid - cross
    n_g = E.sum(1)                                                        # (P, G)
    n_p = n_g.sum(-1)
    pairs_within = (n_g * (n_g - 1)).sum(-1)
    pairs_cross = n_p * (n_p - 1) - pairs_within
    tau = pairs_cross / pairs_within.clamp_min(1.0)                       # (P,)
    mass_x = (A * cross[:, None]).sum((-1, -2))
    mass_w = (A * within[:, None]).sum((-1, -2))
    ncgar = mass_x / mass_w.clamp_min(EPS) / tau[:, None].clamp_min(EPS)  # (P, H)
    ok = ((n_g > 0).sum(-1) >= 2)[:, None] & (pairs_within > 0)[:, None]
    low = (ncgar < 1 - eps_band) & ok
    high = (ncgar > 1 + eps_band) & ok
    with torch.no_grad():
        boost = torch.where(low[..., None, None], cross[:, None], torch.where(high[..., None, None], within[:, None],
                                                                              torch.zeros_like(cross[:, None])))
        T = A * (1 + alpha * boost)
        T = T / T.sum(-1, keepdim=True).clamp_min(EPS)
    rows = (A.sum(-1) > 0).float()
    kl = (T * (torch.log(T + EPS) - torch.log(A + EPS))).sum(-1) * rows
    return kl.sum() / rows.sum().clamp_min(1.0)


def ncgar_value(A: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Mean nCGAR over partitions with >= 2 groups (diagnostic)."""
    with torch.no_grad():
        cross = 1.0 - torch.einsum("png,pmg->pnm", E, E)
        valid = torch.einsum("pn,pm->pnm", (E.sum(-1) > 0).float(), (E.sum(-1) > 0).float())
        valid = valid * (1 - torch.eye(A.shape[-1], device=A.device))
        cross = cross * valid
        within = valid - cross
        n_g = E.sum(1)
        n_p = n_g.sum(-1)
        pw = (n_g * (n_g - 1)).sum(-1)
        tau = (n_p * (n_p - 1) - pw) / pw.clamp_min(1.0)
        r = (A * cross[:, None]).sum((-1, -2)) / (A * within[:, None]).sum((-1, -2)).clamp_min(EPS)
        ok = ((n_g > 0).sum(-1) >= 2) & (pw > 0)
        val = (r / tau[:, None].clamp_min(EPS)).mean(1)
        return val[ok].mean() if ok.any() else torch.tensor(float("nan"))


# ----------------------------------------------------------------- assignment level
def assignment_parity_mi(C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """I(K; S) under soft assignments; zero iff every cluster has the global group shares."""
    J = C.t() @ E / C.shape[0]                              # (K, G)
    return mi_from_joint(J)


def information_term(C: torch.Tensor) -> torch.Tensor:
    """mean_i H(C_i) - H(c_bar): 0 at uniform assignments, -log K at confident balanced ones."""
    h_rows = -(C * torch.log(C + EPS)).sum(1).mean()
    c_bar = C.mean(0)
    return h_rows + (c_bar * torch.log(c_bar + EPS)).sum()


def collapse_term(C: torch.Tensor) -> torch.Tensor:
    """DMoN collapse regularizer (ablation: replaces L_info)."""
    K = C.shape[1]
    return K ** 0.5 * torch.linalg.vector_norm(C.mean(0)) - 1.0


@torch.no_grad()
def graph_aware_target(C: torch.Tensor, edge_index: torch.Tensor, use_graph: bool = True) -> torch.Tensor:
    """P_ik ∝ (C_ik^2 / f_k) exp(tanh(M_ik - mean_k M_ik)); plain DEC target if use_graph=False."""
    f = C.sum(0, keepdim=True)
    P = C ** 2 / f.clamp_min(EPS)
    if use_graph:
        n, K = C.shape
        src, dst = edge_index
        deg = torch.zeros(n, device=C.device).index_add_(0, src, torch.ones_like(src, dtype=C.dtype))
        two_m = deg.sum().clamp_min(1.0)
        neigh = torch.zeros_like(C).index_add_(0, src, C[dst])
        M = neigh - deg[:, None] * (deg[:, None] * C).sum(0, keepdim=True) / two_m
        P = P * torch.exp(torch.tanh(M - M.mean(1, keepdim=True)))
    return P / P.sum(1, keepdim=True).clamp_min(EPS)


def kl_target(P: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return (P * (torch.log(P + EPS) - torch.log(C + EPS))).sum(1).mean()


def soft_modularity(C: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Differentiable modularity of soft assignments (DMoN form), O(|E| K)."""
    n, K = C.shape
    src, dst = edge_index
    deg = torch.zeros(n, device=C.device).index_add_(0, src, torch.ones_like(src, dtype=C.dtype))
    two_m = deg.sum().clamp_min(1.0)
    inside = (C[src] * C[dst]).sum() / two_m
    expected = ((deg[:, None] * C).sum(0) ** 2).sum() / two_m ** 2
    return inside - expected
