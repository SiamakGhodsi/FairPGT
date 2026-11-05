"""Checks of the claims the paper makes about the losses, and of the evaluation code.

    python -m pytest -q tests
"""
import math

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from fgc.eval.metrics import evaluate, matched_scores, modularity
from fgc.models import losses as Lf
from fgc.models.fairpgt import FairPGT, Partitioner

torch.manual_seed(0)


def one_hot(s, G=None):
    s = torch.as_tensor(s)
    return torch.nn.functional.one_hot(s, G or int(s.max()) + 1).float()


def rand_attention(n, H=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(1, H, n, n, generator=g) * (1 - torch.eye(n))
    return A / A.sum(-1, keepdim=True)


# ------------------------------------------------------------------ assignment level
def test_uniform_assignment_is_not_a_minimizer():
    n, K = 40, 4
    s = torch.randint(0, 2, (n,))
    E = one_hot(s, 2)
    U = torch.full((n, K), 1.0 / K)
    ei = torch.randint(0, n, (2, 200))
    ei = torch.cat([ei, ei.flip(0)], 1)
    P = Lf.graph_aware_target(U, ei)
    assert torch.allclose(P, U, atol=1e-6)
    assert abs(Lf.kl_target(P, U).item()) < 1e-6
    assert abs(Lf.assignment_parity_mi(U, E).item()) < 1e-6
    assert abs(Lf.information_term(U).item()) < 1e-6
    hard = torch.eye(K)[torch.arange(n) % K]
    assert Lf.information_term(hard).item() < -math.log(K) + 1e-4


def test_assignment_mi_is_zero_iff_statistical_parity():
    s = torch.tensor([0, 0, 1, 1])
    E = one_hot(s)
    fair = torch.tensor([[1., 0], [0, 1], [1, 0], [0, 1]])
    unfair = torch.tensor([[1., 0], [1, 0], [0, 1], [0, 1]])
    assert Lf.assignment_parity_mi(fair, E).item() < 1e-6
    assert abs(Lf.assignment_parity_mi(unfair, E).item() - math.log(2)) < 1e-5


# ------------------------------------------------------------------ model plumbing
def test_partitioner_and_attention_masks():
    parts = torch.tensor([0, 1, 0, 2, 1, 0, 2, 2, 2])
    part = Partitioner(parts)
    h = torch.randn(9, 8)
    assert torch.allclose(part.scatter(part.gather(h), 9), h)
    model = FairPGT(8, 3, hidden=16, heads=2, layers=2, proj_dim=4, dropout=0.0)
    model.eval()
    z, C, attn = model(h, part, return_attn=True)
    A, B = attn[0]
    rows = A.sum(-1)[part.valid[:, None].expand(-1, 2, -1)]
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
    assert torch.diagonal(A, dim1=-2, dim2=-1).abs().max() == 0
    assert torch.allclose(C.sum(1), torch.ones(9), atol=1e-5)
    assert B.shape == (2, 9, 3)


def test_singleton_partition_gets_no_local_message():
    parts = torch.tensor([0, 0, 0, 1])
    part = Partitioner(parts)
    model = FairPGT(4, 2, hidden=8, heads=2, layers=1, proj_dim=4, dropout=0.0).eval()
    _, _, attn = model(torch.randn(4, 4), part, return_attn=True)
    A = attn[0][0]
    assert A[1].abs().sum() == 0


# ------------------------------------------------------------------ evaluation
def test_metrics_against_reference():
    G = nx.karate_club_graph()
    A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=range(34)))
    A.data[:] = 1
    c = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(34)])
    ref = nx.community.modularity(G, [set(np.where(c == k)[0]) for k in range(2)], weight=None)
    assert abs(modularity(A, c) - ref) < 1e-9
    y = c.copy()
    acc, f1 = matched_scores(y, 1 - c)                  # label permutation is irrelevant
    assert acc == 1.0 and f1 == 1.0
    s = np.arange(34) % 2
    m = evaluate(c, A, s, y, 2)
    assert 0 <= m["delta_sp"] <= 1 and m["nmi"] == pytest.approx(1.0)


def test_sfairsc_satisfies_constraint():
    from fgc.baselines.fair_spectral import sfairsc
    rng = np.random.default_rng(0)
    n = 120
    s = rng.integers(0, 2, n)
    comm = np.repeat([0, 1, 2], 40)
    P = np.where(comm[:, None] == comm[None], 0.3, 0.02) * np.where(s[:, None] == s[None], 2.0, 1.0)
    A = sp.csr_matrix(np.triu(rng.random((n, n)) < P, 1).astype(float))
    A = A + A.T
    lab = sfairsc(A, s, 3)
    m = evaluate(lab, A, s, comm, 3)
    assert m["delta_sp"] < 0.25
