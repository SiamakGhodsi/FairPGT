"""Frozen evaluation: one function computes every reported number from saved predictions.

Clustering quality (labels used here only): NMI, ARI, ACC and macro-F1 under the optimal
one-to-one matching of clusters to classes (Hungarian; clusters left unmatched count as
errors). Structure: modularity and mean conductance (sparse). Group fairness on hard
clusters: Delta_SP = max_{k,g} |p_k(g) - p(g)|, balance (mean over clusters of
min_g n_kg / max_g n_kg, as in DFNMF), and I(cluster; S) in nats. Degeneracy flags:
number of non-empty clusters and the share of the largest cluster.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def _contingency(a, b):
    M = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(M, (a, b), 1)
    return M


def matched_scores(y, c):
    M = _contingency(y, c)                                  # classes x clusters
    r, k = linear_sum_assignment(-M)
    mapping = np.full(M.shape[1], -1)
    mapping[k] = r
    pred = mapping[c]
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, labels=np.arange(M.shape[0]), average="macro", zero_division=0))
    return acc, f1


def modularity(A: sp.csr_matrix, c: np.ndarray) -> float:
    two_m = A.sum()
    deg = np.asarray(A.sum(1)).ravel()
    H = sp.csr_matrix((np.ones(len(c)), (np.arange(len(c)), c)), shape=(len(c), c.max() + 1))
    inside = (H.T @ A @ H).diagonal()
    vol = H.T @ deg
    return float(inside.sum() / two_m - (vol ** 2).sum() / two_m ** 2)


def conductance(A: sp.csr_matrix, c: np.ndarray) -> float:
    deg = np.asarray(A.sum(1)).ravel()
    H = sp.csr_matrix((np.ones(len(c)), (np.arange(len(c)), c)), shape=(len(c), c.max() + 1))
    inside = (H.T @ A @ H).diagonal()
    vol = H.T @ deg
    total = deg.sum()
    out = []
    for k in np.unique(c):
        denom = min(vol[k], total - vol[k])
        if denom > 0:
            out.append((vol[k] - inside[k]) / denom)
    return float(np.mean(out)) if out else 1.0


def group_fairness(c: np.ndarray, s: np.ndarray) -> dict:
    M = _contingency(c, s).astype(float)                    # clusters x groups
    M = M[M.sum(1) > 0]
    p = np.bincount(s) / len(s)
    pk = M / M.sum(1, keepdims=True)
    dsp = float(np.abs(pk - p).max())
    balance = float(np.mean(M.min(1) / np.maximum(M.max(1), 1)))
    J = M / M.sum()
    mi = float(np.sum(J * (np.log(J + 1e-12) - np.log(J.sum(1, keepdims=True) + 1e-12)
                             - np.log(J.sum(0, keepdims=True) + 1e-12))))
    return {"delta_sp": dsp, "balance": balance, "mi_cluster_s": mi}


def evaluate(labels: np.ndarray, A: sp.csr_matrix, S: np.ndarray, Y: np.ndarray | None, K: int) -> dict:
    c = np.unique(np.asarray(labels).astype(np.int64), return_inverse=True)[1]
    out = {"n_clusters_used": int(c.max() + 1), "largest_cluster": float(np.bincount(c).max() / len(c)),
           "modularity": modularity(A, c), "conductance": conductance(A, c), **group_fairness(c, S)}
    sizes = np.bincount(c, minlength=K) / len(c)                       # K clusters, empty ones count as 0
    nz = sizes[sizes > 0]
    out["size_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(K))  # 1 = equal sizes
    out["size_min_max"] = float(sizes.min() / sizes.max())            # 0 if a cluster is empty
    out["collapsed"] = bool(out["n_clusters_used"] < K or out["largest_cluster"] > 0.9)
    if Y is not None:
        out["nmi"] = float(normalized_mutual_info_score(Y, c))
        out["ari"] = float(adjusted_rand_score(Y, c))
        out["acc"], out["f1_macro"] = matched_scores(Y, c)
    return out
