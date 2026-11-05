"""Fair spectral clustering baselines, re-implemented from the papers.


sFairSC  Group-fair normalized spectral clustering (Kleindessner et al., ICML 2019),
         computed scalably as in Wang et al. (AISTATS 2023): the constraint F^T H = 0
         with F = [1[s=g] - pi_g]_g is enforced by an n x (G-1) projector inside the
         eigensolver, so no null-space basis is formed. With H = D^{-1/2} U the problem
         min tr(H^T L H) s.t. H^T D H = I, F^T H = 0 becomes the top-K eigenvectors of
         Pi (I + D^{-1/2} A D^{-1/2}) Pi (A regularized, see below) on range(Pi), Pi = I - Ft (Ft^T Ft)^-1 Ft^T,
         Ft = D^{-1/2} F. k-means on the rows of H.

iFairSC  Individual-fair (representation-graph) spectral clustering, UREPSC
         (Gupta & Dukkipati, NeurIPS 2022): constraint R (I - 11^T/N) H = 0, solved
         with an orthonormal null-space basis Y, top-K eigenvectors of Y^T L Y, k-means
         on H = Y Z. When rank(R) > N - K the paper uses the best low-rank
         approximation of R ("UREPSC approx."). R must be supplied by the experimenter;
         here it is a symmetric kNN graph on the node features (``knn``) and its rank
         is cut to ``rank`` (default N - 2K) if needed. Dense O(N^3): small graphs only.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from fgc.baselines.base import BaseExperiment


def sfairsc(A: sp.csr_matrix, S: np.ndarray, K: int, seed: int = 0, return_embedding: bool = False):
    # Regularized spectral clustering (Qin & Rohe, NeurIPS 2013): A_tau = A + (tau/n) 11^T with
    # tau = mean degree. Without it, on graphs with many connected components (LastFM, Pokec)
    # the leading eigenvectors are component indicators and k-means returns one giant cluster.
    A = sp.csr_matrix(A, dtype=np.float64)
    n = A.shape[0]
    tau = A.sum() / n
    d = np.asarray(A.sum(1)).ravel() + tau
    dis = 1.0 / np.sqrt(d)
    An_sp = sp.diags(dis) @ A @ sp.diags(dis)

    class _An:                                                     # D^-1/2 (A + tau/n 11^T) D^-1/2
        def __matmul__(self, v):
            return An_sp @ v + dis * (tau / n) * (dis @ v)
    An = _An()
    G = S.max() + 1
    F = np.eye(G)[S][:, :G - 1] - np.bincount(S, minlength=G)[:G - 1] / n
    Ft = dis[:, None] * F
    Qf, _ = np.linalg.qr(Ft)                                   # orthonormal basis of the constraint

    def proj(v):
        return v - Qf @ (Qf.T @ v)

    def mv(v):
        v = proj(v.ravel())
        return proj(v + An @ v)

    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    v0 = proj(np.random.default_rng(seed).standard_normal(n))
    _, U = eigsh(op, k=K, which="LA", v0=v0, maxiter=20 * n)
    H = dis[:, None] * U
    labels = KMeans(K, n_init=10, random_state=seed).fit_predict(H)
    return (labels, H) if return_embedding else labels


def ifairsc(A: sp.csr_matrix, X: np.ndarray, K: int, knn: int = 10, rank: int | None = None,
            seed: int = 0) -> np.ndarray:
    n = A.shape[0]
    A = np.asarray(sp.csr_matrix(A).todense(), dtype=np.float64)
    nbr = NearestNeighbors(n_neighbors=knn + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    R = np.zeros((n, n))
    R[np.repeat(np.arange(n), knn), nbr.ravel()] = 1
    R = np.maximum(R, R.T)
    r = rank if rank is not None else n - 2 * K
    U_, s_, Vt = np.linalg.svd(R)
    if (s_ > 1e-10).sum() > n - K:                             # infeasible: low-rank approximation
        R = (U_[:, :r] * s_[:r]) @ Vt[:r]
    M = R - R.mean(1, keepdims=True)                           # R (I - 11^T / N)
    _, sv, Wt = np.linalg.svd(M)
    Y = Wt[(sv > 1e-8 * sv.max()).sum():].T                    # null-space basis
    if Y.shape[1] < K:
        raise RuntimeError(f"iFairSC infeasible: null space of dimension {Y.shape[1]} < K={K}")
    L = np.diag(A.sum(1)) - A
    w, Z = np.linalg.eigh(Y.T @ L @ Y)
    H = Y @ Z[:, np.argsort(w)[:K]]
    return KMeans(K, n_init=10, random_state=seed).fit_predict(H)


class _Spectral(BaseExperiment):
    def build_model(self, **kwargs):
        return None

    def train_epoch(self, A, X, S, **kwargs):
        return {}

    def predict(self, A, X, **kwargs):
        return self.labels


class sFairSCExperiment(_Spectral):
    def __init__(self, dataset_name, n_clusters, **kwargs):
        super().__init__("sfairsc", dataset_name, n_clusters, **kwargs)

    def fit(self, A, X, S, Y=None, graph=None, **kwargs):
        self.labels, self.H = sfairsc(graph.A, graph.S, self.n_clusters, self.seed, return_embedding=True)
        return {}

    def embed(self, A, X, **kwargs):
        return self.H                                          # fair spectral embedding (clustered by k-means)


class iFairSCExperiment(_Spectral):
    def __init__(self, dataset_name, n_clusters, knn=10, rank=None, **kwargs):
        super().__init__("ifairsc", dataset_name, n_clusters, **kwargs)
        self.knn, self.rank = knn, rank

    def fit(self, A, X, S, Y=None, graph=None, **kwargs):
        self.labels = ifairsc(graph.A, graph.X, self.n_clusters, self.knn, self.rank, self.seed)
        return {}
