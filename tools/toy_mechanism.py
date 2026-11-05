"""Toy mechanism behind the FairPGT teaser (illustrative, not FairPGT output).

One attention layer over one partition (the whole toy graph):
  h_i  = [ structural encoding U_i  ||  group-correlated features ]
  A    = softmax(h h^T / T)                    (vanilla attention)
  A*   = CGAR target applied until nCGAR in [1-eps, 1+eps]
         (the fixed point that the alignment loss L_att pulls attention toward)
  z    = A h                                   (aggregated representation)
  clusters = k-means(z), K = 3
Assignment-level-only fairness: vanilla z, but cluster assignment constrained to
match the global group proportions (proportional quotas, optimal transport-style
assignment by the Hungarian method)."""
import json, numpy as np, networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans

def load(path):
    d = json.load(open(path))
    G = nx.Graph(); G.add_nodes_from(range(len(d['pos']))); G.add_edges_from(d['edges'])
    comm_names = {'orange': 0, 'blue': 1, 'red': 2}
    return d, G, np.array(d['group']), np.array([comm_names[c] for c in d['clusters']['balance0']])

def features(G, s, r=4, g=1.0, noise=0.15, seed=0):
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    L = laplacian(A, normed=True); w, V = np.linalg.eigh(L)
    U = V[:, 1:r + 1] / np.abs(V[:, 1:r + 1]).max(0)            # structural encoding
    rng = np.random.default_rng(seed)
    F = g * np.eye(2)[s] + noise * rng.standard_normal((len(s), 2))   # group-correlated features
    return np.hstack([U, F])

def softmax_rows(M):
    M = M - M.max(1, keepdims=True); E = np.exp(M); return E / E.sum(1, keepdims=True)

def ncgar(A, s):
    cross = s[:, None] != s[None, :]
    pi = np.bincount(s) / len(s); tau = (1 - (pi ** 2).sum()) / (pi ** 2).sum()
    return (A[cross].sum() / A[~cross].sum()) / tau

def cgar_fixed_point(A, s, alpha=0.05, eps=0.05, max_steps=5000):
    cross = s[:, None] != s[None, :]; A = A.copy()
    for _ in range(max_steps):
        r = ncgar(A, s)
        if 1 - eps <= r <= 1 + eps: break
        boost = cross if r < 1 - eps else ~cross
        A = np.where(boost, A * (1 + alpha), A); A /= A.sum(1, keepdims=True)
    return A

def kmeans(z, K=3, seed=0):
    return KMeans(K, n_init=50, random_state=seed).fit(z)

def proportional_assignment(z, s, km):
    """Assign nodes to the k-means centres under per-cluster group quotas."""
    K, n = km.n_clusters, len(s); sizes = np.bincount(km.labels_, minlength=K)
    lab = -np.ones(n, int); pi = np.bincount(s) / n
    for gidx in np.unique(s):
        idx = np.where(s == gidx)[0]
        q = np.floor(sizes * pi[gidx]).astype(int)
        while q.sum() < len(idx): q[np.argmax(sizes * pi[gidx] - q)] += 1
        slots = np.repeat(np.arange(K), q)
        cost = ((z[idx][:, None, :] - km.cluster_centers_[slots][None]) ** 2).sum(-1)
        r_, c_ = linear_sum_assignment(cost); lab[idx[r_]] = slots[c_]
    return lab

def summary(G, s, lab, comm):
    from sklearn.metrics import adjusted_rand_score
    parts = [set(np.where(lab == k)[0]) for k in np.unique(lab)]
    Q = nx.community.modularity(G, parts)
    p = s.mean(); dsp = max(abs(s[lab == k].mean() - p) for k in np.unique(lab))
    ratios = {int(k): f"{(s[lab==k]==1).sum()}:{(s[lab==k]==0).sum()}" for k in np.unique(lab)}
    return dict(Q=Q, dSP=dsp, ARI_comm=adjusted_rand_score(comm, lab), ratios=ratios)

if __name__ == '__main__':
    import argparse, os
    ap = argparse.ArgumentParser(description='Parameter scan of the toy mechanism.')
    ap.add_argument('graph', nargs='?',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toy_graph.json'),
                    help='path to toy_graph.json (default: next to this script)')
    d, G, s, comm = load(ap.parse_args().graph)
    for g in (0.6, 1.0, 1.5):
        for T in (0.3, 0.5, 1.0):
            h = features(G, s, g=g)
            A0 = softmax_rows(h @ h.T / T); A1 = cgar_fixed_point(A0, s)
            z0, z1 = A0 @ h, A1 @ h
            km0, km1 = kmeans(z0), kmeans(z1)
            l0, l1 = km0.labels_, km1.labels_; l2 = proportional_assignment(z0, s, km0)
            out = [summary(G, s, l, comm) for l in (l0, l2, l1)]
            print(f"g={g} T={T} nCGAR {ncgar(A0,s):.2f}->{ncgar(A1,s):.2f} | " + " | ".join(
                f"Q={o['Q']:.2f} dSP={o['dSP']:.2f} ARI={o['ARI_comm']:.2f} {list(o['ratios'].values())}" for o in out))
