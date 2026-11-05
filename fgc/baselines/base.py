"""Common interface for all clustering methods (ported baselines and FairPGT).

Protocol (enforced by the runner, identical for every method):
  * labels Y are never passed to a method;
  * a fixed epoch budget; the reported model is the last epoch (``selection='last'``)
    unless a label-free rule is chosen (``selection='modularity'``: best graph
    modularity of the predicted clusters, checked every ``eval_freq`` epochs);
  * checkpoints are kept in memory; the runner saves the final weights and predictions.

Subclasses implement ``build_model``, ``train_epoch`` and ``predict`` (as in the
original benchmark), and may implement ``embed`` (node embeddings, for the S-probe).
"""
from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
import torch


def sparse_modularity(A: sp.csr_matrix, labels: np.ndarray) -> float:
    A = sp.csr_matrix(A)
    two_m = A.sum()
    if two_m == 0:
        return 0.0
    deg = np.asarray(A.sum(1)).ravel()
    K = labels.max() + 1
    H = sp.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), labels)), shape=(len(labels), K))
    inside = (H.T @ A @ H).diagonal().sum()
    vol = H.T @ deg
    return float(inside / two_m - (vol ** 2).sum() / two_m ** 2)


class BaseExperiment(ABC):
    def __init__(self, model_name, dataset_name, n_clusters, device="cuda", seed=42, **kwargs):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.n_clusters = n_clusters
        self.seed = seed
        self.kwargs = kwargs
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.logger = logging.getLogger(f"fgc.{model_name}")
        self.model = None
        self.optimizer = None
        self.epoch = 0
        self.best_metric = -np.inf
        self.history = []
        self._best_state = None

    # ---------------------------------------------------------------- to implement
    @abstractmethod
    def build_model(self, **kwargs):
        ...

    @abstractmethod
    def train_epoch(self, A, X, S, **kwargs) -> dict:
        ...

    @abstractmethod
    def predict(self, A, X, **kwargs) -> np.ndarray:
        ...

    def embed(self, A, X, **kwargs):
        return None

    # ---------------------------------------------------------------- training loop
    def fit(self, A, X, S, Y=None, n_epochs=200, eval_freq=10, patience=None, selection="last", **kwargs):
        if Y is not None:
            raise ValueError("labels must not be passed to fit()")
        A, X, S = self._to_device(A), self._to_device(X), self._to_device(S)
        if self.model is None:
            self.model = self.build_model(n_features=X.shape[1], n_nodes=X.shape[0], **self.kwargs)
            if isinstance(self.model, torch.nn.Module):
                self.model = self.model.to(self.device)
        A_eval = kwargs.get("graph").A if kwargs.get("graph") is not None else None
        stale = 0
        for epoch in range(n_epochs):
            self.epoch = epoch
            logs = self.train_epoch(A, X, S)
            self.history.append({"epoch": epoch, **{k: float(v) for k, v in (logs or {}).items()
                                                    if isinstance(v, (int, float, np.floating))}})
            if selection == "modularity" and A_eval is not None and (epoch % eval_freq == 0 or epoch == n_epochs - 1):
                self.model.eval()
                with torch.no_grad():
                    q = sparse_modularity(A_eval, np.asarray(self.predict(A, X)))
                if q > self.best_metric:
                    self.best_metric, stale = q, 0
                    self._best_state = copy.deepcopy(self.model.state_dict())
                else:
                    stale += 1
                if patience and stale >= patience:
                    break
        if selection == "modularity" and self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return {}

    # ---------------------------------------------------------------- helpers kept from the old base
    def evaluate(self, A, X, S, Y=None):
        raise RuntimeError("evaluation is done by fgc.eval.metrics after training")

    def _save_results(self, *_, **__):
        pass

    def _extract_edge_info(self, A):
        """Edge list of A, computed once per adjacency object (the graph is fixed)."""
        cache = getattr(self, "_edge_cache", None)
        if cache is not None and cache[0] is A:
            return cache[1], cache[2]
        ei, ew = self._edge_info_uncached(A)
        self._edge_cache = (A, ei, ew)
        return ei, ew

    def _edge_info_uncached(self, A):
        from torch_geometric.utils import from_scipy_sparse_matrix
        if sp.issparse(A):
            ei, ew = from_scipy_sparse_matrix(A)
            return ei.to(self.device), ew.float().to(self.device)
        if isinstance(A, torch.Tensor) and A.is_sparse:
            A = A.coalesce()
            return A.indices().to(self.device), A.values().float().to(self.device)
        ei = A.nonzero().t().to(self.device)
        return ei, A[ei[0], ei[1]].float().to(self.device)

    def _to_device(self, t):
        if t is None or sp.issparse(t):
            return t
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        return t.to(self.device)

    def _to_numpy(self, t):
        if t is None or sp.issparse(t):
            return t
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return t
