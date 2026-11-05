"""Training wrapper for FairPGT, with the same interface as the baselines.

Objective
  L = KL(P || C) + lambda_grp * L_grp + lambda_par * L_att + lambda_info * L_info   (Eq. 13)
  lambda_par is lambda_att of the paper.
Variants used in the ablations are switched by the options below.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from fgc.baselines.base import BaseExperiment
from fgc.data.preprocess import metis_partition, spectral_encoding
from fgc.models import losses as Lf
from fgc.models.fairpgt import FairPGT, Partitioner

DEFAULTS = dict(
    hidden=256, heads=4, layers=3, proj_dim=32, dropout=0.1,
    n_partitions="auto", max_part_size=2500, n_spectral=8,
    lr=5e-4, weight_decay=1e-5, grad_clip=1.0,
    lambda_grp=10.0, lambda_par=1.0, lambda_info=1.0,
    attn_reg="ncgar",         # 'ncgar' (L_att, paper) | 'none'
    reg="info",               # 'info' (L_info, paper) | 'col' (DMoN collapse term, ablation)
    target="graph",           # 'graph' (graph-aware DEC target) | 'dec' (plain DEC)
    routing=True,
    standardize=True,
    target_every=5,           # T_P
    ncgar_alpha=0.05, ncgar_eps=0.05,
)


class FairPGTExperiment(BaseExperiment):
    def __init__(self, dataset_name, n_clusters, device="cuda", seed=42, **kwargs):
        unknown = set(kwargs) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown FairPGT options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **kwargs}
        super().__init__("fairpgt", dataset_name, n_clusters, device=device, seed=seed)

    # ------------------------------------------------------------ setup
    def _prepare(self, graph):
        c, dev = self.cfg, self.device
        U = spectral_encoding(graph.A, c["n_spectral"])
        deg = np.asarray(graph.A.sum(1)).ravel()
        x = np.hstack([graph.X, U, (deg / max(deg.max(), 1))[:, None]]).astype(np.float32)
        if c["standardize"]:                                   # per-column z-score (no labels involved)
            x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        self.x = torch.from_numpy(x).to(dev)
        n_parts = c["n_partitions"]
        if n_parts == "auto":                                  # partitions of at most max_part_size nodes
            n_parts = max(8, int(np.ceil(graph.n / c["max_part_size"])))
        self.n_parts = int(n_parts)
        self.parts_np = metis_partition(graph.A, self.n_parts)
        self.part = Partitioner(torch.from_numpy(self.parts_np).to(dev))
        coo = graph.A.tocoo()
        self.edge_index = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)).to(dev)
        E = torch.nn.functional.one_hot(torch.from_numpy(graph.S).to(dev), graph.G).float()
        self.E = E                                             # (n, G), used only by the losses
        self.E_part = self.part.gather(E)                      # (P, m, G)

    def build_model(self, **kwargs):
        c = self.cfg
        return FairPGT(self.x.shape[1], self.n_clusters, c["hidden"], c["heads"], c["layers"],
                       c["proj_dim"], c["dropout"], c["routing"])

    @torch.no_grad()
    def _init_centers(self):
        self.model.eval()
        z, _ = self.model.encode(self.x, self.part)
        km = KMeans(self.n_clusters, n_init=10, random_state=self.seed).fit(z.cpu().numpy())
        self.model.centers.copy_(torch.from_numpy(km.cluster_centers_).float())

    # ------------------------------------------------------------ training
    def _attention_loss(self, attn):
        c = self.cfg
        if c["attn_reg"] == "none" or c["lambda_par"] == 0:
            return torch.zeros((), device=self.device), {}
        if c["attn_reg"] != "ncgar":
            raise ValueError(c["attn_reg"])
        terms = [Lf.ncgar_selfdistill(A, self.E_part, c["ncgar_alpha"], c["ncgar_eps"]) for A, _ in attn]
        return torch.stack(terms).mean(), {}

    def train_epoch(self, A=None, X=None, S=None, **kwargs):
        c = self.cfg
        self.model.train()
        need_attn = c["attn_reg"] != "none" and c["lambda_par"] > 0
        z, C, attn = self.model(self.x, self.part, return_attn=need_attn)
        if self.P is None or self.epoch % c["target_every"] == 0:
            self.P = Lf.graph_aware_target(C.detach(), self.edge_index, use_graph=c["target"] == "graph")
        l_clust = Lf.kl_target(self.P, C)
        l_grp = Lf.assignment_parity_mi(C, self.E)
        l_par, att_logs = self._attention_loss(attn)
        l_reg = Lf.information_term(C) if c["reg"] == "info" else Lf.collapse_term(C)
        loss = l_clust + c["lambda_grp"] * l_grp + c["lambda_par"] * l_par + c["lambda_info"] * l_reg
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["grad_clip"])
        self.optimizer.step()
        return {"loss": loss.item(), "clust": l_clust.item(), "grp": l_grp.item(), "par": float(l_par),
                "reg": l_reg.item(), **att_logs}

    def fit(self, A=None, X=None, S=None, Y=None, n_epochs=300, graph=None, selection="last", **kwargs):
        if Y is not None:
            raise ValueError("labels must not be passed to fit()")
        torch.manual_seed(self.seed)
        self._prepare(graph)
        self.model = self.build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"],
                                          weight_decay=self.cfg["weight_decay"])
        self._init_centers()
        self.P = None
        for epoch in range(n_epochs):
            self.epoch = epoch
            logs = self.train_epoch()
            self.history.append({"epoch": epoch, **logs, "gates": self.model.gates()})
        return {}

    # ------------------------------------------------------------ inference (no S)
    @torch.no_grad()
    def soft_assignments(self):
        self.model.eval()
        _, C, _ = self.model(self.x, self.part)
        return C.cpu().numpy()

    def predict(self, A=None, X=None, **kwargs):
        return self.soft_assignments().argmax(1)

    @torch.no_grad()
    def embed(self, A=None, X=None, **kwargs):
        self.model.eval()
        z, _ = self.model.encode(self.x, self.part)
        return z.cpu().numpy()

    @torch.no_grad()
    def attention_diagnostics(self):
        """nCGAR and routing diagnostics of the trained encoder (evaluation; S used only to measure)."""
        self.model.eval()
        _, _, attn = self.model(self.x, self.part, return_attn=True)
        out = {}
        for l, (A, B) in enumerate(attn):
            # sharpness: mean row entropy relative to uniform attention over the partition
            keys = (A > 0).sum(-1).clamp_min(2).float()
            ent = -(A * torch.log(A + 1e-12)).sum(-1) / torch.log(keys)
            rows = A.sum(-1) > 0
            out[f"attn_entropy_l{l}"] = float(ent[rows].mean())
            out[f"ncgar_l{l}"] = float(Lf.ncgar_value(A, self.E_part))
            if B is not None:
                out[f"mi_route_l{l}"] = float(Lf.routing_group_mi(B, self.E))
        rk = [v for k, v in out.items() if k.startswith("mi_route_l")]
        out["mi_route"] = float(np.mean(rk)) if rk else 0.0
        out["ncgar"] = float(np.nanmean([v for k, v in out.items() if k.startswith("ncgar_l")]))
        for l, g in enumerate(self.model.gates()):
            out[f"gate_l{l}"] = g
        return out
