"""Model registry: name -> (experiment class, default hyperparameters, size limit).

Defaults are fixed across datasets (no per-dataset tuning on labels). ``max_n`` skips
methods whose implementation is quadratic in memory beyond that many nodes.
"""
from __future__ import annotations

import importlib

_SPECS = {
    # ours
    "fairpgt":      ("fgc.models.experiment:FairPGTExperiment", {}, None),
    # fairness-agnostic
    "dmon":         ("fgc.baselines.dmon:DMoNExperiment",
                     dict(hidden_dim=64, lr=1e-3, dropout=0.0, lambda_modularity=1.0, lambda_collapse=1.0), None),
    "sdcn":         ("fgc.baselines.sdcn:SDCNExperiment",
                     dict(hidden_dims=[500, 500, 2000, 10], lr=1e-3, alpha=1.0, lambda_recon=1.0, lambda_kl=0.1,
                          pretrain_epochs=50), None),
    "ccgcn":        ("fgc.baselines.ccgcn:CCGCNExperiment",
                     dict(hidden_dim=256, lr=1e-4, temperature=0.5, lambda_cluster=1.0, lambda_contrastive=1.0), None),
    "scgc":         ("fgc.baselines.scgc:SCGCExperiment",
                     dict(hidden_dim=256, lr=1e-4, temperature=0.5, lambda_instance=1.0, lambda_cluster=1.0), None),
    "dinknet":      ("fgc.baselines.dinknet:DinkNetExperiment",
                     dict(hidden_dim=256, lr=1e-4, tradeoff=0.5, temperature=0.5), None),
    "diffpool":     ("fgc.baselines.diffpool:DiffPoolExperiment",
                     dict(hidden_dim=64, lr=1e-3, lambda_link=1.0, lambda_ent=1.0), 50000),
    "mincutpool":   ("fgc.baselines.mincutpool:MinCutPoolExperiment",
                     dict(hidden_dim=64, lr=1e-3, lambda_mincut=1.0, lambda_ortho=1.0), 50000),
    # fairness-aware
    "dmon_sp":      ("fgc.baselines.fairdmon:FairDMoNExperiment",
                     dict(hidden_dim=64, lr=1e-3, lambda_modularity=1.0, lambda_fairness=0.1), None),
    "fairgc_zhang": ("fgc.baselines.fairgc_zhang:FairGCZhangExperiment", {}, 25000),   # dense O(n^3) random-walk target
    "fairgp":       ("fgc.baselines.fairgp:FairGPExperiment", dict(hidden_dim=64, n_heads=4, lr=1e-3, lambda_fair=0.1),
                     None),
    "fairgt":       ("fgc.baselines.fairgt:FairGTExperiment", dict(hidden_dim=64, n_heads=4, lr=1e-3, lambda_fair=0.1),
                     None),
    "sfairsc":      ("fgc.baselines.fair_spectral:sFairSCExperiment", {}, None),
    "ifairsc":      ("fgc.baselines.fair_spectral:iFairSCExperiment", dict(knn=10), 8000),
}


def names():
    return list(_SPECS)


def spec(name):
    if name not in _SPECS:
        raise KeyError(f"unknown model {name!r}; available: {names()}")
    path, defaults, max_n = _SPECS[name]
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls), dict(defaults), max_n
