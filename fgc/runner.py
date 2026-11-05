"""Run one (method, dataset, seed) and store everything needed to re-evaluate it.

runs/<experiment>/<dataset>/<tag>/seed<k>/
    config.json     method, dataset, seed, hyperparameters, protocol, library versions
    labels.npy      hard cluster labels (the only input of the evaluation)
    soft.npy        soft assignments, if the method has them
    embeddings.npy  node embeddings, if the method exposes them (for the S-probe)
    history.csv     per-epoch training logs
    model.pt        final weights (optional)
    metrics.json    output of fgc.eval.metrics.evaluate (+ probe, attention diagnostics)
One line per finished run is appended to runs/<experiment>/results.jsonl.
A run whose metrics.json exists is skipped unless force=True.
"""
from __future__ import annotations

import json
import os
import platform
import random
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from fgc import registry
from fgc.data.datasets import load
from fgc.eval.metrics import evaluate
from fgc.eval.probe import sensitive_probe

_DATA_CACHE: dict = {}


def get_data(name):
    if name not in _DATA_CACHE:
        _DATA_CACHE.clear()                       # keep one dataset in memory
        _DATA_CACHE[name] = load(name)
    return _DATA_CACHE[name]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _tensors(data, device):
    """Inputs in the format the ported baselines expect (as in the original benchmark)."""
    A = data.A if data.n > 50000 else torch.from_numpy(data.A.toarray()).float()
    return A, torch.from_numpy(data.X).float(), torch.from_numpy(data.S).long()


def run_one(model: str, dataset: str, seed: int, out_root: str, tag: str | None = None, params: dict | None = None,
            epochs: int = 200, selection: str = "last", device: str = "cuda", save_weights: bool = True,
            probe: bool = True, force: bool = False) -> dict:
    tag = tag or model
    out = Path(out_root) / dataset / tag / f"seed{seed}"
    if (out / "metrics.json").exists() and not force:
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    data = get_data(dataset)
    cls, defaults, max_n = registry.spec(model)
    hp = {**defaults, **(params or {})}
    record = dict(model=model, tag=tag, dataset=dataset, seed=seed, epochs=epochs, selection=selection,
                  hyperparameters=hp, n=data.n, K=data.K, device=device,
                  versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    if max_n is not None and data.n > max_n:
        record.update(status="skipped", reason=f"n={data.n} > max_n={max_n}")
        _finish(out, out_root, record, {})
        return record
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        exp = cls(dataset_name=dataset, n_clusters=data.K, device=device, seed=seed, **hp)
        A, X, S = _tensors(data, exp.device)
        exp.fit(A, X, S, n_epochs=epochs, graph=data, selection=selection)
        train_time = time.time() - t0
        with torch.no_grad():
            labels = np.asarray(exp.predict(A.to(exp.device) if isinstance(A, torch.Tensor) else A,
                                            X.to(exp.device)))
        metrics = evaluate(labels, data.A, data.S, data.Y, data.K)
        np.save(out / "labels.npy", labels.astype(np.int32))
        if hasattr(exp, "soft_assignments"):
            np.save(out / "soft.npy", exp.soft_assignments().astype(np.float32))
        Z = exp.embed(A.to(exp.device) if isinstance(A, torch.Tensor) else A, X.to(exp.device))
        if Z is not None:
            Z = np.asarray(Z.detach().cpu() if isinstance(Z, torch.Tensor) else Z, dtype=np.float32)
            np.save(out / "embeddings.npy", Z)
            if probe:
                metrics["probe_s"] = sensitive_probe(Z, data.S, seed)
        if hasattr(exp, "attention_diagnostics"):
            metrics.update(exp.attention_diagnostics())
        if exp.history:
            import pandas as pd
            pd.DataFrame(exp.history).to_csv(out / "history.csv", index=False)
        if save_weights and isinstance(getattr(exp, "model", None), torch.nn.Module):
            torch.save(exp.model.state_dict(), out / "model.pt")
        record.update(status="ok", train_time_s=round(train_time, 2),
                      peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None)
    except Exception as e:                                           # record failures, do not stop the sweep
        metrics = {}
        record.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    _finish(out, out_root, record, metrics)
    return {**record, **metrics}


def _finish(out: Path, out_root: str, record: dict, metrics: dict):
    (out / "config.json").write_text(json.dumps(record, indent=2, default=str))
    row = {**{k: v for k, v in record.items() if k not in ("traceback", "versions")}, **metrics}
    (out / "metrics.json").write_text(json.dumps(row, indent=2, default=str))
    with open(Path(out_root) / "results.jsonl", "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
