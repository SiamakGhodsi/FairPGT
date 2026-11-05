"""Collect what the paper tables and figures need into one folder (then zip it).

    python scripts/export_paper_data.py                      # reads runs/, writes paper_data/
    python scripts/export_paper_data.py --device cuda

Writes
  paper_data/runs/...            metrics.json + config.json of every run (all experiments)
  paper_data/arrays/...          seed-0 labels / soft assignments / embeddings for the
                                 embedding figure (main: graphs below 50k nodes; sbm; ablation)
  paper_data/attention/*.npz     trained attention of one partition per layer and head, for the
                                 attention-matrix figure (FairPGT runs, seed 0)
  paper_data/dataset_stats.json  N, |E|, d, K, groups, majority share, homophily, density, ...
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

SMALL = {"nba", "german", "lastfm", "facebook_large", "credit"}
REAL = ["nba", "german", "lastfm", "facebook_large", "credit", "pokec_n", "pokec_z"]
ATTN_SBM = ["sbm_align0.0_proxy1.0", "sbm_align0.5_proxy1.0", "sbm_align0.75_proxy1.0", "sbm_align1.0_proxy1.0"]
MAX_NODES = 400          # attention submatrix kept per partition (stratified by group)


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def want_arrays(exp, ds):
    return (exp == "main" and ds in SMALL) or (exp == "sbm" and "proxy1.0" in ds) or \
           (exp == "ablation" and ds in SMALL) or exp == "embed"


def dataset_stats(names, dst: Path):
    from fgc.runner import get_data
    out = {}
    for ds in names:
        try:
            g = get_data(ds)
        except Exception as e:                                   # missing data folder etc.
            out[ds] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if ds in SMALL or ds.startswith("sbm"):                  # group labels for the embedding figure
            (dst / "arrays" / "_S").mkdir(parents=True, exist_ok=True)
            np.save(dst / "arrays" / "_S" / f"{ds}.npy", g.S.astype(np.int8))
        A = g.A.tocoo()
        n, m = g.n, A.nnz // 2
        same = float((g.S[A.row] == g.S[A.col]).mean()) if A.nnz else float("nan")
        lab_same = float((g.Y[A.row] == g.Y[A.col]).mean()) if A.nnz else float("nan")
        sh = np.bincount(g.S) / n
        from scipy.sparse.csgraph import connected_components
        out[ds] = dict(N=int(n), E=int(m), d=int(g.X.shape[1]), K=int(g.K), G=int(g.G),
                       majority_share=float(sh.max()), group_shares=sh.round(4).tolist(),
                       label_shares=(np.bincount(g.Y) / n).round(4).tolist(),
                       homophily_S=same, homophily_Y=lab_same, density=2 * m / (n * (n - 1)),
                       mean_degree=2 * m / n, components=int(connected_components(g.A)[0]))
        print(ds, out[ds], flush=True)
    return out


@torch.no_grad()
def export_attention(run_dir: Path, dst: Path, device: str):
    from fgc.models.experiment import FairPGTExperiment
    from fgc.runner import get_data
    cfg = json.loads((run_dir / "config.json").read_text())
    ds, hp = cfg["dataset"], cfg["hyperparameters"]
    g = get_data(ds)
    exp = FairPGTExperiment(dataset_name=ds, n_clusters=g.K, device=device, seed=cfg["seed"], **hp)
    exp._prepare(g)
    exp.model = exp.build_model().to(exp.device)
    exp.model.load_state_dict(torch.load(run_dir / "model.pt", map_location=exp.device))
    exp.model.eval()
    _, C, attn = exp.model(exp.x, exp.part, return_attn=True)
    idx, valid = exp.part.idx.cpu().numpy(), exp.part.valid.cpu().numpy()
    # partition with the most balanced group composition among those with >= 50 nodes
    best, score = 0, -1.0
    for p in range(idx.shape[0]):
        ids = idx[p][valid[p]]
        if len(ids) < 50:
            continue
        sh = np.bincount(g.S[ids], minlength=g.G) / len(ids)
        if sh.min() > score:
            best, score = p, sh.min()
    pos = np.flatnonzero(valid[best])
    ids = idx[best][pos]
    if len(ids) > MAX_NODES:                                    # stratified subsample by group
        rng = np.random.default_rng(0)
        keep = []
        for s in range(g.G):
            k = np.flatnonzero(g.S[ids] == s)
            take = int(round(MAX_NODES * len(k) / len(ids)))
            keep.append(rng.choice(k, min(take, len(k)), replace=False))
        sel = np.sort(np.concatenate(keep))
        pos, ids = pos[sel], ids[sel]
    mats = np.stack([A[best][:, pos][:, :, pos].cpu().numpy() for A, _ in attn])   # (L, H, m, m)
    labels = np.load(run_dir / "labels.npy") if (run_dir / "labels.npy").exists() else C.argmax(1).cpu().numpy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, attention=mats.astype(np.float32), nodes=ids, S=g.S[ids], Y=g.Y[ids],
                        cluster=labels[ids], partition=best, partition_size=int(valid[best].sum()),
                        dataset=ds, tag=cfg["tag"])
    print("attention", run_dir, mats.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper_data")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)

    n_json = n_arr = 0
    for f in runs.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(runs)
        if f.name in ("metrics.json", "config.json", "config.yaml"):
            copy(f, out / "runs" / rel); n_json += 1
        elif f.suffix == ".npy" and len(rel.parts) == 5 and rel.parts[3] == "seed0" \
                and want_arrays(rel.parts[0], rel.parts[1]):
            copy(f, out / "arrays" / rel); n_arr += 1
    print(f"copied {n_json} json/yaml and {n_arr} arrays", flush=True)

    sbm_names = sorted({p.name for p in (runs / "sbm").glob("sbm_*")}) if (runs / "sbm").exists() else []
    stats = dataset_stats(REAL + sbm_names, out)
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    targets = [runs / "sbm" / ds / t / "seed0" for ds in ATTN_SBM
               for t in ("fairpgt", "fairpgt-no-fair", "fairpgt-grp-only", "fairpgt-par-only")]
    targets += [runs / "ablation" / ds / t / "seed0" for ds in ("lastfm", "nba", "credit")
                for t in ("fairpgt", "fairpgt-no-fair", "fairpgt-grp-only", "fairpgt-par-only")]
    for r in targets:
        if (r / "model.pt").exists():
            try:
                export_attention(r, out / "attention" / f"{r.parts[-4]}__{r.parts[-3]}__{r.parts[-2]}.npz", a.device)
            except Exception as e:
                print("attention failed", r, type(e).__name__, e, flush=True)
        else:
            print("no model.pt", r, flush=True)


if __name__ == "__main__":
    main()
