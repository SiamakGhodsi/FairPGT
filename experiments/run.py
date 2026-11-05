"""Run an experiment described by a YAML file.

    python -m experiments.run experiments/configs/main.yaml
    python -m experiments.run experiments/configs/main.yaml --datasets nba german --models fairpgt --seeds 0
    python -m experiments.run experiments/configs/ablation.yaml --device cpu --dry-run

Config keys
  name        experiment name; results go to <out>/<name>/
  datasets    list of dataset names (see fgc.data.datasets.REGISTRY; 'sbm_...' names allowed)
  seeds       list of seeds
  epochs      epoch budget (same for every method), selection: last | modularity
  models      {tag: {base: <registry name>, <hyperparameter>: value, ...}}  (base defaults to tag)
  grid        {tag: {param: [values]}}  -> one variant per value, tag "tag[param=value]"
  dataset_overrides  {dataset: {tag: {param: value}}}
"""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import yaml

from fgc.runner import run_one


def expand_models(cfg):
    models = {tag: dict(spec or {}) for tag, spec in (cfg.get("models") or {}).items()}
    for tag, grid in (cfg.get("grid") or {}).items():
        grid = dict(grid)
        base_name = grid.pop("base", None)
        base = models.pop(tag, {})
        keys = list(grid)
        for vals in itertools.product(*(grid[k] for k in keys)):
            name = tag + "[" + ",".join(f"{k}={v}" for k, v in zip(keys, vals)) + "]"
            models[name] = {**base, **dict(zip(keys, vals)), "base": base_name or base.get("base", tag)}
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*", help="tags to run (default: all in the config)")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--force", action="store_true", help="re-run finished runs")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(args.out) / cfg["name"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.yaml").write_text(Path(args.config).read_text())
    models = expand_models(cfg)
    if args.models:
        models = {t: m for t, m in models.items() if t in args.models}
    datasets = args.datasets or cfg["datasets"]
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
    epochs = args.epochs or cfg.get("epochs", 200)
    jobs = [(d, t, s) for d in datasets for t in models for s in seeds]
    print(f"{cfg['name']}: {len(jobs)} runs ({len(datasets)} datasets x {len(models)} methods x {len(seeds)} seeds)", flush=True)
    for i, (d, tag, s) in enumerate(jobs, 1):
        spec = dict(models[tag])
        base = spec.pop("base", tag)
        spec.update((cfg.get("dataset_overrides") or {}).get(d, {}).get(tag, {}))
        if args.dry_run:
            print(f"[{i}/{len(jobs)}] {d} {tag} seed{s} base={base} {spec}")
            continue
        r = run_one(base, d, s, str(out_root), tag=tag, params=spec, epochs=epochs,
                    selection=cfg.get("selection", "last"), device=args.device,
                    save_weights=not args.no_weights, force=args.force)
        msg = r.get("status")
        if msg == "ok":
            msg = " ".join(f"{k}={r[k]:.3f}" for k in ("nmi", "f1_macro", "modularity", "delta_sp") if k in r)
        print(f"[{i}/{len(jobs)}] {d:15s} {tag:28s} seed{s}  {msg}", flush=True)


if __name__ == "__main__":
    main()
