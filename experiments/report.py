"""Aggregate results.jsonl into mean ± std tables (CSV, Markdown, LaTeX).

    python -m experiments.report runs/main
    python -m experiments.report runs/main --metrics nmi f1_macro modularity delta_sp --latex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT = ["nmi", "f1_macro", "modularity", "delta_sp", "balance", "size_entropy", "size_min_max", "probe_s", "ncgar", "train_time_s"]


def load(run_dir: Path) -> pd.DataFrame:
    """Read every run's metrics.json (robust to parallel jobs writing into the same folder)."""
    rows = [json.loads(p.read_text()) for p in sorted(run_dir.glob("*/*/seed*/metrics.json"))]
    if not rows:
        raise FileNotFoundError(f"no finished runs under {run_dir}")
    return pd.DataFrame(rows).drop_duplicates(["dataset", "tag", "seed"], keep="last")


def table(df: pd.DataFrame, metrics) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    metrics = [m for m in metrics if m in ok]
    g = ok.groupby(["dataset", "tag"])
    mean, std, n = g[metrics].mean(), g[metrics].std(), g.size().rename("n_seeds")
    out = mean.copy().astype(object)
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" if pd.notna(b) else f"{a:.3f}" for a, b in zip(mean[m], std[m])]
    if "collapsed" in ok:
        out["collapsed_runs"] = g["collapsed"].sum()
    return out.join(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--metrics", nargs="*", default=DEFAULT)
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    df = load(run_dir)
    t = table(df, a.metrics)
    t.to_csv(run_dir / "summary.csv")
    (run_dir / "summary.md").write_text(t.to_markdown())
    if a.latex:
        (run_dir / "summary.tex").write_text(t.to_latex())
    failed = df[df["status"] == "failed"]
    print(t.to_markdown())
    if len(failed):
        print(f"\n{len(failed)} failed runs:")
        print(failed[["dataset", "tag", "seed", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
