"""Figures from finished runs.

    python -m experiments.plots tradeoff runs/sensitivity       # NMI vs. Delta_SP per weight
    python -m experiments.plots sbm runs/sbm                    # metrics vs. community-group alignment
    python -m experiments.plots curves runs/main/nba/fairpgt    # training curves of all seeds
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.report import load

plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix"})


def tradeoff(run_dir: Path):
    df = load(run_dir)
    df = df[df.status == "ok"]
    for ds, d in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for fam, f in d.groupby(d.tag.str.extract(r"^([^\[]+)")[0]):
            g = f.groupby("tag")[["nmi", "delta_sp"]].mean()
            g["w"] = [float(re.search(r"=([0-9.]+)", t).group(1)) for t in g.index]
            g = g.sort_values("w")
            ax.plot(g.delta_sp, g.nmi, "o-", ms=3, label=fam)
            for _, r in g.iterrows():
                ax.annotate(f"{r.w:g}", (r.delta_sp, r.nmi), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel(r"$\Delta_{SP}$ (lower is fairer)"); ax.set_ylabel("NMI"); ax.set_title(ds)
        ax.legend(frameon=False, fontsize=6)
        fig.tight_layout(); fig.savefig(run_dir / f"tradeoff_{ds}.pdf"); plt.close(fig)


def sbm(run_dir: Path):
    df = load(run_dir)
    df = df[df.status == "ok"].copy()
    df["align"] = df.dataset.str.extract(r"align([0-9.]+)")[0].astype(float)
    df["proxy"] = df.dataset.str.extract(r"proxy([0-9.]+)")[0].astype(float)
    metrics = [m for m in ("nmi", "modularity", "delta_sp", "ncgar") if m in df]
    for proxy, d in df.groupby("proxy"):
        fig, axes = plt.subplots(1, len(metrics), figsize=(1.9 * len(metrics), 2.0))
        for ax, m in zip(axes, metrics):
            for tag, t in d.groupby("tag"):
                g = t.groupby("align")[m].agg(["mean", "std"])
                ax.errorbar(g.index, g["mean"], g["std"], marker="o", ms=2.5, lw=1, capsize=1.5, label=tag)
            ax.set_xlabel("community-group alignment"); ax.set_title(m)
        axes[0].legend(frameon=False, fontsize=5)
        fig.tight_layout(); fig.savefig(run_dir / f"sbm_proxy{proxy:g}.pdf"); plt.close(fig)


def curves(tag_dir: Path):
    hs = [pd.read_csv(p) for p in sorted(tag_dir.glob("seed*/history.csv"))]
    cols = [c for c in ("clust", "grp", "par", "reg") if c in hs[0]]
    fig, axes = plt.subplots(1, len(cols), figsize=(1.8 * len(cols), 1.8))
    for ax, c in zip(axes, cols):
        for h in hs:
            ax.plot(h.epoch, h[c], lw=0.8, alpha=0.7)
        ax.set_title(c); ax.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(tag_dir / "curves.pdf"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["tradeoff", "sbm", "curves"])
    ap.add_argument("path")
    a = ap.parse_args()
    {"tradeoff": tradeoff, "sbm": sbm, "curves": curves}[a.kind](Path(a.path))
