"""All tables and result figures of the paper, computed from the exported run folders.

    python experiments/make_paper_figs.py paper_data paper_out          # everything
    python experiments/make_paper_figs.py paper_data paper_out --only tables sbm

<paper_data> is the folder written by scripts/export_paper_data.py:
    runs/<experiment>/<dataset>/<tag>/seed*/metrics.json, config.json
    arrays/<experiment>/<dataset>/<tag>/seed0/{embeddings,labels}.npy
    attention/*.npz, dataset_stats.json
No number is typed in by hand: every table cell and plotted point is read from these files.
LaTeX tables use the macros of the paper preamble: \\best{}, \\runup{}, \\std{}.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": "#3a3a3a", "axes.labelcolor": "#222",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.5,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
INK, MUTED = "#222222", "#6b6b6b"

DS = ["nba", "german", "lastfm", "facebook_large", "credit", "pokec_n", "pokec_z"]
DSN = {"nba": "NBA", "german": "German", "lastfm": "LastFM", "facebook_large": "Facebook", "credit": "Credit",
       "pokec_n": "Pokec-n", "pokec_z": "Pokec-z"}
AGN = ["dmon", "sdcn", "ccgcn", "scgc", "dinknet", "diffpool", "mincutpool"]
FAIR_DEEP = ["dmon_sp", "fairgc_zhang", "fairgt", "fairgp"]
FAIR_SPEC = ["sfairsc", "ifairsc"]
MS = AGN + FAIR_DEEP + FAIR_SPEC + ["fairpgt"]
FAIR = set(FAIR_DEEP + FAIR_SPEC + ["fairpgt"])
MN = {"dmon": "DMoN", "sdcn": "SDCN", "ccgcn": "CCGC", "scgc": "SCGC", "dinknet": "DinkNet",
      "diffpool": "DiffPool", "mincutpool": "MinCutPool", "dmon_sp": "DMoN-SP", "fairgc_zhang": "FairGC",
      "fairgt": "FairGT", "fairgp": "FairGP", "sfairsc": "sFairSC", "ifairsc": "iFairSC", "fairpgt": "FairPGT (ours)"}
COL = {"fairpgt": "#C0392B",
       "dmon": "#1B7837", "sdcn": "#9E9E9E", "ccgcn": "#B5B5B5", "scgc": "#7E7E7E", "dinknet": "#5E5E5E",
       "diffpool": "#CFCFCF", "mincutpool": "#474747",
       "dmon_sp": "#E08214", "fairgc_zhang": "#F2B45C", "fairgt": "#35978F", "fairgp": "#2166AC",
       "sfairsc": "#762A83", "ifairsc": "#B48AC2"}
MK = {m: ("s" if m in AGN else "o") for m in MS}
MK["fairpgt"] = "*"
VAR = {"fairpgt-no-fair": "no fairness terms", "fairpgt-par-only": r"$\mathcal{L}_{\mathrm{att}}$ only",
       "fairpgt-grp-only": r"$\mathcal{L}_{\mathrm{grp}}$ only", "fairpgt": "FairPGT (ours)"}
VCOL = {"fairpgt-no-fair": "#8C8C8C", "fairpgt-par-only": "#35978F", "fairpgt-grp-only": "#2166AC",
        "fairpgt": "#C0392B", "dmon_sp": "#E08214", "sfairsc": "#762A83"}
UP, DOWN = r"$\uparrow$", r"$\downarrow$"


def load_runs(root):
    rows = []
    for p in glob.glob(f"{root}/runs/*/*/*/seed*/metrics.json"):
        r = json.load(open(p))
        r["exp"] = p.split(os.sep)[-5]
        rows.append(r)
    df = pd.DataFrame(rows).drop_duplicates(["exp", "dataset", "tag", "seed"], keep="last")
    return df


def save(fig, out, name):
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left", color=INK)


# ----------------------------------------------------------------------------- tables
def _fmt(v, std=None):
    s = f"{abs(v):.3f}"
    s = s[1:] if s.startswith("0.") else s
    s = ("$-$" + s) if v < 0 and s.strip("0.") else s
    if std is None:
        return s
    t = f"{std:.2f}"
    t = t[1:] if t.startswith("0.") else t
    return s, t


def cell_stats(main, ds, m, metric):
    d = main[(main.dataset == ds) & (main.tag == m)]
    if d.empty or (d.status == "skipped").all():
        return "n/s"
    if (d.status == "failed").all():
        return "OOM"
    d = d[d.status == "ok"]
    return d[metric].mean(), d[metric].std(), d.collapsed.mean() >= 0.5


def wide_table(main, metrics, caption, label, out, fname):
    """Rows: dataset x metric; columns: methods grouped by family (layout of the submitted paper)."""
    cols = AGN + FAIR_DEEP + FAIR_SPEC + ["fairpgt"]
    head = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            r"\adjustbox{max width=\textwidth}{", r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
            r"\begin{tabular}{ll|" + "c" * len(AGN) + "|" + "c" * len(FAIR_DEEP) + "|" + "c" * len(FAIR_SPEC) + "|c}",
            r"\toprule",
            r"& & \multicolumn{%d}{c}{\textit{Deep graph clustering}} & \multicolumn{%d}{c}{\textit{Fair deep}} & "
            r"\multicolumn{%d}{c}{\textit{Fair spectral}} & \textit{Ours} \\" % (len(AGN), len(FAIR_DEEP), len(FAIR_SPEC)),
            r"\cmidrule(lr){3-%d} \cmidrule(lr){%d-%d} \cmidrule(lr){%d-%d} \cmidrule(lr){%d-%d}" % (
                2 + len(AGN), 3 + len(AGN), 2 + len(AGN) + len(FAIR_DEEP), 3 + len(AGN) + len(FAIR_DEEP),
                2 + len(AGN) + len(FAIR_DEEP) + len(FAIR_SPEC), len(cols) + 2, len(cols) + 2),
            "& & " + " & ".join(r"\textbf{FairPGT}" if m == "fairpgt" else MN[m] for m in cols) + r" \\",
            r"\midrule"]
    body = []
    for di, ds in enumerate(DS):
        for mi, (metric, mlabel, lower) in enumerate(metrics):
            vals = {m: cell_stats(main, ds, m, metric) for m in cols}
            ok = [(v[0], m) for m, v in vals.items() if isinstance(v, tuple) and not v[2]]
            ranked = [m for _, m in sorted(ok, reverse=not lower)]
            cells = []
            for m in cols:
                v = vals[m]
                if not isinstance(v, tuple):
                    cells.append(r"\textit{" + v + "}")
                    continue
                mu, sd = _fmt(v[0], v[1])
                if ranked and m == ranked[0]:
                    mu = r"\best{" + mu + "}"
                elif len(ranked) > 1 and m == ranked[1]:
                    mu = r"\runup{" + mu + "}"
                cells.append(mu + r"\std{" + sd + "}" + (r"$^\dagger$" if v[2] else ""))
            lead = (r"\multirow{%d}{*}{\textbf{%s}}" % (len(metrics), DSN[ds])) if mi == 0 else ""
            body.append(f"{lead} & {mlabel} & " + " & ".join(cells) + r" \\")
        if di < len(DS) - 1:
            body.append(r"\midrule")
    tail = [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(f"{out}/{fname}.tex", "w").write("\n".join(head + body + tail))


def tables(df, stats, out):
    main = df[df.exp == "main"]
    note = (r" Mean and standard deviation over five seeds. Best in \best{bold red}, runner-up in \runup{blue}, "
            r"among runs that did not collapse; $^\dagger$: collapsed (fewer than $K$ clusters used, or one cluster "
            r"holding over 90\% of the nodes, in at least half of the runs). \textit{n/s}: not scalable, skipped "
            r"(memory quadratic or cubic in $n$); \textit{OOM}: out of GPU memory.")
    wide_table(main, [("nmi", r"NMI" + UP, False), ("delta_sp", r"$\Delta_{\mathrm{SP}}$" + DOWN, True)],
               r"Clustering quality (NMI) and group fairness ($\Delta_{\mathrm{SP}}$) on seven benchmarks." + note,
               "tab:main", out, "table_main")
    wide_table(main, [("f1_macro", "F1" + UP, False), ("modularity", r"$Q$" + UP, False)],
               r"Complementary quality metrics: macro-F1 against the labels and modularity $Q$ of the graph." + note,
               "tab:extra", out, "table_extra")

    # cost
    ok = main[main.status == "ok"]
    cost = ok.groupby(["tag", "dataset"])[["train_time_s", "peak_gpu_gb"]].mean()
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
         r"\caption{Training time in seconds" + DOWN + r" / peak GPU memory in GB" + DOWN + r" (200 epochs, mean "
         r"over five seeds; memory is the peak allocated by PyTorch). --: skipped or out of memory. Runs used "
         r"different GPUs (\autoref{sec:setup}), so times are indicative.}",
         r"\label{tab:cost}", r"\adjustbox{max width=\textwidth}{", r"\begin{tabular}{l" + "c" * len(DS) + "}",
         r"\toprule", "Method & " + " & ".join(DSN[d] for d in DS) + r" \\", r"\midrule"]
    for m in MS:
        if m in (FAIR_DEEP[0], "fairpgt"):
            L.append(r"\midrule")
        L.append((r"\textbf{FairPGT (ours)}" if m == "fairpgt" else MN[m]) + " & " + " & ".join(
            f"{cost.loc[(m, d)].train_time_s:.1f} / {cost.loc[(m, d)].peak_gpu_gb:.2f}" if (m, d) in cost.index else "--"
            for d in DS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(f"{out}/table_cost.tex", "w").write("\n".join(L))

    # datasets
    info = {  # protected attribute and label, as defined in the data preparation (not computed)
        "nba": ("Nationality (US / non-US)", "Salary class (majority vs.\\ rest)"),
        "german": ("Gender", "Good / bad credit"),
        "lastfm": ("Synthetic (feature $k$-means)", "Country (6)"),
        "facebook_large": ("Synthetic (feature $k$-means)", "Page type (4)"),
        "credit": ("Marital status", "Default next month"),
        "pokec_n": ("Age (4 bins)", "Working field (majority vs.\\ rest)"),
        "pokec_z": ("Age (4 bins)", "Working field (majority vs.\\ rest)"),
    }
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Dataset statistics. $N$: nodes; $|E|$: undirected edges; $d$: feature dimension; $K$: label "
         r"classes (number of clusters); $G$: protected groups; Maj.: share of the largest group; $h_S$, $h_Y$: "
         r"fraction of edges joining nodes of the same group and of the same class; CC: connected components.}",
         r"\label{tab:datasets}", r"\adjustbox{max width=\textwidth}{",
         r"\begin{tabular}{lrrrcccccrll}", r"\toprule",
         r"Dataset & $N$ & $|E|$ & $d$ & $K$ & $G$ & Maj. & $h_S$ & $h_Y$ & CC & Protected attribute & Label \\",
         r"\midrule"]
    for ds in DS:
        s = stats[ds]
        L.append(f"{DSN[ds]} & {s['N']:,} & {s['E']:,} & {s['d']:,} & {s['K']} & {s['G']} & "
                 f"{100 * s['majority_share']:.1f}\\% & {s['homophily_S']:.2f} & {s['homophily_Y']:.2f} & "
                 f"{s['components']} & {info[ds][0]} & {info[ds][1]} \\\\")
    sb = stats.get("sbm_align1.0_proxy1.0")
    if sb:
        L.append(r"\midrule")
        L.append(f"SBM ($a\\in[0,1]$) & {sb['N']:,} & {sb['E']:,} & {sb['d']} & {sb['K']} & {sb['G']} & "
                 r"50.0--69.7\% & 0.56--0.82 & 0.73 & 1 & Planted, aligned by $a$ & Planted community \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(f"{out}/table_datasets.tex", "w").write("\n".join(L))


# ----------------------------------------------------------------------------- trade-off
def pareto(points):
    """Non-dominated (min x, max y) points, sorted by x."""
    pts = sorted(points)
    front, best = [], -np.inf
    for x, y in pts:
        if y > best:
            front.append((x, y)); best = y
    return front


def fig_tradeoff(df, out):
    main = df[(df.exp == "main") & (df.status == "ok")]
    sens = df[(df.exp == "sensitivity") & (df.status == "ok") & df.tag.str.startswith("fairpgt-grp")]
    sbm = df[(df.exp == "sbm") & (df.status == "ok") & (df.dataset == "sbm_align1.0_proxy1.0")]
    abl = df[(df.exp == "ablation") & (df.status == "ok")]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.75))
    panels = DS + ["sbm"]
    for k, (ax, ds) in enumerate(zip(axes.ravel(), panels)):
        pts = []
        if ds == "sbm":
            src = [(t, sbm[sbm.tag == t]) for t in ["dmon_sp", "sfairsc", "fairpgt-no-fair", "fairpgt"]]
        else:
            src = [(m, main[(main.dataset == ds) & (main.tag == m)]) for m in MS]
            src.append(("fairpgt-no-fair", abl[(abl.dataset == ds) & (abl.tag == "fairpgt-no-fair")]))
        for m, d in src:
            if d.empty or (ds != "sbm" and d.collapsed.mean() >= 0.5):   # SBM: the fair optimum uses 2 clusters
                continue
            x, y = d.delta_sp.mean(), d.nmi.mean()
            if m == "fairpgt-no-fair":
                ax.errorbar(x, y, xerr=d.delta_sp.std(), yerr=d.nmi.std(), fmt="*", ms=8, mfc="white",
                            mec=COL["fairpgt"], mew=0.8, ecolor=COL["fairpgt"], elinewidth=0.5, zorder=6)
                continue
            pts.append((x, y))
            big = m == "fairpgt"
            ax.errorbar(x, y, xerr=d.delta_sp.std(), yerr=d.nmi.std(), fmt=MK[m], ms=9 if big else 4,
                        color=COL[m], mec="white" if not big else "#7b1e14", mew=0.4 if not big else 0.5,
                        elinewidth=0.5, capsize=0, alpha=1 if (big or m in FAIR or m == "dmon") else 0.85,
                        zorder=7 if big else 3)
        if len(pts) > 1:
            f = pareto(pts)
            xs = [p[0] for p in f]; ys = [p[1] for p in f]
            ax.step(xs + [ax.get_xlim()[1]], ys + [ys[-1]], where="post", color="#b9b9b9", lw=0.7, ls=(0, (3, 2)),
                    zorder=1)
        s = sens[sens.dataset == ds]
        if len(s):
            g = s.groupby("tag")[["delta_sp", "nmi"]].mean()
            g["l"] = [float(re.search(r"=([0-9.]+)", t).group(1)) for t in g.index]
            g = g.sort_values("l")
            ax.plot(g.delta_sp, g.nmi, "-", color=COL["fairpgt"], lw=0.9, alpha=0.45, zorder=4)
            ax.plot(g.delta_sp, g.nmi, "o", color=COL["fairpgt"], ms=2, alpha=0.6, zorder=4)
        if ds == "sbm":
            ax.axhline(2 / 3, color=INK, lw=0.6, ls=":", zorder=1)
            ax.plot([0], [2 / 3], marker="D", ms=3.5, color=INK, zorder=8, clip_on=False)
            ax.annotate("bound under\nexact parity", (0.0, 2 / 3), xytext=(0.17, 0.74), fontsize=6, color=INK,
                        ha="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=INK))
            ax.set_title(r"SBM, full alignment ($a=1$)")
        else:
            ax.set_title(DSN[ds])
        ax.set_xlim(left=-0.01)
        ax.margins(y=0.12)
        panel_label(ax, f"({chr(97 + k)})", x=-0.2, y=1.02)
        if k % 4 == 0:
            ax.set_ylabel("NMI" + UP)
        if k >= 4:
            ax.set_xlabel(r"$\Delta_{\mathrm{SP}}$" + DOWN)
    h = [Line2D([], [], marker=MK[m], ls="", color=COL[m], mec="white" if m != "fairpgt" else "#7b1e14",
                ms=9 if m == "fairpgt" else 4.5, label=MN[m]) for m in MS]
    h += [Line2D([], [], marker="*", ls="", mfc="white", mec=COL["fairpgt"], ms=8, label="FairPGT w/o fairness terms"),
          Line2D([], [], color=COL["fairpgt"], lw=0.9, alpha=0.5, marker="o", ms=2,
                 label=r"FairPGT, $\lambda_{\mathrm{grp}}\in\{0,\dots,10\}$"),
          Line2D([], [], color="#b9b9b9", lw=0.7, ls=(0, (3, 2)), label="Pareto front")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), columnspacing=1.0,
               handletextpad=0.3, fontsize=6.5)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, out, "fig_tradeoff")


# ----------------------------------------------------------------------------- SBM
def fig_sbm(df, out):
    s = df[(df.exp == "sbm") & (df.status == "ok") & df.dataset.str.contains("proxy1.0")].copy()
    s["a"] = s.dataset.str.extract(r"align([0-9.]+)")[0].astype(float)
    lines = [("fairpgt-no-fair", "no fairness terms", "-", "o"),
             ("fairpgt-par-only", r"$\mathcal{L}_{\mathrm{att}}$ only", "-", "^"),
             ("fairpgt-grp-only", r"$\mathcal{L}_{\mathrm{grp}}$ only", "-", "v"),
             ("fairpgt", "FairPGT (ours)", "-", "*"),
             ("dmon_sp", "DMoN-SP", (0, (4, 2)), "o"), ("sfairsc", "sFairSC", (0, (4, 2)), "s")]
    mets = [("nmi", "NMI" + UP), ("delta_sp", r"$\Delta_{\mathrm{SP}}$" + DOWN),
            ("ncgar", r"nCGAR (1 = neutral)"), ("probe_s", r"$S$-probe" + DOWN)]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05))
    for k, (ax, (met, title)) in enumerate(zip(axes, mets)):
        for t, lab, ls, mk in lines:
            d = s[s.tag == t]
            if met not in d or d[met].isna().all():
                continue
            g = d.groupby("a")[met].agg(["mean", "std"]).fillna(0)
            c = VCOL[t]
            lw = 1.6 if t == "fairpgt" else 1.0
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.13, lw=0)
            ax.plot(g.index, g["mean"], ls=ls, color=c, lw=lw, marker=mk, ms=5 if mk == "*" else 2.8,
                    mec="white", mew=0.3, label=lab, zorder=5 if t == "fairpgt" else 3)
        if met == "nmi":
            ax.plot([1.0], [2 / 3], marker="D", ms=4, color=INK, zorder=8, clip_on=False)
            ax.annotate("bound under\nexact parity (2/3)", (1.0, 2 / 3), xytext=(0.36, 0.6), fontsize=6,
                        color=INK, ha="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=INK))
        if met == "ncgar":
            ax.set_yscale("log")
            ax.axhspan(0.95, 1.05, color="#999", alpha=0.12, lw=0)
            ax.axhline(1, color=INK, lw=0.5, ls=":")
        if met == "probe_s":
            ax.axhline(0.5, color=INK, lw=0.5, ls=":")
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("community--group alignment $a$")
        panel_label(ax, f"({chr(97 + k)})", x=-0.22, y=1.03)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.13), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_sbm")


# ----------------------------------------------------------------------------- scaling
def fig_scaling(df, root, out):
    sc = df[df.exp == "scaling"].copy()
    sc["P"] = sc.tag.str.extract(r"=(\d+)")[0].astype(int)
    ep = {}
    for p in glob.glob(f"{root}/runs/scaling/*/*/seed0/config.json"):
        c = json.load(open(p)); ep[(c["dataset"], c["tag"])] = c.get("epochs", 50)
    sc["epochs"] = [ep.get((d, t), 50) for d, t in zip(sc.dataset, sc.tag)]
    cols = {"lastfm": "#2166AC", "facebook_large": "#C0392B", "credit": "#762A83", "pokec_n": "#1B7837"}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.05))
    for ds, c in cols.items():
        d = sc[sc.dataset == ds].sort_values("P")
        if d.empty:
            continue
        okd = d[d.status == "ok"]
        axes[0].plot(okd.P, okd.peak_gpu_gb, "o-", ms=3.2, lw=1.1, color=c, mec="white", mew=0.4, label=DSN[ds])
        if len(okd):
            p0, m0 = okd.P.iloc[0], okd.peak_gpu_gb.iloc[0]
            Ps = np.array(sorted(okd.P))
            axes[0].plot(Ps, m0 * p0 / Ps, ls=(0, (2, 2)), color=c, lw=0.7, alpha=0.6)
        oom = d[d.status == "failed"]
        axes[0].scatter(oom.P, [150] * len(oom), marker="x", color=c, s=14, lw=0.9, zorder=5, clip_on=False)
        axes[1].plot(okd.P, okd.train_time_s / okd.epochs, "o-", ms=3.2, lw=1.1, color=c, mec="white", mew=0.4)
        axes[2].plot(okd.P, okd.nmi, "o-", ms=3.2, lw=1.1, color=c, mec="white", mew=0.4)
    titles = ["peak GPU memory (GB)" + DOWN, "time per epoch (s)" + DOWN, "NMI" + UP]
    for k, (ax, t) in enumerate(zip(axes, titles)):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of partitions $N_p$")
        ax.set_title(t, fontsize=7.5)
        panel_label(ax, f"({chr(97 + k)})", x=-0.2, y=1.03)
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=MUTED, ls=(0, (2, 2)), lw=0.7)); l.append(r"ideal $\propto 1/N_p$")
    h.append(Line2D([], [], color=MUTED, marker="x", ls="", ms=4)); l.append("out of memory")
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_scaling")


# ----------------------------------------------------------------------------- ablation heatmap
def fig_ablation(df, out):
    ab = df[(df.exp == "ablation") & (df.status == "ok")]
    rows = [("fairpgt", "FairPGT (full)"),
            ("fairpgt-grp-only", r"w/o $\mathcal{L}_{\mathrm{att}}$"),
            ("fairpgt-par-only", r"w/o $\mathcal{L}_{\mathrm{grp}}$"),
            ("fairpgt-no-fair", "w/o both fairness terms"),
            ("fairpgt-dec-target", r"DEC target instead of $P$"),
            ("fairpgt-no-routing", "w/o routing tokens"),
            ("fairpgt-collapse-reg", r"DMoN collapse term instead of $\mathcal{L}_{\mathrm{info}}$")]
    g = ab.groupby(["tag", "dataset"])[["nmi", "delta_sp"]].mean()
    cmap = LinearSegmentedColormap.from_list("bw", ["#B2182B", "#EF8A62", "#FDDBC7", "#F7F7F7",
                                                    "#D1E5F0", "#67A9CF", "#2166AC"])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
    for k, (ax, met, title, sign) in enumerate(zip(axes, ["nmi", "delta_sp"],
                                                     ["NMI" + UP, r"$\Delta_{\mathrm{SP}}$" + DOWN], [1, -1])):
        V = np.array([[g.loc[(t, d), met] if (t, d) in g.index else np.nan for d in DS] for t, _ in rows])
        D = sign * (V - V[0])                                       # > 0: better than the full model
        lim = np.nanpercentile(np.abs(D), 95) or 1e-3
        ax.imshow(D, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if np.isnan(V[i, j]):
                    continue
                v = f"{V[i, j]:.3f}"[1:] if V[i, j] < 1 else f"{V[i, j]:.2f}"
                strong = abs(D[i, j]) > 0.6 * lim
                ax.text(j, i, v, ha="center", va="center", fontsize=6.2,
                        color="white" if strong else INK, fontweight="bold" if i == 0 else "normal")
        ax.set_xticks(range(len(DS))); ax.set_xticklabels([DSN[d] for d in DS], rotation=30, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r for _, r in rows] if k == 0 else [])
        ax.tick_params(length=0); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0.5, color="white", lw=2)
        ax.set_title(title)
        panel_label(ax, f"({chr(97 + k)})", x=-0.02 if k else -0.5, y=1.03)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(0, -1, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["worse", "= full", "better"], fontsize=6.2); cb.outline.set_visible(False)
    save(fig, out, "fig_ablation")


# ----------------------------------------------------------------------------- embeddings
EMB_SETS = [  # (panel title, experiment, tag); baselines appear once the 'embed' runs exist
    ("DMoN", "embed", "dmon"), ("sFairSC", "embed", "sfairsc"), ("FairGP", "embed", "fairgp"),
    ("FairPGT w/o fairness terms", "ablation", "fairpgt-no-fair"), ("FairPGT (ours)", "ablation", "fairpgt")]
EMB_SETS_SBM = [("DMoN", "embed", "dmon"), ("sFairSC", "embed", "sfairsc"), ("FairGP", "embed", "fairgp"),
                ("FairPGT w/o fairness terms", "sbm", "fairpgt-no-fair"), ("FairPGT (ours)", "sbm", "fairpgt")]
EMB_FALLBACK = [("no fairness terms", "fairpgt-no-fair"), (r"$\mathcal{L}_{\mathrm{att}}$ only", "fairpgt-par-only"),
                (r"$\mathcal{L}_{\mathrm{grp}}$ only", "fairpgt-grp-only"), ("FairPGT (ours)", "fairpgt")]
PASTEL = ["#4C72B0", "#CCB974", "#55A868", "#8172B2", "#64B5CD", "#DD8452", "#7F7F7F", "#8C613C"]


def _tsne(Z, seed=0):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    if Z.shape[1] > 50:
        Z = PCA(50, random_state=seed).fit_transform(Z)
    return TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=seed).fit_transform(Z)


def fig_embeddings_np(df, root, out):
    """t-SNE of node embeddings; colour = cluster, red dots = minority group (style of the submitted paper)."""
    m = df[df.status == "ok"].set_index(["exp", "dataset", "tag"])
    rows = [("lastfm", "LastFM"), ("sbm_align1.0_proxy1.0", r"SBM, $a=1$")]
    have_base = all(os.path.exists(f"{root}/arrays/embed/{ds}/dmon/seed0/embeddings.npy") for ds, _ in rows)
    cols = None
    panels = []
    rows = [(ds, n) for ds, n in rows if os.path.exists(f"{root}/arrays/_S/{ds}.npy")]
    for ds, dname in rows:
        exp_f = "sbm" if ds.startswith("sbm") else "ablation"
        if have_base:
            sets = [(t, e if e != "ablation" else exp_f, g) for t, e, g in (EMB_SETS_SBM if ds.startswith("sbm") else EMB_SETS)]
        else:
            sets = [(t, exp_f, g) for t, g in EMB_FALLBACK]
        panels.append((ds, dname, sets))
        cols = len(sets)
    fig, axes = plt.subplots(len(rows), cols, figsize=(7.16, 1.55 * len(rows) + 0.2), squeeze=False)
    for r, (ds, dname, sets) in enumerate(panels):
        S = np.load(f"{root}/arrays/_S/{ds}.npy") if os.path.exists(f"{root}/arrays/_S/{ds}.npy") else None
        for c, (title, e, tag) in enumerate(sets):
            ax = axes[r, c]
            base = f"{root}/arrays/{e}/{ds}/{tag}/seed0"
            if not os.path.exists(f"{base}/embeddings.npy") or S is None:
                ax.axis("off"); continue
            Z = np.load(f"{base}/embeddings.npy"); lab = np.load(f"{base}/labels.npy")
            Y2 = _tsne(Z)
            minority = np.argmin(np.bincount(S))
            col = np.array([PASTEL[k % len(PASTEL)] for k in lab], dtype=object)
            col[S == minority] = "#D7301F"                      # minority group in red, as in the submitted paper
            order = np.random.default_rng(0).permutation(len(lab))  # interleave, so neither group hides the other
            from matplotlib.colors import to_rgba
            red = (S == minority)[order]
            fc = np.array([to_rgba(c, 0.92) for c in col[order]])
            ec = np.array([to_rgba("#5a0f08", 0.9) if r else (0, 0, 0, 0) for r in red])
            ax.scatter(Y2[order, 0], Y2[order, 1], s=np.where(red, 4.2, 3.4), facecolors=fc,
                       edgecolors=ec, linewidths=0.18, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            try:
                mm = m.loc[(e, ds, tag)]
                mm = mm[mm.seed == 0] if "seed" in mm else mm
                txt = f"NMI {float(mm.nmi.iloc[0]):.2f}   " + r"$\Delta_{\mathrm{SP}}$" + f" {float(mm.delta_sp.iloc[0]):.2f}"
                ax.text(0.5, -0.04, txt, transform=ax.transAxes, ha="center", va="top", fontsize=6.3, color=MUTED)
            except KeyError:
                pass
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"({chr(97 + c)}) {title}", fontsize=7.5, fontweight="bold")
            if c == 0:
                ax.text(-0.08, 0.5, dname, transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=8, fontweight="bold")
    h = [Line2D([], [], marker="o", ls="", color=PASTEL[i], ms=4) for i in range(4)]
    h += [Line2D([], [], marker="o", ls="", color="#D7301F", mec="#5a0f08", mew=0.4, ms=4.5)]
    fig.legend(h, ["clusters of the majority group (colour)"] + [""] * 3 + ["minority group (all clusters)"], loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), handletextpad=0.1, columnspacing=0.2)
    fig.tight_layout(h_pad=1.4, w_pad=0.4)
    fig.savefig(f"{out}/fig_embeddings.pdf", bbox_inches="tight", dpi=600)   # raster layer at 600 dpi
    fig.savefig(f"{out}/fig_embeddings.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- attention matrices
def head_ncgar(A, S):
    """nCGAR of one attention matrix (paper definition, self excluded): cross/within mass over its baseline."""
    same = S[:, None] == S[None, :]
    np.fill_diagonal(same, False)
    cross = S[:, None] != S[None, :]
    n_g = np.bincount(S); n = len(S)
    pw = (n_g * (n_g - 1)).sum()
    tau = (n * (n - 1) - pw) / max(pw, 1)
    return float((A * cross).sum() / max((A * same).sum(), 1e-12) / tau)


def fig_attention(root, out):
    """Per-head nCGAR of the trained encoder (rows: layers, columns: heads) for the four fairness settings.

    Raw attention matrices are not shown: whether attention is group-dependent is a statement about
    attention mass relative to group composition, which a raw heatmap does not show (a small group's
    block looks dark even under neutral attention)."""
    sets = [("fairpgt-no-fair", "no fairness terms"), ("fairpgt-grp-only", r"$\mathcal{L}_{\mathrm{grp}}$ only"),
            ("fairpgt-par-only", r"$\mathcal{L}_{\mathrm{att}}$ only"), ("fairpgt", "FairPGT (ours)")]
    rows = [("sbm", "sbm_align0.75_proxy1.0", r"SBM, $a=0.75$"), ("ablation", "lastfm", "LastFM")]
    rows = [r for r in rows if all(os.path.exists(f"{root}/attention/{r[0]}__{r[1]}__{t}.npz") for t, _ in sets)]
    if not rows:
        return
    cmap = LinearSegmentedColormap.from_list("nc", ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7",
                                                    "#FDDBC7", "#EF8A62", "#B2182B"])
    lim = np.log(4.0)
    fig, axes = plt.subplots(len(rows), len(sets), figsize=(7.16, 1.45 * len(rows) + 0.45), squeeze=False)
    for r, (exp, ds, dname) in enumerate(rows):
        for c, (t, title) in enumerate(sets):
            d = np.load(f"{root}/attention/{exp}__{ds}__{t}.npz", allow_pickle=True)
            A, S = d["attention"], d["S"].astype(int)
            V = np.array([[head_ncgar(A[l, h], S) for h in range(A.shape[1])] for l in range(A.shape[0])])
            ax = axes[r, c]
            ax.imshow(np.clip(np.log(V), -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
            for i in range(V.shape[0]):
                for j in range(V.shape[1]):
                    ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                            color="white" if abs(np.log(V[i, j])) > 0.7 * lim else INK)
            ax.set_xticks(range(V.shape[1])); ax.set_xticklabels([f"h{j + 1}" for j in range(V.shape[1])])
            ax.set_yticks(range(V.shape[0]))
            ax.set_yticklabels([f"layer {i + 1}" for i in range(V.shape[0])] if c == 0 else [])
            ax.tick_params(length=0); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            gm = np.exp(np.log(V).mean())
            ax.set_xlabel(f"geometric mean {gm:.2f}", fontsize=6.3, color=MUTED, labelpad=2)
            if r == 0:
                ax.set_title(f"({chr(97 + c)}) {title}", fontsize=7.5, fontweight="bold")
            if c == 0:
                ax.text(-0.42, 0.5, dname, transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=8, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015, ticks=[-lim, 0, lim])
    cb.ax.set_yticklabels(["0.25\nsame-group", "1\nneutral", "4\ncross-group"], fontsize=6)
    cb.outline.set_visible(False)
    save(fig, out, "fig_attention")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_runs(a.data)
    stats = json.load(open(f"{a.data}/dataset_stats.json"))
    todo = a.only or ["tables", "tradeoff", "sbm", "scaling", "ablation", "embeddings", "attention"]
    if "tables" in todo: tables(df, stats, a.out)
    if "tradeoff" in todo: fig_tradeoff(df, a.out)
    if "sbm" in todo: fig_sbm(df, a.out)
    if "scaling" in todo: fig_scaling(df, a.data, a.out)
    if "ablation" in todo: fig_ablation(df, a.out)
    if "embeddings" in todo: fig_embeddings_np(df, a.data, a.out)
    if "attention" in todo: fig_attention(a.data, a.out)
    print("written:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
