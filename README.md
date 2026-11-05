# FairPGT: Partitioned Graph Transformers for Dual-Level Fair Graph Clustering

Official code of the paper *FairPGT: Partitioned Graph Transformers for Dual-Level Fair
Graph Clustering* (S. Ghodsi, A. Seyedi, S. Tahmasebi, E. Ntoutsi, S. Auer).

FairPGT clusters the nodes of an attributed graph with a partitioned graph transformer and
imposes fairness at two levels: a mutual-information term makes the clusters independent of
the protected attribute, and an nCGAR self-distillation term keeps the encoder's attention
balanced across groups. The protected attribute is used only in the training objective;
inference uses the graph and the features alone.

The repository contains FairPGT, all 13 baselines under one protocol, and the configuration
of every experiment in the paper.

## Layout

```
fgc/
  models/      fairpgt.py (encoder + DEC head), losses.py (all loss terms), experiment.py (training)
  baselines/   base.py (common interface and protocol) and the 13 baselines
  data/        datasets.py (loading), sbm.py (attributed SBM generator), preprocess.py (spectral encodings, METIS)
  eval/        metrics.py (NMI, macro-F1, modularity, ΔSP, ...), probe.py (S-probe)
  registry.py  method name -> class and fixed default hyperparameters
  runner.py    one run = (method, dataset, seed) -> runs/<exp>/<dataset>/<tag>/seed<k>/
experiments/
  configs/     main, sbm, ablation, sensitivity, scaling, embed (one per experiment of the paper)
  run.py       runs a config
  report.py    mean ± std tables (also LaTeX)
  make_paper_figs.py  all tables and figures of the paper
scripts/       environment setup, GPU/tmux and SLURM launchers, export of figure data
tools/         the toy example of Figure 1
tests/         unit tests (python -m pytest -q tests)
```

## Installation

```bash
conda create -n fgc python=3.12 && conda activate fgc
pip install torch --index-url https://download.pytorch.org/whl/cu128   # choose your CUDA build
pip install -r requirements.txt
python -m pytest -q tests
```
or simply `bash scripts/setup_env.sh`. We used Python 3.12 and PyTorch 2.11.

## Data

Put the preprocessed datasets in `data/` (or set `FGC_DATA` to their location):

```
data/nba/cleaned/            A.npy X.npy s.npy y_binary.npy
data/german/german_knn/      german_A.npz german_X.npz german_S_gender.npy german_Y.npy
data/lastfm/cleaned/         A_lastfm.npz X_lfm_svd_128_embed.npy sensitive_featkmeans.npy labels.npy
data/facebook_large/cleaned/ adjacency.npz features.npz sensitive_featkmeans.npy labels.npy
data/credit/cleaned/         adjacency.npz features.npy sensitive.npy labels.npy
data/pokec_n/cleaned/        pokec_A.npz pokec_X_svd_128.npy pokec_S.npy pokec_Y_binary.npy
data/pokec_z/cleaned/        (same file names as pokec_n)
```

The preprocessed files are available at **[Zenodo DOI]**. They are derived from public datasets;
please also cite the original sources:

| Dataset | Source | Protected attribute | Label |
|---|---|---|---|
| NBA, Pokec-n, Pokec-z | Dai & Wang, WSDM 2021 ([FairGNN](https://github.com/EnyanDai/FairGNN)) | nationality; age (4 bins) | salary; working field (binarized) |
| German | UCI Statlog (German Credit); symmetric 10-NN graph built by us | gender | credit risk |
| Credit | UCI default of credit card clients; graph of Agarwal et al., UAI 2021 ([NIFTY](https://github.com/chirag126/nifty)) | marital status | default |
| LastFM | Rozemberczki & Sarkar, CIKM 2020 ([SNAP](https://snap.stanford.edu/data/feather-lastfm-social.html)) | k-means on features (k=2) | country (6 largest) |
| Facebook | Rozemberczki et al., J. Complex Netw. 2021 ([SNAP](https://snap.stanford.edu/data/facebook-large-page-page-network.html)) | k-means on features (k=2) | page type |

The SBMs of Section 4.3 are generated on the fly (`fgc/data/sbm.py`); no download is needed.
Labels are used only for evaluation (`fgc/eval/metrics.py`); `fit()` never receives them.

## Reproducing the paper

Each experiment is one config; runs are written to `runs/<exp>/` and finished runs are
skipped on restart.

| Paper | Command |
|---|---|
| Tables 3–5, Figure 3 (Q1) | `python -m experiments.run experiments/configs/main.yaml` |
| Figures 3h, 5, 6 (Q2, SBMs) | `python -m experiments.run experiments/configs/sbm.yaml` |
| Figures 6, 7 (Q3, ablation) | `python -m experiments.run experiments/configs/ablation.yaml` |
| Figure 3 λ-trace, Section 4.5 | `python -m experiments.run experiments/configs/sensitivity.yaml` |
| Figure 8 (partitions) | `python -m experiments.run experiments/configs/scaling.yaml` |
| Figure 4 (embeddings) | `python -m experiments.run experiments/configs/embed.yaml` |
| Figure 1 (toy example) | `python tools/teaser_fig.py` |

Useful options: `--datasets nba german`, `--models fairpgt dmon`, `--seeds 0`, `--device cuda`.
On several GPUs, run one process per GPU on different datasets; all may write to the same
`runs/` folder (`bash scripts/run_gpu.sh <config> <gpu-id> --datasets ...`, or
`sbatch scripts/slurm.sbatch <config> ...` on SLURM).

Tables and figures:
```bash
python -m experiments.report runs/main --latex                 # mean ± std tables
python scripts/export_paper_data.py --runs runs --out paper_data --device cuda
python experiments/make_paper_figs.py paper_data paper_out     # every table and figure
```

## Protocol

* One fixed hyperparameter setting per method across all datasets (`fgc/registry.py`,
  `fgc/models/experiment.py:DEFAULTS`); nothing is tuned on labels.
* 200 epochs for every method; the reported clustering is the last epoch.
* K = number of label classes; five seeds (three on the SBMs).
* A run is *collapsed* if it uses fewer than K clusters or puts more than 90% of the nodes
  into one cluster.

FairPGT options map to the paper as follows: `lambda_grp` = λ_grp, `lambda_par` = λ_att,
`lambda_info` = λ_info, `ncgar_alpha` = α, `ncgar_eps` = ε, `target_every` = T_P.

## Baselines

DMoN, SDCN, CCGC, SCGC, DinkNet, DiffPool and MinCutPool are ported from their reference
implementations. Adaptations (described in Section 4.1 of the paper):
* `dmon_sp`: DMoN with a statistical-parity penalty (weight 0.1).
* `fairgt`, `fairgp`: the encoders of FairGT and FairGP with a DEC head and the same parity
  penalty (both papers address node classification); FairGT receives the protected attribute.
* `ccgcn` (CCGC): GCN encoders in place of the filtered-feature MLPs.
* `sfairsc`, `ifairsc`: re-implemented from the papers; sFairSC on the regularized graph
  A + (τ/n)11ᵀ, iFairSC with a symmetric 10-NN representation graph and the low-rank
  approximation when the constraint is infeasible.

## Citation

```bibtex
@article{ghodsi2026fairpgt,
  title   = {FairPGT: Partitioned Graph Transformers for Dual-Level Fair Graph Clustering},
  author  = {Ghodsi, Siamak and Seyedi, Amjad and Tahmasebi, Sahar and Ntoutsi, Eirini and Auer, S{\"o}ren},
  journal = {Neural Networks},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT (see `LICENSE`). Baselines ported from other repositories keep the licenses of their
original code.
