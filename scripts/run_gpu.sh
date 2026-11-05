#!/usr/bin/env bash
# Run one experiment config on one GPU, logging to logs/.  Start it inside tmux (see README).
# Usage: bash scripts/run_gpu.sh <config> <gpu-id> [extra args, e.g. --datasets nba german]
set -e
CFG=$1; GPU=$2; shift 2
mkdir -p logs
NAME=$(basename "$CFG" .yaml)_gpu${GPU}_$(date +%m%d_%H%M)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fgc
CUDA_VISIBLE_DEVICES=$GPU python -m experiments.run "$CFG" --device cuda "$@" 2>&1 | tee logs/$NAME.log
