#!/bin/bash
# SLURM template. Submit from the repository root, e.g.
#   sbatch -p <partition> scripts/slurm.sbatch experiments/configs/main.yaml --datasets nba german lastfm
#SBATCH --job-name=fgc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

eval "$(conda shell.bash hook)"
conda activate fgc
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:$PYTHONPATH"
echo "node $(hostname)  gpu $CUDA_VISIBLE_DEVICES  $(date)"
python -m experiments.run "$@" --device cuda
