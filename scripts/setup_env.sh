#!/usr/bin/env bash
# One-time setup of the conda environment "fgc" (run on a machine with a GPU).
#   bash scripts/setup_env.sh            # default: CUDA 12.8 wheels of PyTorch
#   CUDA=cu121 bash scripts/setup_env.sh # other CUDA builds: see pytorch.org
set -e
eval "$(conda shell.bash hook)"
conda create -y -n fgc python=3.12
conda activate fgc
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA:-cu128}"
pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available())"
export PYTHONPATH=".:$PYTHONPATH"
python -m pytest -q tests
