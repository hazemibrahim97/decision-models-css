#!/bin/bash
#SBATCH -J dmcss-opendecision
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/hi387/decision-models-kev/logs/%x-%j.out
set -euo pipefail
cd /scratch/hi387/decision-models-kev
module purge
module load miniconda/3-4.11.0
export TORCH_DISABLE_NATIVE_JIT=1  # no C toolchain on nodes; eager op instead of triton JIT (ModernBERT RoPE bmm)
export HF_HOME=/scratch/hi387/hf_cache
export PIP_CACHE_DIR=/scratch/hi387/pip_cache
export PYTHONNOUSERSITE=1
ENV=/scratch/hi387/conda-envs/od313
trap 'rm -rf "$PIP_CACHE_DIR"' EXIT
[ -x $ENV/bin/python ] || conda create -y -q -p $ENV python=3.13
$ENV/bin/python -c 'import opendecision' 2>/dev/null || $ENV/bin/pip install -q OpenDecision==0.1.1
$ENV/bin/python -c 'import torch; assert torch.cuda.is_available()'
$ENV/bin/python run_local_ecosystem.py opendecision
