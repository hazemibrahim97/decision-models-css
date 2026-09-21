#!/bin/bash
#SBATCH -J dmcss-nimble
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/hi387/decision-models-nimble/logs/%x-%j.out
set -euo pipefail
cd /scratch/hi387/decision-models-nimble
module purge
module load python/3.11.3

export HF_HOME=/scratch/hi387/hf_cache
export PIP_CACHE_DIR=/scratch/hi387/pip_cache

if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
  venv/bin/pip install -q torch==2.8.0 transformers==5.17.0 peft==0.21.0 \
    accelerate==1.15.0 sentencepiece==0.2.2 pillow==12.3.0 huggingface_hub==1.16.1
fi
if [ ! -f nimble-model/inference.py ]; then
  venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('bespokelabs/Bespoke-Nimble-9B', local_dir='nimble-model')"
fi
venv/bin/python run_nimble_jubail.py
rm -rf "$PIP_CACHE_DIR"
