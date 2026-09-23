#!/bin/bash
#SBATCH -J dmcss-kev9b
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/hi387/decision-models-kev/logs/%x-%j.out
set -euo pipefail
cd /scratch/hi387/decision-models-kev
module purge
module load miniconda/3-4.11.0 gcc/13.2.0
export CC=$(which gcc)
export TRITON_CACHE_DIR=/scratch/hi387/triton_cache
export HF_HOME=/scratch/hi387/hf_cache
export PIP_CACHE_DIR=/scratch/hi387/pip_cache
export PYTHONNOUSERSITE=1
ENV=/scratch/hi387/conda-envs/kev312
MODEL=${1:-kev-9b}   # usage: sbatch -J dmcss-<model> kev_job.sbatch <model> <port>
PORT=${2:-8019}
trap 'kill $SERVER 2>/dev/null || true; rm -rf "$PIP_CACHE_DIR"' EXIT

if [ ! -x $ENV/bin/python ]; then
  conda create -y -q -p $ENV python=3.12
fi
if [ ! -d kev ]; then git clone -q https://github.com/jaredpalmer/kev.git && git -C kev checkout -q 557598f; fi
$ENV/bin/python -c 'import kev' 2>/dev/null || $ENV/bin/pip install -q -e './kev[serve]'
$ENV/bin/pip uninstall -y -q flash-linear-attention fla-core || true  # its Triton kernels fail to build on jubail; transformers' torch path

$ENV/bin/python -c "import importlib.util as u; assert u.find_spec(\"fla\") is None"
$ENV/bin/python -m kev.serve --run jaredpalmer/$MODEL --port $PORT > logs/server-$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
until curl -sf localhost:$PORT/v1/models > /dev/null; do sleep 10; kill -0 $SERVER; done
curl -s localhost:$PORT/v1/models > logs/models-$MODEL-$SLURM_JOB_ID.json
$ENV/bin/python run_local_ecosystem.py $MODEL --url http://127.0.0.1:$PORT
