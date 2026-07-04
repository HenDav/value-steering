#!/bin/bash
#SBATCH --job-name=vsteer-profK
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=profile_K.%j.out
# WHERE does the K=8->K=40 slowdown come from? Per-phase CUDA-event timing (VFD_PROFILE=1):
# meta/copy/forward/score/commit/seed ms per step, at K=8 vs K=40 compiled single-stream.
# Submit: sbatch --partition=llm examples/research/profile_K.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
conda activate "$ENVDIR"
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
pip install -e ".[dev,train]" -q 2>/dev/null || true
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:?set VALUE_STEER_VHEAD to a trained value head}
export VALUE_STEER_UTIL=0.45 SPEED_TOKENS=256
export SINGLE_STREAM=1 SPEED_MAX_NUM_SEQS=64 VFD_PROFILE=1 VFD_PROFILE_EVERY=128
echo "=== VFD K=8  per-phase ms/step ==="; VFD_K=8  python scripts/real_model_validate.py speed_vfd 2>&1 | grep -E "VFD-PROFILE|SPEED" || echo K8FAIL
echo "=== VFD K=40 per-phase ms/step ==="; VFD_K=40 python scripts/real_model_validate.py speed_vfd 2>&1 | grep -E "VFD-PROFILE|SPEED" || echo K40FAIL
echo "=== DONE job ${SLURM_JOB_ID} ==="
