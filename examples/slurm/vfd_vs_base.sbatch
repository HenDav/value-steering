#!/bin/bash
#SBATCH --job-name=vsteer-vsb
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=vfd_vs_base.%j.out
# Attribute the VFD-vs-base gap: wall tok/s for base vs VFD(K=1) vs VFD(K=4), PLUS the per-phase
# breakdown (VFD_PROFILE) for the VFD runs. K=1 forward is base-shaped (1 row), so VFD(K=1)-base
# = pure orchestration overhead, and K4-K1 = the K-scaling cost. cudagraphs ON for all (the
# realistic regime). Submit: sbatch --partition=llm,normal,msc vfd_vs_base.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
conda activate "$ENVDIR" 2>/dev/null || { echo "no env on $(hostname -s)"; exit 1; }
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
pip install -e ".[dev,train]" -q 2>/dev/null || true
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:?set VALUE_STEER_VHEAD to a trained value head}
export VALUE_STEER_UTIL=0.45 SPEED_TOKENS=256
echo "=== base (cudagraph) wall tok/s ==="
python scripts/real_model_validate.py speed_base || echo "BASE FAILED"
echo "=== VFD K=1 (cudagraph) wall tok/s + per-phase ==="
VFD_PROFILE=1 VFD_K=1 python scripts/real_model_validate.py speed_vfd || echo "VFD1 FAILED"
echo "=== VFD K=4 (cudagraph) wall tok/s + per-phase ==="
VFD_PROFILE=1 VFD_K=4 python scripts/real_model_validate.py speed_vfd || echo "VFD4 FAILED"
echo "=== DONE job ${SLURM_JOB_ID} ==="
