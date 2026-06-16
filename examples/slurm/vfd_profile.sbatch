#!/bin/bash
#SBATCH --job-name=vsteer-prof
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=vfd_profile.%j.out
# Per-phase profile of VFD's steady decode step (copy/forward/score/commit/seed) under
# cudagraphs, to TARGET further optimization. Runs opt-125m (orchestration upper bound) then
# Mistral-7B (real target). Submit: sbatch --partition=llm,normal,msc vfd_profile.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
if [ ! -d "$ENVDIR" ]; then
  conda create -y -p "$ENVDIR" python=3.12
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  pip install vllm==0.19.1
  pip install -e ".[dev,train]"
else
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install -e ".[dev,train]" -q
fi
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
export VFD_PROFILE=1 VFD_PROFILE_EVERY=64

echo "=== profile: opt-125m (orchestration UPPER bound) ==="
VALUE_STEER_TEST_MODEL=facebook/opt-125m python scripts/vfd_profile.py || echo "OPT PROFILE FAILED"

echo "=== profile: Mistral-7B (real target) ==="
VALUE_STEER_TEST_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3} \
  python scripts/vfd_profile.py || echo "MISTRAL PROFILE FAILED"

echo "=== DONE job ${SLURM_JOB_ID} ==="
