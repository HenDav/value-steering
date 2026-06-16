#!/bin/bash
#SBATCH --job-name=vsteer-speedK
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=speed_K.%j.out
# Confirm "flat in K": decode tok/s for base vs VFD compiled single-stream at K=8 and K=40
# (max_num_seqs=64 so K=40 captures; scratch stays K*1). If K=40 ~= K=8 tok/s, the candidate
# forward is bandwidth-bound at 40 rows and higher K is ~free on latency.
# Submit: sbatch --partition=llm speed_K.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
conda activate "$ENVDIR" 2>/dev/null || { echo "no vsteer env on $(hostname -s); building"; \
  conda create -y -p "$ENVDIR" python=3.12 && conda activate "$ENVDIR" && \
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 && pip install vllm==0.19.1; }
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
pip install -e ".[dev,train]" -q 2>/dev/null || true
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:?set VALUE_STEER_VHEAD to a trained value head}
export VALUE_STEER_UTIL=0.45 SPEED_TOKENS=256
export SINGLE_STREAM=1 SPEED_MAX_NUM_SEQS=64   # compiled single-stream; mns>=40 so K=40 captures
echo "=== base [cudagraph] ===";                 python scripts/real_model_validate.py speed_base || echo "BASE FAILED"
echo "=== VFD K=8  [cudagraph,single-stream] ==="; VFD_K=8  python scripts/real_model_validate.py speed_vfd || echo "K8 FAILED"
echo "=== VFD K=40 [cudagraph,single-stream] ==="; VFD_K=40 python scripts/real_model_validate.py speed_vfd || echo "K40 FAILED"
echo "=== (ref) VFD K=8 EAGER ===";  SPEED_ENFORCE_EAGER=1 SINGLE_STREAM=0 VFD_K=8  python scripts/real_model_validate.py speed_vfd || true
echo "=== DONE job ${SLURM_JOB_ID} ==="
