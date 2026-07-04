#!/bin/bash
#SBATCH --job-name=vsteer-safe
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=safety_eval.%j.out
# Does VFD make outputs SAFER on harmful HH prompts? base vs VFD(real head) continuations,
# scored by the INDEPENDENT Ray2333 harmless/helpful reward models (the paper's metric).
# Submit: sbatch --partition=llm,normal,msc safety_eval.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
# Build the node-local env if missing (the scheduler may place us on a node without vsteer);
# else reuse. One-time ~15-20min build per node.
if [ ! -d "$ENVDIR" ]; then
  echo "=== building env at $ENVDIR on $(hostname -s) ==="
  conda create -y -p "$ENVDIR" python=3.12
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  pip install vllm==0.19.1
  pip install -e ".[dev,train]"
else
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install -e ".[dev,train]" -q 2>/dev/null || true
fi
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:?set VALUE_STEER_VHEAD to a trained value head}
export SAFETY_PROMPTS=${SAFETY_PROMPTS:?set SAFETY_PROMPTS to a jsonl of harmful prompts, each {\"prompt\": ...}}
export VALUE_STEER_UTIL=0.45 SAFETY_N=64 VFD_K=8 SAFETY_MAXTOK=128
export ENFORCE_EAGER=1   # FIX: batched (R>1) VFD must run eager -- the compile path is R=1-only
echo "=== gen base (eager) ===";       python scripts/safety_eval.py gen base   || echo "BASE FAILED"
echo "=== gen vfd_t50 (thr 0.5, eager) ==="; VFD_THRESHOLD=0.5 python scripts/safety_eval.py gen vfd_t50 || echo "VFD50 FAILED"
echo "=== gen vfd_t30 (thr 0.3, eager) ==="; VFD_THRESHOLD=0.3 python scripts/safety_eval.py gen vfd_t30 || echo "VFD30 FAILED"
echo "=== score (Ray2333 harmless+helpful) ==="; python scripts/safety_eval.py score base vfd_t50 vfd_t30 || echo "SCORE FAILED"
echo "=== DONE job ${SLURM_JOB_ID} ==="
