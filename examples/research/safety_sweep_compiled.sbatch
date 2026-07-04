#!/bin/bash
#SBATCH --job-name=vsteer-sweepKc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=safety_sweep_compiled.%j.out
# Sweep K in {8,16,24,40} on the harmful-HH safety eval (EAGER -- correct for all K, no compile
# capture ceiling; safety effect is identical to compiled, already proven). thr 0.3, 64 prompts.
# Reports argmin-fallback rate (does higher K reduce off-distribution forcing?) + Ray2333
# harmless/helpful per K. Llama-judge unsafe rate = safety_sweep_judge.sbatch afterward.
# Submit: sbatch --partition=llm examples/research/safety_sweep_compiled.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
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
export VALUE_STEER_UTIL=0.45 SAFETY_N=64 SAFETY_MAXTOK=128
export VFD_THRESHOLD=0.3 SINGLE_STREAM=1 SAFETY_MAX_NUM_SEQS=64
export SAFETY_OUTDIR=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/safety_sweep_compiled
mkdir -p "$SAFETY_OUTDIR"
echo "=== gen base (eager, K-independent) ==="; python scripts/safety_eval.py gen base || echo "BASE FAILED"
for K in 8 16 24 40; do
  echo "=== gen vfd_k${K} (thr 0.3, K=${K}, eager) ==="
  VFD_K=$K python scripts/safety_eval.py gen "vfd_k${K}" || echo "VFD_K${K} FAILED"
done
echo "=== score (Ray2333 harmless+helpful) ==="
python scripts/safety_eval.py score base vfd_k8 vfd_k16 vfd_k24 vfd_k40 || echo "SCORE FAILED"
echo "=== DONE job ${SLURM_JOB_ID} ==="
