#!/bin/bash
#SBATCH --job-name=vsteer-testhead
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=test_new_head.%j.out
# Test the freshly-trained value head: base vs VFD(new head) on the held-out harmful-HH test set,
# at the CALIBRATED threshold (from the sidecar) and a fixed 0.3, judged by Llama-3.1-8B.
# Run after train_value_head.sbatch (chained via --dependency). Submit: sbatch --partition=llm,normal,msc test_new_head.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
conda activate "$ENVDIR" 2>/dev/null || { echo "no vsteer env on $(hostname -s); building"; \
  conda create -y -p "$ENVDIR" python=3.12 && conda activate "$ENVDIR" && \
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)} && pip install torch==2.10.0 vllm==0.19.1 && \
  pip install -e ".[dev,train]"; }
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:-${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety/value_head.bin}
export VALUE_STEER_UTIL=0.45 SAFETY_N=64 SAFETY_MAXTOK=128 VFD_K=16 ENFORCE_EAGER=1 SAFETY_SEED=15
export SAFETY_OUTDIR=${SAFETY_OUTDIR:-$(dirname "$VALUE_STEER_VHEAD")/eval}
mkdir -p "$SAFETY_OUTDIR"

# calibrated threshold from the trained head's sidecar
THR=$(python -c "import json;print(json.load(open('${VALUE_STEER_VHEAD}.meta.json'))['threshold'])" 2>/dev/null || echo 0.5)
echo "=== trained head calibrated threshold = $THR ==="

echo "=== gen base ===";                     python scripts/safety_eval.py gen base || echo "BASE FAILED"
echo "=== gen vfd_cal (thr=$THR) ===";        VFD_THRESHOLD=$THR python scripts/safety_eval.py gen vfd_cal || echo "VFD_CAL FAILED"
echo "=== gen vfd_t30 (thr 0.3) ===";        VFD_THRESHOLD=0.3 python scripts/safety_eval.py gen vfd_t30 || echo "VFD_T30 FAILED"
echo "=== score Ray2333 ===";                 python scripts/safety_eval.py score base vfd_cal vfd_t30 || echo "SCORE FAILED"
echo "=== Llama judge (primary metric) ===";  JUDGE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct \
    python scripts/safety_eval.py judge base vfd_cal vfd_t30 || echo "JUDGE FAILED"
echo "=== DONE job ${SLURM_JOB_ID} ==="
