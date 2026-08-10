#!/bin/bash
#SBATCH --job-name=vsteer-reftest
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=refactor_reftest.%j.out
# Bit-exact behavior-preservation gate for VFD-plugin refactors. Save a reference on the baseline
# commit, then check after each refactor step; identical token ids per config => behavior preserved.
#   REFTEST_MODE=save|check      (default check)
#   REFTEST_CONFIG=main|compiled|churn   (default main; run all three for full coverage)
# Requires VALUE_STEER_VHEAD (value head .bin) and SAFETY_PROMPTS (prompts jsonl).
# Submit: REFTEST_MODE=save REFTEST_CONFIG=main sbatch --partition=llm,normal,msc \
#           examples/research/refactor_reftest.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  mode=${REFTEST_MODE:-check} config=${REFTEST_CONFIG:-main} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
export HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface
# keep the saved reference off the repo tree by default (a per-user, node-local artifact)
export REFTEST_DIR=${REFTEST_DIR:-${VS_SCRATCH:-/tmp/$USER}/reftest}
mkdir -p "$REFTEST_DIR"
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
python examples/research/refactor_reftest.py 2>&1 | grep -E "^###|Error|Traceback|RuntimeError"
echo "=== DONE ==="
