#!/bin/bash
#SBATCH --job-name=vsteer-coh
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=coherence_check.%j.out
# Pin the VFD degeneration bug: is the diverse-K candidate forward's per-candidate hidden correct?
# Submit: sbatch --partition=llm,normal,msc examples/research/coherence_check.sbatch
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
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:?set VALUE_STEER_VHEAD to a trained value head}
export VALUE_STEER_UTIL=0.45 DIAG_TOKENS=64 VFD_K=8 ENFORCE_EAGER=0 VFD_THRESHOLD=2.0
# compile path; n=R*K. max_cudagraph_capture_size~32: R=4->n=32 (captured), R=8->n=64 (NONE/raw)
echo "=== compile R=1 (n=8, captured) ===";  NUM_PROMPTS=1 MAX_NUM_SEQS=4  python scripts/coherence_check.py || echo F1
echo "=== compile R=4 (n=32, captured) ===";  NUM_PROMPTS=4 MAX_NUM_SEQS=4  python scripts/coherence_check.py || echo F4
echo "=== compile R=8 (n=64, NONE/raw) ===";  NUM_PROMPTS=8 MAX_NUM_SEQS=8  python scripts/coherence_check.py || echo F8
echo "=== DONE job ${SLURM_JOB_ID} ==="
