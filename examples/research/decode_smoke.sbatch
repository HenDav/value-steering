#!/bin/bash
#SBATCH --job-name=vsteer-decsmoke
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=decode_smoke.%j.out
set -e
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface XDG_CACHE_HOME=${VS_SCRATCH:-/tmp/$USER}/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
conda activate ${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
python scripts/decode_extract.py --phase gen --cache-dir ${VS_SCRATCH:-/tmp/$USER}/decode_smoke \
  --model ${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3} \
  --head ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety_full_v2/value_head.bin \
  --source ${SAFETY_PROMPTS:?set SAFETY_PROMPTS to a prompts jsonl} \
  --n 128 --max-tokens 32 --gen-chunk 64
echo "=== cache check ==="; ls -la ${VS_SCRATCH:-/tmp/$USER}/decode_smoke/ 2>/dev/null
echo "=== DONE smoke ==="
