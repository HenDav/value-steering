#!/bin/bash
#SBATCH --job-name=vsteer-decdiag
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=feature_decode_diag.%j.out
set -e
echo "=== node: $(hostname) job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface XDG_CACHE_HOME=${VS_SCRATCH:-/tmp/$USER}/.cache && mkdir -p "$XDG_CACHE_HOME"
conda activate ${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
python scripts/decode_feature_diag.py \
    --model ${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3} \
    --head ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety_full_v2/value_head.bin \
    --data ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety_full_v2/eval/safety_gen_base.jsonl \
    --n 8 --max-tokens 24
echo "=== DONE job ${SLURM_JOB_ID} ==="
