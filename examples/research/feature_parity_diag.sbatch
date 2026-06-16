#!/bin/bash
#SBATCH --job-name=vsteer-parity
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=feature_parity_diag.%j.out
# Diagnose the weak full-data head: (1) feature parity HF-vs-vLLM-pooling, (2) head AUROC on both
# feature sources. Tells us if it's a train/inference feature mismatch or a training-recipe gap.
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface
conda activate ${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
python scripts/feature_parity_diag.py \
    --model ${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3} \
    --data ${SAFETY_PROMPTS:?set SAFETY_PROMPTS to a prompts jsonl} \
    --n 128 --head-ours trained/safety_full/value_head.bin
echo "=== DONE job ${SLURM_JOB_ID} ==="
