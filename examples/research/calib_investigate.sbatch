#!/bin/bash
#SBATCH --job-name=vsteer-calib
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=calib_investigate.%j.out
set -e
echo "=== node: $(hostname) job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface XDG_CACHE_HOME=${VS_SCRATCH:-/tmp/$USER}/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
conda activate ${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
HEAD=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety_decode/value_head.bin
SRC=${VALUE_STEER_DATA:?set VALUE_STEER_DATA to a training-data jsonl}
CACHE=${VS_SCRATCH:-/tmp/$USER}/calib_fc
rm -rf "$CACHE"
echo "### gen+capture decode features (val sample) ###"
python scripts/decode_extract.py --phase gen --cache-dir "$CACHE" --model "$MODEL" --head "$HEAD" \
    --source "$SRC" --n 1500 --max-tokens 96 --gen-chunk 256 --max-num-seqs 16 --seed 99
echo "### judge-label + split ###"
python scripts/decode_extract.py --phase label --cache-dir "$CACHE" --model "$MODEL" \
    --judge-model NousResearch/Meta-Llama-3.1-8B-Instruct --val-split 0.0
echo "### analyze calibration (use train split = all, val-split 0) ###"
python scripts/calib_analyze.py --cache-dir "$CACHE/train" --head "$HEAD"
rm -rf "$CACHE"
echo "=== DONE job ${SLURM_JOB_ID} ==="
