#!/bin/bash
#SBATCH --job-name=vsteer-decpipe
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=decode_pipeline.%j.out
# Fix-A experiment: train the value head on DECODE-matched features (the exact hidden VFD scores)
# instead of pooling/prefill features, to close the ~0.97 train/inference gap.
#   gen (VFD never-intervene + capture decode hidden) -> judge-label + split -> train -> [test chained]
# Same node throughout (cache is node-local). Submit: sbatch -p normal,llm -w <env-node> examples/slurm/decode_pipeline.sbatch
set -e
echo "=== node: $(hostname) job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface XDG_CACHE_HOME=${VS_SCRATCH:-/tmp/$USER}/.cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
conda activate ${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false

MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
HEAD0=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety_full_v2/value_head.bin   # any head (feature is head-independent)
SRC=${VALUE_STEER_DATA:?set VALUE_STEER_DATA to a training-data jsonl}
CACHE=${VS_SCRATCH:-/tmp/$USER}/decode_fc
OUT=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety_decode/value_head.bin
N=${DEC_N:-8000}
mkdir -p "$(dirname "$OUT")"; rm -rf "$CACHE"

echo "########## 1. gen + capture DECODE features (N=$N) ##########"
python scripts/decode_extract.py --phase gen --cache-dir "$CACHE" --model "$MODEL" --head "$HEAD0" \
    --source "$SRC" --n "$N" --max-tokens 96 --gen-chunk 256 --max-num-seqs 16 --util 0.45

echo "########## 2. judge-label + split train/val (Llama; gen model freed) ##########"
python scripts/decode_extract.py --phase label --cache-dir "$CACHE" --model "$MODEL" \
    --judge-model NousResearch/Meta-Llama-3.1-8B-Instruct --val-split 0.1

echo "########## 3. train head on decode features (ref recipe: lr1e-4+warmup,batch128,10ep) ##########"
python scripts/train_value_head.py --phase train --out "$OUT" --cache-dir "$CACHE" \
    --domain safety_decode --num-workers 4 --calibrate

echo "=== cleaning cache ($(du -sh "$CACHE" 2>/dev/null|cut -f1)) ==="; rm -rf "$CACHE"
echo "=== DONE job ${SLURM_JOB_ID}: head=$OUT ==="
