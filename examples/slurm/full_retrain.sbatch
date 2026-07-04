#!/bin/bash
#SBATCH --job-name=vsteer-full
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=full_retrain.%j.out
# FULL retrain of the safety value head on ALL 119,860 already-labeled Mistral generations
# (no regeneration -- the data is Mistral outputs + Llama-judge scores). extract -> train ->
# calibrate, as TWO separate processes (the pooling model frees before the head trains; never
# co-resident). The ~255 GB float16 feature cache lives node-local on ${VS_SCRATCH:-/tmp/$USER}
# (needs a node with ample local scratch). Submit: sbatch -p <partition> --gres=gpu:1 examples/slurm/full_retrain.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
# redirect ~/.cache (flashinfer/etc.) to node-local scratch -- $HOME/.cache is flaky on some nodes
export XDG_CACHE_HOME=${VS_SCRATCH:-/tmp/$USER}/.cache && mkdir -p "$XDG_CACHE_HOME"
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
if [ ! -d "$ENVDIR" ]; then
  echo "=== building env at $ENVDIR on $(hostname -s) ==="
  conda create -y -p "$ENVDIR" python=3.12 && conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 && pip install vllm==0.19.1
  pip install -e ".[dev,train]"
else
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install -e ".[dev,train]" -q 2>/dev/null || true
fi
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false

MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
DATA=${VALUE_STEER_DATA:?set VALUE_STEER_DATA to a training-data jsonl}
CACHE=${VS_SCRATCH:-/tmp/$USER}/vsteer_fc_full
OUT=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety_full_v2/value_head.bin
mkdir -p "$(dirname "$OUT")"
rm -rf "$CACHE"        # clear any stale cache

# fail fast if scratch can't hold the cache (~255 GB) -- don't OOM mid-write
AVAIL_G=$(df -BG --output=avail ${VS_SCRATCH:-/tmp/$USER} | tail -1 | tr -dc '0-9')
echo "=== ${VS_SCRATCH:-/tmp/$USER} free: ${AVAIL_G}G (need ~300G for the full-length cache) ==="
[ "${AVAIL_G:-0}" -lt 300 ] && { echo "INSUFFICIENT SCRATCH on $(hostname); resubmit on a node with >300G free"; exit 1; }

echo "########## 1. extract features for ALL examples -> $CACHE (pooling model, then frees) ##########"
python scripts/train_value_head.py --phase extract --model "$MODEL" --data "$DATA" \
    --out "$OUT" --cache-dir "$CACHE" --val-split 0.1 --max-len 1024 --util 0.85 --seed 0

echo "########## 2. train head + calibrate (reference recipe: lr 1e-4 + warmup/decay, batch 128) -> $OUT ##########"
python scripts/train_value_head.py --phase train --out "$OUT" --cache-dir "$CACHE" \
    --domain safety --epochs 10 --batch-size 128 --num-workers 4 \
    --lr 1e-4 --warmup-frac 0.1 --patience 3 --calibrate --seed 0

echo "=== cleaning up $CACHE ($(du -sh "$CACHE" 2>/dev/null | cut -f1)) ==="
rm -rf "$CACHE"
echo "=== DONE job ${SLURM_JOB_ID}: head=$OUT ==="
