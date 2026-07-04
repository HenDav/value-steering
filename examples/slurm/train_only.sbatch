#!/bin/bash
#SBATCH --job-name=vsteer-trainonly
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=train_only.%j.out
# Train-only (reuses the existing node-local feature cache from a prior full run -- NO
# re-extraction). Faster data loading (--num-workers) + per-epoch loss curve. MUST run on the SAME
# node that holds the cache. Submit: sbatch -p <partition> --gres=gpu:1 -w <that-node> examples/slurm/train_only.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
conda activate ${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false

CACHE=${VS_SCRATCH:-/tmp/$USER}/vsteer_fc_full
OUT=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/safety_full_v2/value_head.bin
mkdir -p "$(dirname "$OUT")"
[ -f "$CACHE/train/meta.json" ] || { echo "MISSING cache at $CACHE on $(hostname); aborting"; exit 1; }
echo "=== reusing cache $CACHE ($(du -sh "$CACHE" | cut -f1)) ==="

# reference recipe: lr 1e-4 + linear warmup/decay, batch 128, 10 epochs (the diff from the weak head)
python scripts/train_value_head.py --phase train --out "$OUT" --cache-dir "$CACHE" \
    --domain safety --epochs 10 --batch-size 128 --num-workers 4 \
    --lr 1e-4 --warmup-frac 0.1 --patience 3 --calibrate --seed 0
echo "=== DONE job ${SLURM_JOB_ID}: head=$OUT  (cache kept at $CACHE -- rm when done) ==="
