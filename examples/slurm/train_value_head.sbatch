#!/bin/bash
#SBATCH --job-name=vsteer-train
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=train_value_head.%j.out
# One-command value-head pipeline: generate responses -> verify-label -> train -> calibrate ->
# write checkpoint+sidecar. Domain-pluggable via DOMAIN (default safety). The head predicts
# P(undesirable); VFD then commits the lowest-scoring candidate (unchanged across domains).
#   Submit: sbatch --partition=llm train_value_head.sbatch
#   Override: DOMAIN=safety SAFETY_N=512 SPP=2 sbatch --partition=llm train_value_head.sbatch
# (math/code verifiers are stubs in this pass -- safety is the working end-to-end domain.)
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
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

DOMAIN=${DOMAIN:-safety}
MODEL=${MODEL:-${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}}
SOURCE=${SOURCE:-${VALUE_STEER_DATA:?set VALUE_STEER_DATA to a training-data jsonl}}
N=${SAFETY_N:-512}
SPP=${SPP:-2}                       # samples per prompt (>1 balances good/bad labels)
JUDGE=${JUDGE_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}
WORK=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/trained/${DOMAIN}
DATA=$WORK/data.jsonl
OUT=$WORK/value_head.bin
CACHE=$WORK/featcache
mkdir -p "$WORK"

# Each step is a SEPARATE process, so only ONE big model is resident at a time (generator, then
# judge, then the pooling extractor) -- none co-resident, each gets the full GPU. The head trains
# in its own model-free process, streaming features from the on-disk cache (big-dataset safe).
echo "########## 1. generate responses ($DOMAIN, n=$N, spp=$SPP) ##########"
python scripts/gen_value_data.py --domain "$DOMAIN" --phase gen --model "$MODEL" \
    --source "$SOURCE" --out "$DATA" --n "$N" --samples-per-prompt "$SPP" --max-tokens 128

echo "########## 2. verify-label (judge model; generator already freed) ##########"
python scripts/gen_value_data.py --domain "$DOMAIN" --phase verify --out "$DATA" --judge-model "$JUDGE"

echo "########## 3. extract features (pooling model -> on-disk cache, then frees) ##########"
python scripts/train_value_head.py --phase extract --model "$MODEL" --data "$DATA" \
    --out "$OUT" --cache-dir "$CACHE" --domain "$DOMAIN"

echo "########## 4. train head + calibrate (NO model loaded) -> $OUT (+ .meta.json) ##########"
python scripts/train_value_head.py --phase train --out "$OUT" --cache-dir "$CACHE" \
    --domain "$DOMAIN" --epochs 3 --calibrate

echo "=== DONE job ${SLURM_JOB_ID}: head=$OUT ==="
