#!/bin/bash
#SBATCH --job-name=vs-canoneval
#SBATCH --partition=llm,normal,msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=canonical_eval.%j.out
# Eval the CANONICAL head (trained/canonical, 17k prompts x4 samples = 68k) on the SAME 256 eval
# prompts/thresholds/vLLM judge as the others. base + 4500-head (vfd_t*) + 17k-head (full_t*) gens
# already live in $SAFETY_OUTDIR; we only gen the canonical head (canon_t*), then judge all together:
# base vs 4500 vs 17k(x1) vs 68k(x4) -- the data-scaling frontier.
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
# Force EVERY temp/cache off the tiny (~8 GB) shared /tmp onto node-local scratch -- set
# UNCONDITIONALLY (not ${VAR:-...}) so an inherited TMPDIR can't shadow it (CLAUDE.md gotcha).
SC=${VS_SCRATCH:-/tmp/$USER}
export TMPDIR="$SC/tmp" TMP="$SC/tmp" TEMP="$SC/tmp"
export PIP_CACHE_DIR="$SC/.cache/pip" CONDA_PKGS_DIRS="$SC/conda_pkgs"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
export PYTHONNOUSERSITE=1
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
if [ ! -f "$ENVDIR/.complete" ]; then
  rm -rf "$ENVDIR"; conda create -y -p "$ENVDIR" python=3.12; conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  pip install vllm==0.19.1; pip install -e ".[dev,train]"; touch "$ENVDIR/.complete"
else
  conda activate "$ENVDIR"; cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}; pip install -e ".[dev,train]" -q
fi
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-${VS_SCRATCH:-/tmp/$USER}/huggingface}

ROOT=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=$ROOT/trained/canonical/value_head.bin
export SAFETY_PROMPTS=$ROOT/data/hh_eval_prompts.jsonl
export SAFETY_OUTDIR=$ROOT/vocab_scalar_eval
export SAFETY_N=256 SAFETY_MAXTOK=128 VALUE_STEER_UTIL=0.45 ENFORCE_EAGER=1 VFD_K=8
export JUDGE_MODEL=${JUDGE_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}

declare -A THR=( [canon_t30]=0.3 [canon_t50]=0.5 [canon_t70]=0.7 [canon_t80]=0.7975 [canon_t83]=0.83 [canon_t90]=0.9 )
CAN=(canon_t30 canon_t50 canon_t70 canon_t80 canon_t83 canon_t90)
for tag in "${CAN[@]}"; do
  echo "=== gen ${tag} (canonical 68k head, threshold=${THR[$tag]}) ==="
  VFD_THRESHOLD=${THR[$tag]} python scripts/safety_eval.py gen "$tag" || echo "${tag} FAILED"
done

ALL="base vfd_t30 vfd_t50 vfd_t70 vfd_t83 vfd_t90 full_t30 full_t50 full_t70 full_t83 full_t90 ${CAN[*]}"
echo "=== score (Ray2333) ==="; python scripts/safety_eval.py score $ALL || echo "SCORE FAILED"
echo "=== judge (vLLM): base | 4500(vfd) | 17k(full) | 68k(canon) ==="; python scripts/safety_eval.py judge $ALL || echo "JUDGE FAILED"
echo "=== DONE job ${SLURM_JOB_ID} ==="
