#!/bin/bash
#SBATCH --job-name=vsteer-isocj
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=isolate_compile_judge.%j.out
# Llama-judge the isolation arms (eager-R1 vs compiled-R1, K=16). If unsafe rates match within
# a few/64 -> "compile is noise" holds. Submit AFTER isolate_compile.sbatch finishes.
# Submit: sbatch --partition=llm,normal,msc isolate_compile_judge.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export HF_HOME=${VS_SCRATCH:-/tmp/$USER}/huggingface PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
conda activate "$ENVDIR" 2>/dev/null || { echo "no vsteer env on $(hostname -s); building minimal"; \
  conda create -y -p "$ENVDIR" python=3.12 && conda activate "$ENVDIR" && \
  pip install torch==2.10.0 transformers accelerate; }
cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export TOKENIZERS_PARALLELISM=false
OUT=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/isolate_compile
echo "########## judge arm A: eager-R1 ##########"
SAFETY_OUTDIR="$OUT/eager"    JUDGE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct python scripts/safety_eval.py judge base vfd_k16
echo "########## judge arm B: compiled-R1 ##########"
SAFETY_OUTDIR="$OUT/compiled" JUDGE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct python scripts/safety_eval.py judge base vfd_k16
echo "=== DONE job ${SLURM_JOB_ID} ==="
