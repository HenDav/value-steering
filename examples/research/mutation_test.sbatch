#!/bin/bash
#SBATCH --job-name=vsteer-mut
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=mutation_test.%j.out
# Prove the GPU behavioral suite catches the two VFD prefix bugs (revert each fix ->
# the strict greedy test must fail; restored -> pass). Submit across a100 partitions:
#   sbatch --partition=llm,normal,msc mutation_test.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
if [ ! -d "$ENVDIR" ]; then
  conda create -y -p "$ENVDIR" python=3.12
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  pip install vllm==0.19.1
  pip install -e ".[dev,train]"
else
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install -e ".[dev,train]" -q
fi
export VALUE_STEER_TEST_MODEL=facebook/opt-125m
export TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_HOST_IP=127.0.0.1
python scripts/mutation_test.py
echo "=== DONE job ${SLURM_JOB_ID} ==="
