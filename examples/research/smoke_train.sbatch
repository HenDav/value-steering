#!/bin/bash
#SBATCH --job-name=vsteer-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=smoke_train.%j.out
# Cheap GPU gate for the value-head training pipeline: the train->save->load->separate->calibrate
# smoke test on a tiny model, plus the pure verifier tests. Submit: sbatch --partition=llm examples/research/smoke_train.sbatch
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
export VALUE_STEER_TEST_MODEL=facebook/opt-125m
echo "=== CPU verifier tests ==="
python -m pytest tests/test_verifiers.py -q
echo "=== GPU training smoke test ==="
python -m pytest tests/test_gpu_smoke.py -q -m gpu
echo "=== DONE job ${SLURM_JOB_ID} ==="
