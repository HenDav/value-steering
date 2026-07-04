#!/bin/bash
#SBATCH --job-name=vsteer-gpu
#SBATCH --partition=llm
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=gpu_validate.%j.out
# Value-steer GPU validation on an a100 compute node.
# Builds (once per node) a conda env on the node's LOCAL ${VS_SCRATCH:-/tmp/$USER} -- envs are
# node-local. The value_steer package is installed editable from the NFS repo, so code
# edits are picked up on every run WITHOUT rebuilding. Compute nodes have internet, so
# wheels and the test model download directly.

set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh

# Keep pip's wheel extraction off the tiny (~8 GB) /tmp -> node-local scratch (CLAUDE.md gotcha).
export TMPDIR=${TMPDIR:-${VS_SCRATCH:-/tmp/$USER}/tmp}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${VS_SCRATCH:-/tmp/$USER}/.cache/pip}
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# One reusable env per node (hostname-keyed). Build only if missing.
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
if [ ! -d "$ENVDIR" ]; then
  echo "=== building env at $ENVDIR (node-local, one-time) ==="
  conda create -y -p "$ENVDIR" python=3.12
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  pip install vllm==0.19.1
  pip install -e ".[dev,train]"
else
  echo "=== reusing env at $ENVDIR ==="
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install -e ".[dev,train]" -q   # ensure editable link present (no-op if already)
fi

export VALUE_STEER_TEST_MODEL=facebook/opt-125m   # downloads on first use (node has internet)
export TOKENIZERS_PARALLELISM=false
export VFD_DEBUG=1                                  # verbose VFD position/block tracing
export VLLM_ENABLE_V1_MULTIPROCESSING=0             # in-process engine: reach the live runner; no forked-CUDA crash
export VLLM_HOST_IP=127.0.0.1                       # avoid IPv6 hostname rendezvous stall (c10d errno 97)

echo "=== versions ==="
python -c "import torch,vllm;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('vllm',vllm.__version__)"

echo "=== CPU suite (no gpu marker) ==="
pytest tests/ -q -m "not gpu" || true

echo "=== value-steer-compat (static contracts) ==="
value-steer-compat || true

echo "=== GPU behavioral suite (the gate: abstention + VFD) ==="
pytest tests/test_gpu_behavioral.py -q -m gpu -rA -s 2>&1 || true

echo "=== full GPU diagnostic (verbose, supplementary) ==="
python scripts/gpu_diag.py 2>&1 || true

echo "=== DONE job ${SLURM_JOB_ID} ==="
