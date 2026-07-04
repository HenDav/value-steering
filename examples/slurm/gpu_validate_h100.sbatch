#!/bin/bash
#SBATCH --job-name=vsteer-h100
#SBATCH --partition=gh200
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=gpu_validate_h100.%j.out
# Value-steer validation on a free h100 node to dodge a100 contention. Confirms the pinned
# stack (torch 2.10.0+cu128 / vLLM 0.19.1) loads on h100 (CUDA forward-compat), then:
#   1) runs the 5 previously-OOMed tests IN ISOLATION (few engines -> always fits) to prove the
#      vocab runner logic + that the suite failures were engine-accumulation OOM, not bugs;
#   2) runs the FULL suite to test the engine-teardown order fix.
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv

source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${TMPDIR:-${VS_SCRATCH:-/tmp/$USER}/tmp}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${VS_SCRATCH:-/tmp/$USER}/.cache/pip}
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
# ARM/GH200: ~/.local holds x86 wheels (psutil/pygments) that shadow + fail to load on aarch64. Set
# this BEFORE any pip install so pip treats the env as the only site -> installs every dep INTO the
# env (self-complete), and runtime ignores ~/.local. Safe: the node's env has its own transformers.
export PYTHONNOUSERSITE=1

ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)
# Build authority: the .complete marker is written ONLY after a clean build (PYTHONNOUSERSITE=1
# was set above, so pip installs the FULL dependency closure into the env instead of treating
# ~/.local x86 wheels as "already satisfied" and skipping them -- the root cause of the ARM
# missing-dep whack-a-mole). An env without the marker (e.g. one built before this fix) is rebuilt.
if [ ! -f "$ENVDIR/.complete" ]; then
  echo "=== clean build at $ENVDIR (user-site OFF -> full closure; cu128 ARM torch) ==="
  rm -rf "$ENVDIR"
  conda create -y -p "$ENVDIR" python=3.12
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  pip install vllm==0.19.1
  pip install -e ".[dev,train]"
  touch "$ENVDIR/.complete"
else
  echo "=== reusing env at $ENVDIR ==="
  conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install -e ".[dev,train]" -q
fi
echo "=== pip check (systematic: lists ALL unmet deps at once, not import-by-import) ==="
pip check || echo "WARNING: pip check reported unmet deps above"

export VALUE_STEER_TEST_MODEL=facebook/opt-125m
export TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_HOST_IP=127.0.0.1

echo "=== versions (does vLLM 0.19.1 load on h100?) ==="
python -c "import torch,vllm;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0));print('vllm',vllm.__version__)"

echo "=== ISOLATED vocab + prev-OOM tests (few engines, always fits -> clean logic signal) ==="
pytest tests/test_gpu_behavioral.py -q -m gpu -rA -s \
  -k "vocab or test_vfd_kv_ops_captured_match_base or hidden_matches_real_prefix" 2>&1 || true

echo "=== FULL GPU behavioral suite (tests the engine-teardown order fix) ==="
pytest tests/test_gpu_behavioral.py -q -m gpu -rA 2>&1 || true

echo "=== DONE job ${SLURM_JOB_ID} ==="
