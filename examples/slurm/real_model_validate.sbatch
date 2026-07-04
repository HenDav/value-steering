#!/bin/bash
#SBATCH --job-name=vsteer-realmodel
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=real_model_validate.%j.out
# Validate the plugin with the REAL VFD artifacts: trained scalar safety head on
# mistralai/Mistral-7B-Instruct-v0.3, under the behaviorally-validated vLLM 0.19.1.
# Submit:
#   sbatch --partition=llm,normal,msc examples/slurm/real_model_validate.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
# /tmp is tiny (~8GB) and overflows when pip extracts torch+CUDA wheels -> point pip's temp
# AND cache at the big node-local scratch (the "No space left" was /tmp, not ${VS_SCRATCH:-/tmp/$USER}).
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
# Reclaim node-local scratch: drop leftover throwaway sweep venvs (real env is vsteer-<host>).
rm -rf ${VS_SCRATCH:-/tmp/$USER}/envs/sweep-* ${VS_SCRATCH:-/tmp/$USER}/envs/bsweep-* 2>/dev/null || true
echo "=== ${VS_SCRATCH:-/tmp/$USER} free on $(hostname -s): $(df -h ${VS_SCRATCH:-/tmp/$USER} | tail -1 | awk '{print $4}') ==="
ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/vsteer-$(hostname -s)        # node-local vLLM 0.19.1 env (reuse from gpu_validate)
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
python -c "import vllm;print('vllm',vllm.__version__)"
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:?set VALUE_STEER_VHEAD to a trained value head}
export VALUE_STEER_UTIL=0.45
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
# One section per process so the 7B weights are fully released between (in-process
# vLLM does not free them on shutdown).
echo "=== section: signal (real head load + safety signal) ==="
python scripts/real_model_validate.py signal || echo "SIGNAL SECTION FAILED"
echo "=== section: prefix (VFD greedy == base on Mistral-7B) ==="
python scripts/real_model_validate.py prefix || echo "PREFIX SECTION FAILED"
echo "=== section: speed base ==="
python scripts/real_model_validate.py speed_base || echo "SPEED base FAILED"
echo "=== section: speed vfd K=4 ==="
VFD_K=4 python scripts/real_model_validate.py speed_vfd || echo "SPEED vfd4 FAILED"
echo "=== section: speed vfd K=8 ==="
VFD_K=8 python scripts/real_model_validate.py speed_vfd || echo "SPEED vfd8 FAILED"
echo "=== DONE job ${SLURM_JOB_ID} ==="
