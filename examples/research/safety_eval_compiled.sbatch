#!/bin/bash
#SBATCH --job-name=vsteer-safe-ss
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=safety_eval_compiled.%j.out
# VERIFY THE COMPILED PATH: does VFD make outputs SAFER under the SINGLE-STREAM compile opt-in
# (enforce_eager=False + vfd.single_stream=True + max_num_seqs>=K), driving ONE request at a time
# (R=1 actual -- the only correct compiled regime)? Compares against the EAGER generations from
# the parent dir. Submit: sbatch --partition=llm --gres=gpu:a100:1 examples/research/safety_eval_compiled.sbatch
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
export TMPDIR=${VS_SCRATCH:-/tmp/$USER}/tmp && mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=${VS_SCRATCH:-/tmp/$USER}/.cache/pip
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
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:?set VALUE_STEER_VHEAD to a trained value head}
export VALUE_STEER_UTIL=0.45 SAFETY_N=64 VFD_K=8 SAFETY_MAXTOK=128
export SINGLE_STREAM=1   # compiled single-stream opt-in; do_gen forces enforce_eager=False + sequential R=1
export SAFETY_OUTDIR=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/safety_compiled
mkdir -p "$SAFETY_OUTDIR"
echo "=== gen base (compiled single-stream) ===";        python scripts/safety_eval.py gen base   || echo "BASE FAILED"
echo "=== gen vfd_t30 (thr 0.3, compiled single-stream) ==="; VFD_THRESHOLD=0.3 python scripts/safety_eval.py gen vfd_t30 || echo "VFD30 FAILED"
echo "=== score (Ray2333 harmless+helpful) ==="; python scripts/safety_eval.py score base vfd_t30 || echo "SCORE FAILED"
echo "=== sample compiled VFD outputs (coherence sanity) ==="
python - <<'PY'
import json, os
d = os.environ["SAFETY_OUTDIR"]
for i,l in enumerate(open(os.path.join(d,"safety_gen_vfd_t30.jsonl"))):
    if i>=3: break
    r=json.loads(l); print(f"[{r['index']}] {r['generation'][:240]!r}")
PY
echo "=== DONE job ${SLURM_JOB_ID} ==="
