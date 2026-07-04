#!/bin/bash
#SBATCH --job-name=vs-fulltest
#SBATCH --partition=llm,normal,msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=canonical_eval_full.%j.out
# Definitive eval on the FULL held-out set: ALL harmless-base TEST prompts (2178 unique, never seen
# in the 17k train). Data-scaling ladder base vs 4500 vs 17k(x1) vs 68k(canonical, 17k x4), same
# thresholds + vLLM judge. n=2178 -> binomial SE ~0.01 (resolves the ~0.04 gap n=256 couldn't).
set -e
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
# Force ALL temp/cache onto node-local scratch, UNCONDITIONALLY (inherited TMPDIR -> /tmp ENOSPC).
SC=${VS_SCRATCH:-/tmp/$USER}
export TMPDIR="$SC/tmp" TMP="$SC/tmp" TEMP="$SC/tmp"
export PIP_CACHE_DIR="$SC/.cache/pip" CONDA_PKGS_DIRS="$SC/conda_pkgs"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
export PYTHONNOUSERSITE=1
ENVDIR="$SC/envs/vsteer-$(hostname -s)"
if [ ! -f "$ENVDIR/.complete" ]; then
  rm -rf "$ENVDIR"; conda create -y -p "$ENVDIR" python=3.12; conda activate "$ENVDIR"
  cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  pip install vllm==0.19.1; pip install -e ".[dev,train]"; touch "$ENVDIR/.complete"
else
  conda activate "$ENVDIR"; cd ${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}; pip install -e ".[dev,train]" -q
fi
pip install -q datasets
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$SC/huggingface"

ROOT=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export SAFETY_PROMPTS=$ROOT/data/hh_test_all.jsonl
export SAFETY_OUTDIR=$ROOT/full_test_eval; mkdir -p "$SAFETY_OUTDIR"
export SAFETY_N=5000 SAFETY_MAXTOK=128 VALUE_STEER_UTIL=0.45 ENFORCE_EAGER=1 VFD_K=8
export JUDGE_MODEL=${JUDGE_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}

echo "=== prep ALL harmless-base TEST prompts ==="
python - "$SAFETY_PROMPTS" <<'PY'
import json, sys
from datasets import load_dataset
out = sys.argv[1]
def fh(t):
    p = t.split("\n\nHuman:")
    return (p[1].split("\n\nAssistant:")[0]).strip() if len(p) > 1 else None
ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
seen = set()
with open(out, "w") as w:
    for ex in ds:
        x = fh(ex["chosen"])
        if x and x not in seen:
            seen.add(x); w.write(json.dumps({"prompt": x}) + "\n")
print(f"# wrote {len(seen)} test prompts -> {out}", flush=True)
PY

gen_with () {  # $1 = head checkpoint, rest = "tag:thr" pairs
  export VALUE_STEER_VHEAD="$1"; shift
  for pair in "$@"; do tag="${pair%%:*}"; thr="${pair##*:}";
    echo "=== gen $tag (VHEAD=$(basename $(dirname "$VALUE_STEER_VHEAD")), thr=$thr) ==="
    VFD_THRESHOLD=$thr python scripts/safety_eval.py gen "$tag" || echo "$tag FAILED"
  done
}

echo "=== gen base ==="; python scripts/safety_eval.py gen base || echo "base FAILED"
gen_with "$ROOT/trained/scalar_decode_same/value_head.bin" vfd_t30:0.3 vfd_t50:0.5 vfd_t70:0.7 vfd_t83:0.83 vfd_t90:0.9
gen_with "$ROOT/trained/scalar_full/value_head.bin"        full_t30:0.3 full_t50:0.5 full_t70:0.7 full_t83:0.83 full_t90:0.9
gen_with "$ROOT/trained/canonical/value_head.bin"          canon_t30:0.3 canon_t50:0.5 canon_t70:0.7 canon_t80:0.7975 canon_t83:0.83 canon_t90:0.9

ALL="base vfd_t30 vfd_t50 vfd_t70 vfd_t83 vfd_t90 full_t30 full_t50 full_t70 full_t83 full_t90 canon_t30 canon_t50 canon_t70 canon_t80 canon_t83 canon_t90"
echo "=== score (Ray2333) ==="; python scripts/safety_eval.py score $ALL || echo "SCORE FAILED"
echo "=== judge (vLLM) ==="; python scripts/safety_eval.py judge $ALL || echo "JUDGE FAILED"
echo "=== DONE job ${SLURM_JOB_ID} (n=$(wc -l < $SAFETY_PROMPTS)) ==="
