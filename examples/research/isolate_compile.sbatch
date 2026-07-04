#!/bin/bash
#SBATCH --job-name=vsteer-isoc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=isolate_compile.%j.out
# NOISE-vs-BUG test of the claim "compiling should not affect safety beyond noise".
# Isolates COMPILE from the R (batch-size) confound: BOTH arms drive ONE request at a time
# (SINGLE_STREAM=1 -> R=1 actual), SAME seed, K=16, thr 0.3, 64 prompts. The ONLY difference is
# enforce_eager (compile off vs on):
#   arm A = eager-R1     (ENFORCE_EAGER=1, single-stream submission, no compile)
#   arm B = compiled-R1  (ENFORCE_EAGER=0, vfd.single_stream=True, cudagraph replay)
# If A and B match within sampling noise (~few/64) -> claim holds (earlier sweep gap was the
# eager-batched-R64 vs compiled-R1 confound). If they diverge systematically -> real compile bug.
# Also dumps a PAIRED per-prompt token-divergence report (identical seqs / first-diff step).
# Llama-judge afterward: isolate_compile_judge.sbatch.  Submit: sbatch --partition=llm examples/research/isolate_compile.sbatch
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
export VALUE_STEER_UTIL=0.45 SAFETY_N=64 SAFETY_MAXTOK=128
export VFD_THRESHOLD=0.3 VFD_K=16 SINGLE_STREAM=1 SAFETY_MAX_NUM_SEQS=64
OUT=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/isolate_compile
mkdir -p "$OUT/eager" "$OUT/compiled"

echo "########## arm A: eager-R1 (ENFORCE_EAGER=1, single-stream submission) ##########"
SAFETY_OUTDIR="$OUT/eager"   ENFORCE_EAGER=1 python scripts/safety_eval.py gen base   || echo "A base FAILED"
SAFETY_OUTDIR="$OUT/eager"   ENFORCE_EAGER=1 python scripts/safety_eval.py gen vfd_k16 || echo "A vfd FAILED"

echo "########## arm B: compiled-R1 (ENFORCE_EAGER=0, vfd.single_stream=True) ##########"
SAFETY_OUTDIR="$OUT/compiled" ENFORCE_EAGER=0 python scripts/safety_eval.py gen base   || echo "B base FAILED"
SAFETY_OUTDIR="$OUT/compiled" ENFORCE_EAGER=0 python scripts/safety_eval.py gen vfd_k16 || echo "B vfd FAILED"

echo "########## Ray2333 score (per arm) ##########"
SAFETY_OUTDIR="$OUT/eager"    python scripts/safety_eval.py score base vfd_k16 || echo "A score FAILED"
SAFETY_OUTDIR="$OUT/compiled" python scripts/safety_eval.py score base vfd_k16 || echo "B score FAILED"

echo "########## PAIRED token-divergence: eager-R1 vs compiled-R1 (vfd_k16) ##########"
python - "$OUT" <<'PY'
import json, sys, os
out = sys.argv[1]
def load(d, tag):
    rows = {}
    with open(os.path.join(out, d, f"safety_gen_{tag}.jsonl")) as fh:
        for line in fh:
            r = json.loads(line); rows[r["index"]] = r
    return rows
for tag in ("vfd_k16", "base"):
    try:
        E, C = load("eager", tag), load("compiled", tag)
    except FileNotFoundError as e:
        print(f"[{tag}] missing: {e}"); continue
    n = len(E); ident = 0; firstdiffs = []
    for i in sorted(E):
        et, ct = E[i].get("token_ids", []), C[i].get("token_ids", [])
        if et == ct:
            ident += 1; continue
        # first index where they differ
        m = min(len(et), len(ct)); fd = next((k for k in range(m) if et[k] != ct[k]), m)
        firstdiffs.append(fd)
    print(f"[{tag}] identical token seqs: {ident}/{n}  "
          f"({100*ident/n:.0f}%)  divergent: {n-ident}")
    if firstdiffs:
        firstdiffs.sort()
        med = firstdiffs[len(firstdiffs)//2]
        print(f"        first-divergence step among divergent: "
              f"min={firstdiffs[0]} median={med} max={firstdiffs[-1]} "
              f"(early divergence = chaotic float-noise amplification; "
              f"step-0 divergence on many = a real candidate-forward bug)")
PY
echo "=== DONE job ${SLURM_JOB_ID} ==="
