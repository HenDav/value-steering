#!/bin/bash
#SBATCH --job-name=vsteer-isoN
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=isolate_compile_bigN.%j.out
# Higher-power rerun of the compile isolation: eager-R1 vs compiled-R1, K=16, thr 0.3, SAME seed,
# N=256 prompts (vs 64) -> the Llama-judge gap goes from ~1.8sigma to conclusive, and the
# argmin-fallback rate is measured over ~32k steps. Does compile SYSTEMATICALLY change safety, or
# was the n=64 judge gap noise? Gen + Ray2333 + paired token-divergence + Llama judge, all in the
# vsteer env (judge = transformers).  Submit: sbatch --partition=llm examples/research/isolate_compile_bigN.sbatch
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
export VALUE_STEER_MODEL=${VALUE_STEER_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
export VALUE_STEER_VHEAD=${VALUE_STEER_VHEAD:?set VALUE_STEER_VHEAD to a trained value head}
export VALUE_STEER_UTIL=0.45 SAFETY_N=256 SAFETY_MAXTOK=128
export VFD_THRESHOLD=0.3 VFD_K=16 SINGLE_STREAM=1 SAFETY_MAX_NUM_SEQS=64
OUT=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}/isolate_compile_bigN
mkdir -p "$OUT/eager" "$OUT/compiled"

echo "########## arm A: eager-R1 (ENFORCE_EAGER=1) ##########"
SAFETY_OUTDIR="$OUT/eager"    ENFORCE_EAGER=1 python scripts/safety_eval.py gen base   || echo "A base FAILED"
SAFETY_OUTDIR="$OUT/eager"    ENFORCE_EAGER=1 python scripts/safety_eval.py gen vfd_k16 || echo "A vfd FAILED"
echo "########## arm B: compiled-R1 (ENFORCE_EAGER=0, single_stream) ##########"
SAFETY_OUTDIR="$OUT/compiled" ENFORCE_EAGER=0 python scripts/safety_eval.py gen base   || echo "B base FAILED"
SAFETY_OUTDIR="$OUT/compiled" ENFORCE_EAGER=0 python scripts/safety_eval.py gen vfd_k16 || echo "B vfd FAILED"

echo "########## Ray2333 score ##########"
SAFETY_OUTDIR="$OUT/eager"    python scripts/safety_eval.py score base vfd_k16 || echo "A score FAILED"
SAFETY_OUTDIR="$OUT/compiled" python scripts/safety_eval.py score base vfd_k16 || echo "B score FAILED"

echo "########## PAIRED token-divergence eager-R1 vs compiled-R1 ##########"
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
    n=len(E); ident=0; fds=[]
    for i in sorted(E):
        et, ct = E[i].get("token_ids", []), C[i].get("token_ids", [])
        if et == ct: ident += 1; continue
        m=min(len(et),len(ct)); fds.append(next((k for k in range(m) if et[k]!=ct[k]), m))
    print(f"[{tag}] identical token seqs: {ident}/{n} ({100*ident/n:.0f}%)  divergent: {n-ident}")
    if fds:
        fds.sort(); print(f"        first-divergence step: min={fds[0]} median={fds[len(fds)//2]} max={fds[-1]}")
PY

echo "########## Llama judge (primary metric) ##########"
SAFETY_OUTDIR="$OUT/eager"    JUDGE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct python scripts/safety_eval.py judge base vfd_k16 || echo "A judge FAILED"
SAFETY_OUTDIR="$OUT/compiled" JUDGE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct python scripts/safety_eval.py judge base vfd_k16 || echo "B judge FAILED"
echo "=== DONE job ${SLURM_JOB_ID} ==="
