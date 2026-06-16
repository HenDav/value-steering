#!/bin/bash
# GPU behavioral sweep, driver-aware. vLLM 0.20+ pins torch 2.11.0 whose DEFAULT wheel is
# cu130 (CUDA 13.0) -> too new for this cluster's 12.5 driver -> CUDA unavailable -> tests
# SKIP (which is NOT a pass). So we install torch 2.11.0 from the cu128 index first (runs on
# the 12.5 driver via CUDA 12.x compat, like 0.19.1's cu128), THEN vLLM. We then REQUIRE the
# 8 tests to actually run+pass; skips/fails do not count as behavioral=True.
# Usage (in a SLURM GPU job):  bash scripts/behavioral_sweep.sh 0.20.0 0.20.1 ...
set -u
REPO=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
# Optional torch CUDA index override. On the a100 nodes (driver CUDA 12.5) set
# VSTEER_TORCH_INDEX=https://download.pytorch.org/whl/cu128 to pre-install a 12.x torch.
# On a CUDA-13 node (h100), leave it unset to use vLLM's default (cu130) wheels. NOTE: the
# vllm._C extension itself is CUDA-13-built for vllm>=0.20, so on a 12.5 driver the
# torch.cuda guard below will (correctly) fast-skip and record behavioral=None.
CU=${VSTEER_TORCH_INDEX:-}
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
conda activate value-steer
BASEPY=$(which python)
cd "$REPO"
export VALUE_STEER_TEST_MODEL=facebook/opt-125m
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
NTESTS=8                                   # the gpu suite size; all must run+pass

record() {  # version result static behavioral note
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import sys
from value_steer.versions import record_validation
v,res,st,beh,note=sys.argv[1:6]
b={"true":True,"false":False,"none":None}[beh]
record_validation(v,res,static=(st=="true"),behavioral=b,note=note)
print(f"  recorded {v}: result={res} static={st} behavioral={beh}")
PY
}

for V in "$@"; do
  echo "############### behavioral: vLLM $V ###############"
  ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/bsweep-$V
  rm -rf "$ENVDIR"; "$BASEPY" -m venv "$ENVDIR" || { echo "$V venv FAILED"; continue; }
  # shellcheck disable=SC1091
  source "$ENVDIR/bin/activate"
  python -m pip install -q --upgrade pip >/dev/null 2>&1
  if [ -n "$CU" ]; then
    # driver-compatible torch FIRST (e.g. cu128), then vLLM (its torch==2.11.0 pin satisfied)
    if ! python -m pip install -q torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
          --index-url "$CU" 2>/tmp/bsw_torch_$V.log; then
      echo "$V: torch install ($CU) FAILED"; tail -3 /tmp/bsw_torch_$V.log; deactivate; rm -rf "$ENVDIR"; continue
    fi
  fi
  if ! python -m pip install -q "vllm==$V" pytest 2>/tmp/bsw_pip_$V.log; then
    echo "$V: vllm install FAILED"; tail -3 /tmp/bsw_pip_$V.log; deactivate; rm -rf "$ENVDIR"; continue
  fi
  python -m pip install -q -e . --no-deps >/dev/null 2>&1
  echo "--- $(python -c 'import torch,vllm;print("vllm",vllm.__version__,"torch",torch.__version__,"cuda",torch.cuda.is_available())' 2>&1 | tail -1) ---"
  # Guard: behavioral only means anything if CUDA actually works.
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "$V: CUDA UNAVAILABLE -> cannot behaviorally validate"
    record "$V" pass true none "static contracts pass; behavioral not validated (CUDA unavailable: torch/driver mismatch)"
    deactivate; rm -rf "$ENVDIR"; continue
  fi
  OUT=$(python -m pytest tests/test_gpu_behavioral.py -q -m gpu -rA 2>&1); RC=$?
  echo "$OUT" | tail -16
  PASS_N=$(echo "$OUT" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+" | tail -1); PASS_N=${PASS_N:-0}
  SKIP_N=$(echo "$OUT" | grep -oE "[0-9]+ skipped" | grep -oE "[0-9]+" | tail -1); SKIP_N=${SKIP_N:-0}
  echo "$V: rc=$RC passed=$PASS_N skipped=$SKIP_N"
  if [ "$RC" -eq 0 ] && [ "$PASS_N" -eq "$NTESTS" ] && [ "$SKIP_N" -eq 0 ]; then
    record "$V" pass true true "behavioral validated (a100, opt-125m, torch2.11+cu128, $PASS_N/$NTESTS gpu tests pass)"
  else
    record "$V" fail true false "behavioral FAILED/incomplete (rc=$RC passed=$PASS_N skipped=$SKIP_N)"
  fi
  deactivate; rm -rf "$ENVDIR"
done
echo "############### behavioral sweep done ###############"
