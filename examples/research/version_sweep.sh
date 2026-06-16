#!/bin/bash
# Drive the compatibility agent (value-steer-compat record) across vLLM versions.
# For each version: fresh venv, install vllm==V (pulls its pinned torch) + the package
# (--no-deps so the pyproject pin doesn't fight the requested version), run the STATIC
# contract checks, which record_validation()s pass/fail into validated_versions.json.
# CPU-only (static needs import vllm, no GPU). Behavioral (GPU) is a separate step for any
# version that passes static.  Usage:  bash scripts/version_sweep.sh 0.20.0 0.20.1 ...
set -u
REPO=${VALUE_STEER_ROOT:-$(git rev-parse --show-toplevel)}
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
conda activate value-steer
BASEPY=$(which python)                       # python 3.12 to seed the venvs
cd "$REPO"

for V in "$@"; do
  echo "================= vLLM $V ================="
  ENVDIR=${VS_SCRATCH:-/tmp/$USER}/envs/sweep-$V
  rm -rf "$ENVDIR"
  "$BASEPY" -m venv "$ENVDIR" || { echo "$V: venv create FAILED"; continue; }
  # shellcheck disable=SC1091
  source "$ENVDIR/bin/activate"
  python -m pip install -q --upgrade pip >/dev/null 2>&1
  if ! python -m pip install -q "vllm==$V" 2>/tmp/sweep_pip_$V.log; then
    echo "$V: pip install vllm==$V FAILED (see /tmp/sweep_pip_$V.log)"; tail -3 /tmp/sweep_pip_$V.log
    deactivate; rm -rf "$ENVDIR"; continue
  fi
  python -m pip install -q pytest >/dev/null 2>&1
  python -m pip install -q -e . --no-deps >/dev/null 2>&1
  echo "--- installed: $(python -c 'import vllm;print("vllm",vllm.__version__)' 2>&1 | tail -1) ---"
  # The agent: static contract checks -> record pass/fail for the live vLLM version.
  value-steer-compat record "version-sweep static ($(date -u +%FT%TZ))" 2>&1 | grep -E "PASS|FAIL|recorded|vLLM|NOT in" | sed 's/^/  /'
  deactivate
  rm -rf "$ENVDIR"                            # reclaim disk; envs are throwaway
done
echo "================= sweep done ================="
