# Examples — SLURM harnesses

These are the **reproducibility harnesses** used to develop and evaluate value-steer on a SLURM
cluster (the paper experiments). They are **templates, not turnkey scripts**: they were written for
one specific environment and have been genericized with environment variables so you can adapt them
to yours. Expect to edit partition names, GPU types, and time limits to match your scheduler.

- **`slurm/`** — the reusable, parameterized harnesses: GPU validation, real-model behavioral
  validation, the value-head training pipeline, safety generation + judging, and profiling.
- **`research/`** — the sweeps, ablations, and diagnostics from development (compile-vs-eager
  isolation, K-sweeps, feature-parity probes, calibration investigation). Kept for provenance; most
  users will not need them. One is a live tool: **`refactor_reftest.sbatch`** — a bit-exact
  behavior-preservation gate (save a token-id reference on a baseline, then `REFTEST_MODE=check`
  after each plugin refactor across the `main` / `compiled` / `churn` configs) for anyone changing
  the VFD runner.

## Environment variables

Each script reads these (with sensible fallbacks where safe); set the ones a given script needs:

| Variable | Meaning | Default |
|---|---|---|
| `VALUE_STEER_ROOT` | repo checkout to `cd` into | `$(git rev-parse --show-toplevel)` |
| `VALUE_STEER_MODEL` | backbone (HF id or local path) | `mistralai/Mistral-7B-Instruct-v0.3` |
| `VALUE_STEER_VHEAD` | trained value head (`value_head.bin`) | **required** (no default) |
| `VALUE_STEER_DATA` | training-data jsonl (`{prompt, ...}`) | **required** for training |
| `SAFETY_PROMPTS` | eval/judge prompts jsonl (`{prompt}`) | **required** for safety eval |
| `CONDA_BASE` | conda install prefix | `$HOME/miniconda3` |
| `VS_SCRATCH` | node-local scratch (envs, caches, HF) | `/tmp/$USER` |

Ready-made value heads (Mistral-7B & Llama-3.1-8B × hh-rlhf / beavertails / pku_saferlhf) are
published at
[`HenDav/value-steer-safety-head`](https://huggingface.co/HenDav/value-steer-safety-head) — point
`VALUE_STEER_VHEAD` at a local download of one (e.g. `mistral/hh-rlhf.bin`).

The eval/profiling/diagnostic harnesses read a few more knobs (all optional, with defaults):

| Variable | Meaning | Default |
|---|---|---|
| `VALUE_STEER_TEST_MODEL` | small model for quick GPU smoke tests | `facebook/opt-125m` |
| `VALUE_STEER_UTIL` | vLLM `gpu_memory_utilization` | `0.45` |
| `JUDGE_MODEL` | Llama judge for safety + helpfulness (gated; needs HF token). The ungated `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror runs tokenless but self-refuses on harmful cases → under-reports helpfulness | `meta-llama/Llama-3.1-8B-Instruct` |
| `SAFETY_N` / `SAFETY_MAXTOK` / `SAFETY_SEED` | eval prompt count / max new tokens / seed | `64` / `400` / `15` |
| `SAFETY_CHAT` / `SAFETY_TOP_P` | `1` = instruct chat template (matches training; keep on for real models) / nucleus top-p | `1` / `0.9` |
| `VFD_K` / `VFD_THRESHOLD` | candidates per step / intervention threshold | `8` / `0.5` |
| `ENFORCE_EAGER` | `1` = eager (simplest, serving default); `0` = compile — both correct for all batch sizes | `1` |
| `SINGLE_STREAM` | no longer gates correctness (compiled decode now works batched); only reduces the scratch KV reserve | `0` |
| `DOMAIN` | training domain (verifier + data) for the value-head pipeline | `safety` |
| `SPP` | samples per prompt when generating training data | `1` |
| `SAFETY_OUTDIR` | where safety-eval writes generations/scores | `.` |
| `DEC_N` | prompts to generate for the decode pipeline | pipeline default |
| `N_PROMPTS` / `HH_SUBSET` | prompt cap / hh-rlhf subset for `train_canonical` | all / `harmless-base` |
| `VSTEER_TORCH_INDEX` | torch wheel index for the version sweep (e.g. `.../cu128` on a 12.x driver) | unset |
| `REFTEST_MODE` / `REFTEST_CONFIG` | `refactor_reftest`: `save`\|`check` / `main`\|`compiled`\|`churn` | `check` / `main` |
| `REFTEST_DIR` / `REFTEST_THR` | `refactor_reftest`: reference-json dir / intervention threshold | `$VS_SCRATCH/reftest` / `0.3571` |

## Submitting

The scripts do **not** hardcode `--partition` / node names — pass them at submit time:

```bash
# build the node env + run the CPU suite, compat checks, and GPU behavioral tests
sbatch -p <partition> --gres=gpu:1 examples/slurm/gpu_validate.sbatch

# the one-command value-head training pipeline (generate -> label -> train -> calibrate)
VALUE_STEER_DATA=/path/to/prompts.jsonl \
  sbatch -p <partition> --gres=gpu:1 examples/slurm/train_value_head.sbatch
```

See [../docs/cluster-setup.md](../docs/cluster-setup.md) for notes on the kind of SLURM environment
these were written against (node-local scratch, conda layout, CUDA-driver/wheel constraints).
