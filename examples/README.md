# Examples — SLURM harnesses

These are the **reproducibility harnesses** used to develop and evaluate value-steer on a SLURM
cluster (the paper experiments). They are **templates, not turnkey scripts**: they were written for
one specific environment and have been genericized with environment variables so you can adapt them
to yours. Expect to edit partition names, GPU types, and time limits to match your scheduler.

- **`slurm/`** — the reusable, parameterized harnesses: GPU validation, real-model behavioral
  validation, the value-head training pipeline, safety generation + judging, and profiling.
- **`research/`** — the sweeps, ablations, and diagnostics from development (compile-vs-eager
  isolation, K-sweeps, feature-parity probes, calibration investigation). Kept for provenance; most
  users will not need them.

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

A ready-made value head is published at
[`HenDav/value-steer-safety-head`](https://huggingface.co/HenDav/value-steer-safety-head) — point
`VALUE_STEER_VHEAD` at a local download of it.

The eval/profiling/diagnostic harnesses read a few more knobs (all optional, with defaults):

| Variable | Meaning | Default |
|---|---|---|
| `VALUE_STEER_TEST_MODEL` | small model for quick GPU smoke tests | `facebook/opt-125m` |
| `VALUE_STEER_UTIL` | vLLM `gpu_memory_utilization` | `0.45` |
| `JUDGE_MODEL` | judge/reward model for safety scoring | `NousResearch/Meta-Llama-3.1-8B-Instruct` |
| `SAFETY_N` / `SAFETY_MAXTOK` / `SAFETY_SEED` | eval prompt count / max new tokens / seed | `64` / `128` / `15` |
| `VFD_K` / `VFD_THRESHOLD` | candidates per step / intervention threshold | `8` / `0.5` |
| `ENFORCE_EAGER` | `1` = eager (serving default); `0` = compile | `1` |
| `SINGLE_STREAM` | opt into the compiled single-stream path (one request at a time) | `0` |
| `DOMAIN` | training domain (verifier + data) for the value-head pipeline | `safety` |
| `SPP` | samples per prompt when generating training data | `1` |
| `SAFETY_OUTDIR` | where safety-eval writes generations/scores | `.` |
| `DEC_N` | prompts to generate for the decode pipeline | pipeline default |
| `N_PROMPTS` / `HH_SUBSET` | prompt cap / hh-rlhf subset for `train_canonical` | all / `harmless-base` |
| `VSTEER_TORCH_INDEX` | torch wheel index for the version sweep (e.g. `.../cu128` on a 12.x driver) | unset |

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
