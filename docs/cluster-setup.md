# Cluster setup (example SLURM environment)

The harnesses in [../examples/](../examples/) were developed on a multi-node SLURM GPU cluster. This
note describes the *kind* of environment they assume so you can adapt them — it is illustrative, not
a description of any specific machine. The scripts read the environment variables in
[../examples/README.md](../examples/README.md).

## Golden rule

Run GPU work on **compute nodes via `sbatch`**, not on the login/entrance node (whose GPUs are
shared and usually saturated). The harnesses are written to be submitted, not run interactively.

## Filesystems

- A **shared network filesystem** (NFS-style) holds the repo checkout and job outputs — visible on
  every node. Point `VALUE_STEER_ROOT` at the checkout here.
- **Node-local scratch** (per-node SSD, *not* shared) holds the conda env, pip cache, and the
  Hugging Face cache. Set `VS_SCRATCH` to a node-local path (the scripts default to `/tmp/$USER`).
  Anything written here exists only on the node that wrote it.
- `/tmp` is often small and overflows when pip extracts torch/CUDA wheels; set `TMPDIR` onto
  node-local scratch.

## Environments (node-local)

- Build the env **on the compute node** and key it by hostname (e.g. `$VS_SCRATCH/envs/vsteer-<host>`).
  Install `value_steer` **editable** (`pip install -e .`) from the shared checkout so login-node code
  edits are picked up without a rebuild.
- A representative pinned stack: `torch==2.10.0` (CUDA 12.x build), `vllm==0.19.1`,
  `pip install -e ".[dev,train]"`.
- Set `CONDA_BASE` to your conda install prefix (the scripts default to `$HOME/miniconda3`).

## CUDA driver / wheel compatibility

- vLLM is pinned to `>=0.19.1,<0.20`. Those wheels are built for CUDA 12.x and run on a CUDA 12.5
  driver via minor-version forward-compat.
- vLLM ≥ 0.20 PyPI wheels are **CUDA-13** and will not load on a CUDA-12 driver — this is part of
  why the pin stays `<0.20` (the behavioral validation also has not been extended past it). Widen
  only after `value-steer-compat` passes on a GPU box.
- VFD's KV-copy path needs **FlashAttention v2** (compute capability ≥ 8.0, e.g. A100/H100). Older
  GPUs without FA2 are not supported for VFD.

## Hugging Face models

- Set `HF_HOME` onto node-local scratch. Compute nodes with internet download models directly; do
  not force offline mode unless the model is known to be cached on that node.
- **Gated** models (e.g. `meta-llama/*`, Llama-Guard) need an HF token. Where a token is not
  available, use an **ungated mirror** of the same weights (e.g.
  `NousResearch/Meta-Llama-3.1-8B-Instruct`).

## Required env vars for in-process vLLM (validation/eval jobs)

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` — run the engine **in-process** so tests/eval can reach the live
  `model_runner`, and to avoid the forked-EngineCore CUDA re-init crash.
- `VLLM_HOST_IP=127.0.0.1` — avoids an IPv6 c10d rendezvous stall at model init.
- `TOKENIZERS_PARALLELISM=false`.

## Scheduling reality

GPUs on a shared cluster are frequently fully allocated. A submitted job either queues or gets packed
onto an already-busy GPU, which then fails at engine init with a "free memory … less than desired GPU
memory utilization" message — that is **contention, not a code bug**. Requeue or wait for a free GPU;
inspect with `squeue`, `sinfo`, and `scontrol show node <n>`.
