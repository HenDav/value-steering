# CLAUDE.md — value-steer

Guidance for AI coding agents (and humans) working in this repo. The full contributor guide is
[CONTRIBUTING.md](CONTRIBUTING.md); this file is the short version of the invariants that keep the
code correct.

## What this is

A vLLM plugin implementing two inference-time interventions driven by one shared scalar value head:

- **Dynamic abstention** — gate generation to EOS when the value crosses a calibrated threshold.
- **Value-filtered decoding (VFD)** — per step, sample K candidates and commit one by the value
  head in a *single* forward (the K-candidate forward *is* the decode forward; no extra model pass).

Both score the **same** feature with the **same** head, so one trained probe serves either mode.

## Non-negotiable conventions

1. **Ground every vLLM API against the pinned source before writing.** vLLM internals shift across
   minor versions — that's why the compat harness exists. Clone the tag and read it:
   `git clone --branch v0.19.1 --depth 1 https://github.com/vllm-project/vllm`. Code comments cite
   0.19.1 line numbers as anchors; verify, don't trust.
2. **No orphan/dead code.** A gap is one explicit `raise NotImplementedError` at a located point,
   not a method that silently never runs.
3. **Features must FIRE, not just run.** The runner hooks swallow exceptions in production so
   decoding never crashes — therefore "it ran" ≠ "it worked". Assert observable behavior;
   `strict=True` in `additional_config` re-raises instead of swallowing (use it in tests).
4. **Pure logic stays vLLM-free.** `value_probe`, `steering_ops`, `calibration`, `train_probe`,
   `scratch_alloc`, `versions`, `verifiers` import no vLLM and are CPU-unit-tested. Only the runners
   and `worker` import vLLM. Never pull vLLM into the pure modules or into `__init__.py`.
5. **Validation leads, the pin trails.** Record a vLLM version as validated only after the checks
   pass (`value-steer-compat` static + GPU behavioral). Never widen `>=0.19.1,<0.20` on faith.

## Feature contract (don't drift)

The value head scores the backbone's **final-layer, POST-final-norm** `last_hidden_state` (the exact
tensor `lm_head` consumes), per token, in **fp32**. The head architecture is fixed
(`value_probe.ValueHead`) and a checkpoint must match it (strict load). Abstention and VFD score the
*same* feature with the *same* head.

## Commands

```bash
make test          # CPU unit suite (no GPU, no vLLM needed) — must stay green
make compat        # static contract checks against the installed vLLM
make gpu-test      # behavioral tests (needs CUDA + $VALUE_STEER_TEST_MODEL)
make lint          # ruff
```

SLURM reproducibility harnesses live in [examples/](examples/); see
[docs/cluster-setup.md](docs/cluster-setup.md) for the environment they assume and
[docs/training-a-value-head.md](docs/training-a-value-head.md) for training a head.
