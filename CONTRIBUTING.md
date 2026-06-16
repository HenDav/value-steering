# Contributing to value-steer

Thanks for your interest! `value-steer` is a vLLM plugin that binds to a handful of vLLM
internals, so it follows a few non-negotiable conventions that keep it correct across vLLM
versions. Please read these before opening a PR.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"        # core + pytest/ruff/build + transformers (training)
pip install -e ".[vllm]"             # the runtime (vLLM, pinned to the validated span) — needs CUDA
```

The **pure modules** (`value_probe`, `steering_ops`, `calibration`, `train_probe`,
`scratch_alloc`, `verifiers`, `versions`) import without vLLM, so the CPU test suite and
training/calibration work on a box with no GPU.

## Running the checks

```bash
make lint            # ruff
make test            # pure-logic CPU suite (no GPU, no vLLM)
make compat          # static vLLM-contract checks (needs `import vllm`)
VALUE_STEER_TEST_MODEL=facebook/opt-125m make gpu-test   # behavioral tests (needs CUDA)
```

CI runs lint + the CPU suite + a build check. The **GPU behavioral tests and `value-steer-compat`
behavioral pass cannot run in CI** (no GPU) — run them yourself on a CUDA box and paste the result
in the PR.

## The conventions (these earned the current quality)

1. **Ground every vLLM API against the pinned source before writing.** Don't write from memory of
   vLLM internals — they shift across minor versions. Clone the tag and read it
   (`git clone --branch v0.19.1 --depth 1 https://github.com/vllm-project/vllm`). The code cites
   line numbers as anchors; verify, don't trust.
2. **No orphan/dead code.** A gap is an explicit `raise NotImplementedError` at one located point,
   not a method that silently never runs. Implement a seam → delete its raise; can't → leave it.
3. **Features must FIRE, not just run.** The runner hooks swallow exceptions in production so
   decoding never crashes — therefore "it ran" ≠ "it worked". Tests must assert *observable*
   behavior. `strict=True` in `additional_config` re-raises instead of swallowing; use it in tests.
4. **Pure logic stays vLLM-free.** The pure modules import no vLLM and are CPU-unit-tested. Don't
   pull vLLM (or transformers) into them or into `__init__.py`.
5. **Mark anything you can't verify** with a `GPU-VALIDATE` comment saying what would confirm it
   (`grep -rn GPU-VALIDATE value_steer/`).
6. **Validation leads, the pin trails.** Only record a vLLM version as validated (in
   `value_steer/validated_versions.json`, via `value_steer.versions.record_validation`) after the
   static **and** behavioral checks pass. Never widen the `pyproject` pin on faith.

## Pull requests

- Keep changes focused; match the surrounding style (the code is `ruff`-clean).
- Add/extend tests — CPU tests for pure logic, a `@pytest.mark.gpu` behavioral test if a feature
  must fire on an accelerator.
- Update `CHANGELOG.md` under "Unreleased".
- By contributing you agree your work is licensed under Apache-2.0 (see `LICENSE`).
