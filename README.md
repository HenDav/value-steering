# value-steer

[![CI](https://github.com/HenDav/value-steering/actions/workflows/ci.yml/badge.svg)](https://github.com/HenDav/value-steering/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/value-steer.svg)](https://pypi.org/project/value-steer/)
[![Python](https://img.shields.io/pypi/pyversions/value-steer.svg)](https://pypi.org/project/value-steer/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Inference-time value steering for [vLLM](https://github.com/vllm-project/vllm): two
decode-time interventions driven by a shared scalar value head.

- **Dynamic abstention** ([Knowing When to Quit](https://arxiv.org/abs/2604.18419), ICML 2026) —
  gate generation to EOS when the value crosses a calibrated threshold.
- **Value-filtered decoding** ([Selective Safety Steering via Value-Filtered Decoding](https://arxiv.org/abs/2605.14746)) —
  at each step, sample K candidates and commit one by a safety value, keeping the natural sample
  when it is already safe.

Both score the *same* feature (the backbone's final post-norm hidden state, the tensor
`lm_head` consumes) with the *same* head, so one trained probe serves either mode.

## Install

```bash
pip install value-steer              # core (torch, numpy) — pure modules + training/calibration
pip install "value-steer[vllm]"      # + the vLLM runtime (serving / decoding)
pip install "value-steer[train]"     # + probe training (transformers)
pip install "value-steer[dev]"       # + pytest, ruff, build, twine
```

vLLM is an **optional dependency** pinned to the behaviorally-validated span
(`>=0.19.1,<0.20`); install it to match your CUDA driver, and run `value-steer-compat`
before widening the pin (see *Compatibility*). The pure modules (value head, steering
ops, calibration, probe training) import **without** vLLM, so training/calibration boxes
need only the core install.

Pre-trained safety value heads are published at
[`HenDav/value-steer-safety-head`](https://huggingface.co/HenDav/value-steer-safety-head),
laid out per backbone: **Mistral-7B-Instruct-v0.3** and **Llama-3.1-8B-Instruct** × **hh-rlhf /
beavertails / pku_saferlhf** (e.g. `mistral/hh-rlhf.bin`, `llama/pku_saferlhf.bin`), labeled by a
Llama-3.1 judge. Each sidecar carries the feature contract and a **decode-matched conformal
threshold curve** ĉ(α) — pick the α (false-intervention budget) that fits your risk tolerance
rather than a single baked-in threshold. See the model card.

## Use

Both modes plug in via vLLM's supported `--worker-cls` surface — no monkeypatching.

Abstention:
```bash
vllm serve <model> \
  --worker-cls value_steer.worker.ValueSteerWorker \
  --additional-config '{"abstain": {"value_head_path": "mistral/hh-rlhf.bin", "threshold": 0.5}}'
```

Value-filtered decoding (run with speculative decoding OFF — VFD owns the decode forward):
```bash
vllm serve <model> \
  --worker-cls value_steer.worker.ValueSteerWorker \
  --additional-config '{"vfd": {"value_head_path": "mistral/hh-rlhf.bin", "threshold": 0.36, "num_candidates": 8}}'
```

Per-request override via `SamplingParams.extra_args` (`abstain_threshold` / `vfd_threshold`).

## Train a probe

```python
from value_steer.train_probe import train_probe, save_probe_checkpoint
from value_steer.value_probe import ValueHead

head = ValueHead(hidden_size)            # shared head; fp32 on the post-norm feature
train_probe(backbone, head, train_loader, loss_name="focal", use_td=True, coh_weight=0.1)
save_probe_checkpoint("value_head.bin", head, threshold=calibrated_c, meta={"loss": "focal"})
```

`save_probe_checkpoint` writes the bare head weights to `value_head.bin` (loaded by the
runners) plus a `value_head.bin.meta.json` sidecar with the feature spec, calibrated
threshold, and metadata. The objective is label-agnostic — your labels define whether
the value means P(unsafe) or P(should-quit).

## Calibrate the threshold

```python
from value_steer.calibration import posterior_threshold   # VFD / posterior filter
from value_steer.calibration import martingale_threshold  # time-to-unsafe martingale

c = posterior_threshold(safe_labels, trajectories, tau=0.05)
```

Given held-out `(label, per-step value trajectory)` pairs, this returns the threshold `ĉ(α)`
(the `tau` argument **is** α) carrying a finite-sample conformal guarantee: **VFD intervenes on at
most an α fraction of safe generations** — formally `P_H0(max_t p_t ≥ ĉ) ≤ α` with H0 = "the
generation is safe". That is the guarantee the threshold carries — not a hand-tuned number. Lower α
→ higher `ĉ` → fewer interventions.

## Compatibility

The runners bind to a few vLLM internals. `compat_check.py` is the version gate:

```bash
value-steer-compat            # static contract checks (needs only `import vllm`)
value-steer-compat abstain    # abstention subset
```

Run it on every vLLM bump. Static checks fail loudly if a bound internal moved;
behavioral checks (GPU) assert the feature actually *fires* — necessary because the
runner hooks swallow errors in production, so "it ran" is not "it worked." Pair the
agent's per-version run: static first (no GPU, pinpoints the broken contract), GPU
behavioral only if static is green.

## Tests

```bash
pip install -e ".[dev]"     # pytest lives in the dev extra
pytest -q                   # pure-logic suite (no GPU, no vLLM): ops, calibration, training, allocator
```

## Status

| Component | State |
|---|---|
| value head, steering ops, calibration, training | complete, CPU-tested |
| abstention runner | complete vs pinned APIs; EOS-fires check is the GPU behavioral test |
| VFD runner | complete and **GPU-validated** (A100, vLLM 0.19.1, Mistral-7B): single-forward K-candidate decode, end-to-end safer outputs under a Llama-3.1 judge; no silent gaps remain. Compiled batched decode and batched cudagraph capture additionally validated on H100 |
| `--worker-cls` entry point, packaging, compat harness, version registry | complete |

**Release validation (v0.2.0).** Built as a wheel, `pip`-installed into a clean env, and run
end-to-end on **A100 (FA2)** and **H200 (Hopper/FA3)**: CPU suite 72 passed, `value-steer-compat`
8/8, GPU behavioral 17/17 (each GPU test in its own process). An H100 confirmatory run is pending.

The VFD candidate forward goes through `_model_forward` + the attention-metadata builder
(standard paged decode); the KV cache-write is backend-specific and requires FlashAttention
v2's KV layout (compute capability ≥ 8.0). GPU behavioral tests live in
`tests/test_gpu_behavioral.py` (marked `gpu`, skipped without CUDA) and assert the features
*fire*, not merely run.

## Limitations

- **vLLM pin.** Bound to the behaviorally-validated span `>=0.19.1,<0.20`; the runners ground
  against vLLM internals that shift across minor versions. The registry in
  `value_steer/validated_versions.json` is authoritative and warns at runtime for untested
  in-range versions — widen only after `value-steer-compat` passes on a GPU box.
- **Batched serving: both eager and compiled are correct for all batch sizes.**
  `enforce_eager=True` (the serving default) is the simplest option and is correct for **all batch
  sizes** — batched, concurrent VFD serving under continuous batching is a fully supported path. The
  CUDA-graph/`torch.compile` path is now **also correct batched** and cudagraph-captures the
  per-step candidate forward for a throughput gain over eager (measured ~+10–26% tok/s across
  R=1–16). Capturing the R>1 candidate graph requires `max_num_seqs >= R*K` (K = candidates per
  step); above that it falls back to a still-compiled path. `vfd.single_stream` no longer gates
  correctness — it only sizes the scratch KV reserve.
- **VFD threshold.** Each published head's sidecar carries a **decode-matched conformal ĉ(α) curve**,
  not a single number: `ĉ(α)` is calibrated so VFD intervenes on **at most an α fraction of safe
  generations** (`P(intervene | safe) ≤ α`). E.g. the Mistral hh-rlhf head is ~0.36 at α=0.45, rising
  to ~0.75 at α=0.05 (lower α → fewer interventions). Pick the α that matches your risk tolerance. See
  [docs/training-a-value-head.md](docs/training-a-value-head.md).

## Citation

If you use value-steer, please cite the software and the two papers it implements; see
[CITATION.cff](CITATION.cff).
