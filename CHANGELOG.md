# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-10

### Added
- **Decode-matched conformal calibration.** `scripts/decode_extract.py --phase calibrate` scores the
  head over each labeled example's captured *decode* trajectory and calls `posterior_threshold` at
  each α, writing the ĉ(α) **curve** into the head's sidecar with the guarantee
  `P(intervene | safe) ≤ α`. New generation knobs: `--single-stream`, `--disable-hybrid-kv-cache`
  and `--force-arch` (mixed-attention / multimodal backbones), and `--strip-thinking` (judge the
  answer, not a reasoning model's chain-of-thought).
- A bit-exact VFD behavior-preservation gate (`examples/research/refactor_reftest.{py,sbatch}`):
  save a token-id reference on a baseline, then `REFTEST_MODE=check` after a refactor.

### Changed
- Helpfulness is now judged by the paper's Llama-3.1-8B **helpful/unhelpful compliance judge**
  (`build_helpfulness_judge_messages`, replicated verbatim from `llm_safety`), reported alongside the
  unsafe rate by `safety_eval.py judge`. This replaces the Ray2333 reward model as the helpfulness
  metric (Ray2333 remains only as an independent harmlessness check). Adds a `helpfulness` verifier.
- The published head artifact is now **six per-backbone safety heads**
  ([`HenDav/value-steer-safety-head`](https://huggingface.co/HenDav/value-steer-safety-head):
  `mistral/` + `llama/` × hh-rlhf / beavertails / pku_saferlhf), each carrying a decode-matched
  ĉ(α) curve in its sidecar instead of a single threshold.

### Removed
- The `VFD_PROFILE` / `VFD_DEBUG` profiling scaffolding (`scripts/vfd_profile.py`, the profiling
  sbatches, and the runner hooks) — it measured the pre-rework single-stream decode path.

### Fixed
- **Batched / continuous-batching VFD decoding is now correct** under eager serving, so
  `enforce_eager=True` is a supported path for **all batch sizes**, not just single-request. Two
  fixes make the K-candidate KV surgery robust when multiple requests decode concurrently: a
  block-boundary commit **deferral** (a winner whose new token starts a fresh KV block is held until
  the scheduler allocates that block, instead of corrupting block 0) and a **slot-sink** for
  in-flight decoders (a base prefill/mixed step no longer overwrites an already-decoding request's
  KV slot).
- **Compiled batched VFD decoding is now correct for all batch sizes**, so the
  CUDA-graph/`torch.compile` path is no longer single-stream only. Two fixes make the compiled path
  robust under concurrent R>1 decode: `fast_build` for the attention-metadata builder (fixes the
  FlashAttention-metadata crash on the candidate forward) and populating the compiled model's
  persistent input buffers before each captured step. **Batched cudagraph capture is enabled** and
  runs faster than eager (~+10–26% tok/s measured across R=1–16); capturing the R>1 candidate graph
  needs `max_num_seqs >= R*K`, above which it falls back to a still-compiled path.

## [0.1.1]

### Fixed
- `load_value_head` now defaults `device` to CUDA-when-available (else CPU, with a warning) and
  returns a **frozen, eval-mode** head — so scoring with `.p()`/`.logit()` no longer emits spurious
  autograd warnings, and the pre-trained-head snippet works without a `device` argument.
- `value_steer.worker` raises a clear "install `value-steer[vllm]`" `ModuleNotFoundError` when vLLM
  is absent, matching the CLI. `train_probe` detaches the loss before logging.
- Example harnesses: `safety_eval.sbatch` requires `SAFETY_PROMPTS` up front; the version/behavioral
  sweep drivers are called from their `examples/research/` path; research scripts fail with a clear
  message instead of a bare `KeyError`/`IndexError` when a required env var or arg is missing.
- README pre-trained-head snippets reference the real artifact `value_head.bin` (not `vhead.pt`) and
  note that `pytest` needs the `[dev]` extra; `CITATION.cff` version bumped to 0.1.1.

### Documentation (paper fidelity)
- Clarified that the abstention head scores **P(continue)** (gate when the value is LOW), the
  sign-opposite of VFD's **P(undesirable)** — matching the runner default and the abstention paper's
  abstain-if-below rule (previously mislabeled "P(should quit)").
- Attributed the conformal false-intervention bound to the value-filtered-decoding / safety line;
  the dynamic-abstention paper calibrates empirically and carries no such bound.
- New "Relation to the papers" section in `docs/training-a-value-head.md` documenting the deliberate
  simplifications (per-mode polarity, VFD first-safe vs the paper's two-phase rollback, calibration
  provenance, β=0-only abstention).

### Added
- Canon safety-head training/eval harnesses (`examples/slurm/{train_canonical,canonical_eval*,gpu_validate_h100}.sbatch`).
- **Decode-matched feature extraction** (`scripts/decode_extract.py`) — the supported way to build
  value-head training data. The head is scored at inference on the hidden VFD computes during
  *decode*, which differs from a *prefill* extraction (pooling/HF) by ~0.97 cosine; training on the
  decode-matched features makes the head steer, whereas a prefill-trained head barely moves the
  unsafe rate. Generate-and-capture via the new `VFD_DUMP_HIDDEN` runner hook; see
  `docs/training-a-value-head.md`.

### Changed
- Probe training matches the reference recipe: linear warmup+decay LR schedule (pure torch),
  `lr=1e-4`, batch 128; per-epoch loss logging (`train_probe(..., verbose=True)`); `DataLoader`
  `num_workers`/`pin_memory` for the cached-feature path.

### Deprecated
- The prefill/pooling feature path (`scripts/gen_value_data.py` + `train_value_head.py --phase
  extract`) carries a train/inference feature mismatch; prefer `decode_extract.py`.

## [0.1.0]

Initial public release.

### Added
- **Dynamic abstention** runner — gates generation to EOS when the value head crosses a calibrated
  threshold (sampling-site intervention).
- **Value-filtered decoding (VFD)** runner — per step, samples K candidates and commits one by a
  scalar value head in a single forward (no extra model pass), keeping the natural sample when it is
  already acceptable.
- Shared **`ValueHead`** + feature contract (final post-norm hidden state, the tensor `lm_head`
  consumes), pure **steering ops**, and **conformal calibration** (`posterior_threshold`,
  `martingale_threshold`) with a finite-sample bound on false interventions.
- **Probe training** (`train_probe`) with focal + TD-coherence loss; a vLLM-pooling feature
  extractor and an on-disk feature cache for training at scale; a domain-pluggable **verifier**
  interface (safety judge implemented; math/code are located stubs).
- **`--worker-cls value_steer.worker.ValueSteerWorker`** entry point (no monkeypatching), the
  `value-steer-compat` version-contract harness, and a validated-vLLM registry.

### Known limitations
- vLLM is pinned to a behaviorally-validated span (`>=0.19.1,<0.20`); other versions warn at runtime
  and must pass `value-steer-compat` before the pin is widened.
- The VFD CUDA-graph/compile path is **single-stream only**; eager (`enforce_eager=True`) is the
  correct serving default for all batch sizes.

[Unreleased]: https://github.com/HenDav/value-steering/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/HenDav/value-steering/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/HenDav/value-steering/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/HenDav/value-steering/releases/tag/v0.1.0
