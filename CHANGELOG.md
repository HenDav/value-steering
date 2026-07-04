# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/HenDav/value-steering/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/HenDav/value-steering/releases/tag/v0.1.0
