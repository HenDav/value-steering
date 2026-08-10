# Training a value head

The value head is a small scalar probe that scores the backbone's post-final-norm hidden state
per token. VFD commits the candidate with the **lowest** value; abstention gates on it. The head
predicts **P(undesirable) ∈ [0, 1]** (0 = good, 1 = bad), so the *same* head, loss, feature, and
selection logic work for any domain — only the **verifier** that labels training data differs.

## Critical: train on DECODE features, not prefill features

The head is *scored at inference* on the hidden VFD computes during **decode** (its candidate
forward). That tensor differs from a **prefill** extraction (running the full sequence through the
model at once, e.g. an HF forward or vLLM pooling) by **~0.97 cosine** — far more than float noise.
A head trained on prefill features can be an excellent *classifier* yet fail to *steer*, because
its sharp decision boundary sits in the wrong place for the decode features it's actually scored
on. Measured on hh-rlhf (Mistral-7B, Llama judge): a **prefill/pooling** head barely moves the
unsafe rate, while a **decode-matched** head steers it down at threshold 0.3.

So **extract training features the way they're scored**: generate responses with the VFD runner
while capturing the per-token decode hidden (`VFD_DUMP_HIDDEN`), then train on those. This is what
`scripts/decode_extract.py` does, and it's the supported path.
(A robust enough head *can* tolerate the prefill→decode shift — that's why some prefill-trained
heads work — but decode-matched extraction is the principled fix and works without relying on it.)

## One command (SLURM)

```bash
DEC_N=8000 sbatch examples/slurm/decode_pipeline.sbatch
```

Runs, on one node: VFD-generate + capture decode features → judge-label + split → train head +
calibrate → `trained/safety_decode/value_head.bin` (+ `.meta.json` with FeatureSpec, calibrated
threshold, recipe). The feature cache is node-local; the sbatch is a template — see
[examples/README.md](../examples/README.md) for the env vars it expects (model, partition, scratch).

## The steps (run directly)

```bash
# 1. generate responses with VFD + capture the DECODE hidden -> feature cache + gen.jsonl
VFD_DUMP_HIDDEN=1 python scripts/decode_extract.py --phase gen --cache-dir vh.cache \
    --model <model> --head <any value_head.bin> --source <prompts.jsonl> --n 8000

# 2. judge-label the generations + split train/val (separate process: judge loads after gen frees)
python scripts/decode_extract.py --phase label --cache-dir vh.cache \
    --judge-model NousResearch/Meta-Llama-3.1-8B-Instruct --val-split 0.1

# 3. train (+ optionally write ONE conservative threshold, tau=alpha=0.05, into the sidecar)
python scripts/train_value_head.py --phase train --cache-dir vh.cache \
    --out value_head.bin --domain safety --calibrate --cal-tau 0.05

# 4. calibrate the full conformal ĉ(α) CURVE (what the published heads carry) from the labeled
#    decode cache -> vh.cache/val/thresholds.json  (CPU-only; no model, no retraining)
python scripts/decode_extract.py --phase calibrate --cache-dir vh.cache/val \
    --head value_head.bin --alphas 0.05,0.25,0.45,0.65,0.85
```

Recipe defaults match the reference (focal + TD-coherence, coh 0.1, lr 1e-4 with linear
warmup+decay, batch 128, 10 epochs early-stopped).

Step 3's `--calibrate` bakes a **single** threshold (at `--cal-tau`) into `value_head.bin.meta.json`
— handy for a quick default. Step 4 instead emits the **whole ĉ(α) curve** (`P(intervene | safe) ≤ α`
at each α); that curve is what the published sidecars carry, so callers pick an α rather than inherit
one number. Both read the same decode-matched, judge-labeled cache — the value the VFD runner scores
at inference — so either threshold is valid for decode (a prefill-extracted threshold would not be).

> **Deprecated — prefill path.** `scripts/gen_value_data.py` + `train_value_head.py --phase
> extract` (vLLM **pooling**) extract *prefill* features and carry the train/inference mismatch
> above. Kept for reference; prefer `decode_extract.py`.

## Using the trained head

Pre-trained safety heads are published at
[`HenDav/value-steer-safety-head`](https://huggingface.co/HenDav/value-steer-safety-head), one per
backbone × dataset — **Mistral-7B-Instruct-v0.3** and **Llama-3.1-8B-Instruct** × **hh-rlhf /
beavertails / pku_saferlhf** (e.g. `mistral/hh-rlhf.bin`), each decode-matched. Every sidecar
carries a conformal **ĉ(α) curve**: ĉ(α) is calibrated so VFD intervenes on **at most an α fraction
of safe generations** (`P(intervene | safe) ≤ α`). Pick the α that fits your risk tolerance.

```python
LLM(model=..., worker_cls="value_steer.worker.ValueSteerWorker",
    additional_config={"vfd": {"enabled": True, "value_head_path": "mistral/hh-rlhf.bin",
                               "threshold": 0.36, "num_candidates": 8}})
```

The threshold *is* a point on that curve — the Mistral hh-rlhf head is ~0.36 at α=0.45, rising to
~0.75 at α=0.05 (lower α → higher threshold → fewer interventions). The published curves were fit
**decode-matched** (score the values the VFD runner produces during decode, then take the conformal
quantile over the safe trajectories); recalibrate on your own held-out data the same way via
`scripts/decode_extract.py`.

## Adding a domain (math, code, …)

Domain-pluggable; a new domain is two additions:

1. **A verifier** mapping `(prompt, generation) → P(undesirable) ∈ [0, 1]`: a class with
   `score(prompt, generation, meta=None) -> float` and `register("<name>", Factory)`.
   - pure (no heavy deps) → `value_steer/verifiers.py`; needs transformers/sandbox → register from
     `scripts/value_verifiers.py`, keeping `value_steer/` vLLM-free.
   - **Math:** wrap an existing grader (e.g. `math_verify`); ground truth in `meta["answer"]`. (stub)
   - **Code:** execute against unit tests in a sandbox; return the failure fraction. (stub)
2. **A prompt source** in `scripts/dataset_loaders.py:load_prompts` yielding `{"prompt", "meta"}`.

Then run the decode pipeline for that domain. The loss, decode-feature extraction, training,
calibration, and VFD selection are all domain-agnostic.

## Relation to the papers

This library implements the two methods faithfully but with a few deliberate, documented
simplifications — worth knowing if you cross-reference the papers:

- **Value polarity differs by mode.** VFD scores **P(undesirable)** and commits the lowest-scoring
  candidate. Dynamic abstention gates when the value is **below** the threshold, so its head scores
  **P(desirable / continue)** — the sign of VFD's. (In *Selective Safety Steering via Value-Filtered
  Decoding* the value is P(safe) and the filter is `V ≥ c`; here it is the equivalent
  `P(unsafe) < c`. In *Knowing When to Quit* the value is P(correct) and the rule is abstain-if-below,
  which the abstention runner's default `V < c` matches.)
- **VFD uses first-safe selection, not the paper's two-phase rollback.** The optimistic
  "sample one, keep if acceptable, else roll back the KV cache" path is dropped; committing the
  **first** candidate under threshold (else the lowest) recovers the same selective behavior with a
  single forward and no rollback. See `value_steer/vfd_model_runner.py`.
- **Calibration.** `value_steer.calibration` provides the value-filtered-decoding / safety-line
  conformal bound (`P_H0(max_t p_t ≥ c) ≤ tau`). The dynamic-abstention paper calibrates its
  threshold empirically (quantile + isotonic) and carries no such conformal guarantee.
- **Abstention implements the β = 0 special case** (force EOS), not the paper's full regularized-RL /
  abstention-token formulation.
