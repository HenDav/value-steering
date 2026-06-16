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
on. Measured on hh-rlhf (Mistral-7B, Llama judge, K=16):

| training feature | net unsafe-rate reduction (thr 0.3) |
|---|---|
| prefill / pooling | **≈ 0** (does not steer) |
| **decode-matched** | **+0.17** (base 0.45 → 0.28) |

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

# 3. train + calibrate on the decode-feature cache (no model loaded)
python scripts/train_value_head.py --phase train --cache-dir vh.cache \
    --out value_head.bin --domain safety --calibrate
```

Recipe defaults match the reference (focal + TD-coherence, coh 0.1, lr 1e-4 with linear
warmup+decay, batch 128, 10 epochs early-stopped).

> **Deprecated — prefill path.** `scripts/gen_value_data.py` + `train_value_head.py --phase
> extract` (vLLM **pooling**) extract *prefill* features and carry the train/inference mismatch
> above. Kept for reference; prefer `decode_extract.py`.

## Using the trained head

```python
LLM(model=..., worker_cls="value_steer.worker.ValueSteerWorker",
    additional_config={"vfd": {"enabled": True, "value_head_path": "value_head.bin",
                               "threshold": 0.3, "num_candidates": 16}})
```

The head steers around **threshold 0.3**. The conformal `--calibrate` threshold in the sidecar is
*conservative* (bounds false interventions at `tau`) and can sit higher than the steering sweet
spot — start at 0.3 and tune. Recalibrate on held-out data with `scripts/calibrate_threshold.py`.

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
