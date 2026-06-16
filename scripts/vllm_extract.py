# SPDX-License-Identifier: Apache-2.0
"""
vLLM-backed feature extractor for value-head training.

DEPRECATED for training: this is a PREFILL extraction (the full sequence at once via the pooling
runner). The value head is SCORED at inference on the DECODE hidden VFD computes, which differs by
~0.97 cosine -- a head trained on these prefill features can classify well yet fail to steer. Use
scripts/decode_extract.py (decode-matched) for training data. This module remains useful for
prefill feature parity/diagnostics.

Extracts the backbone's per-token, POST-FINAL-NORM hidden state via vLLM's pooling path --
the SAME forward (kernels, dtype) the runners score at inference, so the trained head sees
exactly the inference feature (no HF-vs-vLLM numerical gap, one model load, reuse the served
stack). Verified against vLLM 0.19.1:
  * LLM(runner="pooling", convert="embed", pooler_config=PoolerConfig(pooling_type="ALL"))
    -> DispatchPooler.for_embedding registers the "token_embed" task -> AllPool returns
    hidden_states[first:last+1] per sequence (all positions, in order);
  * that hidden is post-`model.norm` (== lm_head input);
  * PoolingParams(use_activation=False) skips the embedding head's PoolerNormalize, so the
    raw post-norm hidden comes out (heads.py: `if activation is not None and use_activation`).

`write_feature_cache(llm, cache_dir, sequences, ...)` runs ONE pooling pass over all sequences
and STREAMS the per-token [Li,H] post-norm features to an on-disk cache (a float16 blob +
index) that value_steer.train_probe.FeatureCacheDataset memmaps. Since the backbone is frozen
the features never change, so we extract once (vLLM batches optimally at high GPU util) and the
head trains over the cache for many epochs. Extraction runs in its OWN process (the sbatch's
extract phase), so the big model is gone -- not co-resident with anything -- before the head
trains, and the cache streams from disk so dataset size is bounded by disk, not RAM.
"""

from __future__ import annotations

import json
import os

import torch


def build_pooling_llm(model: str, *, gpu_memory_utilization: float = 0.4,
                      max_model_len: int = 2048):
    """Load `model` as a pooling model that returns all-token post-norm hidden states."""
    from vllm import LLM
    from vllm.config import PoolerConfig
    return LLM(
        model=model,
        runner="pooling",
        convert="embed",
        pooler_config=PoolerConfig(pooling_type="ALL"),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=False,   # deterministic per-call extraction
    )


def hidden_size(llm) -> int:
    """Backbone hidden size (for sizing the value head). LLM exposes model_config directly
    (entrypoints/llm.py: `self.model_config = self.llm_engine.model_config`)."""
    return int(llm.model_config.get_hidden_size())


def write_feature_cache(llm, cache_dir, sequences, prompt_lens, labels, *, chunk: int = 256,
                        extra_meta: dict | None = None) -> str:
    """Stream a one-pass pooling extraction of `sequences` (list of token-id lists) to an
    on-disk cache under `cache_dir`: a float16 blob `feats.f16` of all [Li,H] rows concatenated,
    `index.jsonl` (offset/length/prompt_len/label per example), and `meta.json` (H, total_rows,
    count, dtype). Peak RAM is one chunk -- so arbitrarily large datasets are supported. Returns
    cache_dir."""
    from vllm import PoolingParams
    pp = PoolingParams(use_activation=False)   # raw post-norm hidden, no PoolerNormalize
    os.makedirs(cache_dir, exist_ok=True)
    feats_path = os.path.join(cache_dir, "feats.f16")
    H, total = None, 0
    with open(feats_path, "wb") as fb, \
         open(os.path.join(cache_dir, "index.jsonl"), "w", encoding="utf-8") as ix:
        for i in range(0, len(sequences), chunk):
            grp = sequences[i:i + chunk]
            res = llm.encode([{"prompt_token_ids": s} for s in grp], pooling_params=pp,
                             pooling_task="token_embed", use_tqdm=False)
            for j, (s, r) in enumerate(zip(grp, res)):
                data = r.outputs.data                          # [li, H], post-norm, in order
                if data.shape[0] != len(s):
                    raise RuntimeError(
                        f"vLLM returned {data.shape[0]} token states for a {len(s)}-token input "
                        "(pooling did not return all positions in order)")
                if H is None:
                    H = int(data.shape[1])
                data.to("cpu", torch.float16).numpy().tofile(fb)
                gi = i + j
                ix.write(json.dumps({"offset": total, "length": int(data.shape[0]),
                                     "prompt_len": int(prompt_lens[gi]),
                                     "label": float(labels[gi])}) + "\n")
                total += int(data.shape[0])
    meta = {"H": H, "total_rows": total, "count": len(sequences), "dtype": "float16"}
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return cache_dir
