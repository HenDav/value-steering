# SPDX-License-Identifier: Apache-2.0
"""
Calibrate the VFD intervention threshold from a trained head + held-out data, and write it into
the checkpoint's .meta.json sidecar.

Scores the head over each held-out generation's per-response-token P(undesirable) trajectory,
then calls value_steer.calibration.posterior_threshold -- the conformal (+1) quantile over the
GOOD examples (label 0), giving the documented guarantee P(good triggers intervention) <= tau.
The runner compares `p_unsafe >= threshold`, the same direction calibration returns.

Features come from the SAME vLLM pooling cache used to train (exact inference parity), streamed
from disk (big-dataset safe). Used:
  * by train_value_head.py (--calibrate): calibrate_from_dataset(val_cache, head) -- features
    already extracted, no extra forward;
  * standalone to (re)calibrate an existing checkpoint (extracts a fresh cache, then calibrates):
        python scripts/calibrate_threshold.py --model <m> --head value_head.bin --data val.jsonl --tau 0.05
"""

from __future__ import annotations

import os
import sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import json
import tempfile

import torch

from value_steer.train_probe import ProbeDataset, FeatureCacheDataset, load_probe_meta, save_probe_checkpoint
from value_steer.value_probe import load_value_head, DEFAULT_SPEC
from value_steer import calibration


@torch.no_grad()
def calibrate_from_dataset(ds, head, *, tau: float = 0.05, device: str = "cuda") -> float:
    """Stream a FeatureCacheDataset: each example -> response-token P(undesirable) trajectory +
    binary good/bad label (0=good, score<0.5); return the conformal posterior threshold. Only
    float trajectories are held, so this is big-dataset safe."""
    trajectories, safe_labels = [], []
    for i in range(len(ds)):
        item = ds[i]
        probs = head.p(item["features"].to(device))              # [Li] in (0,1)
        trajectories.append(probs[item["prompt_len"]:].float().tolist())
        safe_labels.append(int(item["label"] >= 0.5))            # 1 = undesirable, 0 = good
    return calibration.posterior_threshold(safe_labels, trajectories, tau=tau)


def main():
    import vllm_extract
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True, help="checkpoint path (value_head.bin)")
    ap.add_argument("--data", required=True, help="held-out labeled jsonl")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = vllm_extract.build_pooling_llm(args.model, gpu_memory_utilization=args.util,
                                         max_model_len=args.max_model_len)
    tok = llm.get_tokenizer()
    head = load_value_head(args.head, vllm_extract.hidden_size(llm), device)

    rows = [json.loads(l) for l in open(args.data)]
    ds_tok = ProbeDataset(tok, [r["prompt"] for r in rows], [r["generation"] for r in rows],
                          [float(r["score"]) for r in rows])
    seqs, plens, labels = [], [], []
    for it in ds_tok:
        ids = it["input_ids"][:args.max_len] if args.max_len else it["input_ids"]
        seqs.append(ids); plens.append(min(it["prompt_len"], len(ids))); labels.append(it["label"])

    with tempfile.TemporaryDirectory() as cache:
        vllm_extract.write_feature_cache(llm, cache, seqs, plens, labels)
        thr = calibrate_from_dataset(FeatureCacheDataset(cache), head, tau=args.tau, device=device)
    print(f"# calibrated threshold (tau={args.tau}, n={len(rows)}) = {thr:.4f}", flush=True)

    prev_meta = {}
    if os.path.exists(args.head + ".meta.json"):
        prev_meta = load_probe_meta(args.head).get("meta", {})
    save_probe_checkpoint(args.head, head, feature_spec=DEFAULT_SPEC, threshold=thr, meta=prev_meta)
    print(f"[OK] wrote threshold to {args.head}.meta.json", flush=True)


if __name__ == "__main__":
    main()
