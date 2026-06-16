# SPDX-License-Identifier: Apache-2.0
"""
Diagnose why a freshly-trained value head underperforms the canonical one.

Two questions, one job:
  (1) FEATURE PARITY -- is our TRAINING feature (vLLM pooling path) the same tensor as the
      reference feature (HF post-norm last_hidden_state, which the canonical head trained on and
      which == the vLLM decode feature VFD scores)? Per-token cosine between the two.
  (2) HEAD DISCRIMINATION -- AUC of each head's per-example score (max p_unsafe over the response)
      vs the true label, for BOTH heads on BOTH feature sources. Disambiguates:
        * our head low on its OWN (pooling) feature  -> training is weak (recipe), not feature;
        * our head ok on pooling but parity low       -> train/inference feature MISMATCH (the bug);
        * canonical high, ours low everywhere         -> our weights are bad regardless.

Usage:
  python scripts/feature_parity_diag.py --model <m> --data <test.jsonl> --n 128 \
      --head-ours trained/safety_full/value_head.bin --head-canon <canonical value_head.bin>
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

from value_steer.train_probe import ProbeDataset, extract_features, FeatureCacheDataset
from value_steer.value_probe import load_value_head


def auc(scores, labels):
    """AUROC via Mann-Whitney (no sklearn dep). labels: 1=unsafe (positive)."""
    pos = [s for s, l in zip(scores, labels) if l >= 0.5]
    neg = [s for s, l in zip(scores, labels) if l < 0.5]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@torch.no_grad()
def head_scores(head, hidden_list, plens, device):
    """Per-example score = max P(undesirable) over the RESPONSE tokens (matches calibration)."""
    out = []
    for hs, pl in zip(hidden_list, plens):
        p = head.p(hs.to(device))            # [L]
        out.append(float(p[pl:].max()) if pl < hs.shape[0] else float(p.max()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--head-ours", required=True)
    ap.add_argument("--head-canon", default=None, help="optional baseline head (e.g. the canonical one)")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(args.data)][: args.n]
    ds = ProbeDataset(tok, [r["prompt"] for r in rows], [r["generation"] for r in rows],
                      [float(r["score"]) for r in rows])
    seqs, plens, labels = [], [], []
    for it in ds:
        ids = it["input_ids"][: args.max_len]
        seqs.append(ids); plens.append(min(it["prompt_len"], len(ids))); labels.append(it["label"])
    print(f"# {len(seqs)} examples, {sum(l>=0.5 for l in labels)} unsafe", flush=True)

    # ---- HF post-norm features (reference / ~inference feature) ----
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
    bb = hf.model
    hf_hidden = []
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor(s, device=device).unsqueeze(0)
            hs = extract_features(bb, ids, torch.ones_like(ids))[0]   # [L,H]
            hf_hidden.append(hs.float().cpu())
    H = hf_hidden[0].shape[-1]
    del hf, bb
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ---- vLLM pooling features (our TRAINING feature) ----
    import vllm_extract
    llm = vllm_extract.build_pooling_llm(args.model, gpu_memory_utilization=0.4,
                                         max_model_len=max(args.max_len, 2048))
    with tempfile.TemporaryDirectory() as cache:
        vllm_extract.write_feature_cache(llm, cache, seqs, plens, labels)
        cds = FeatureCacheDataset(cache)
        pool_hidden = [cds[i]["features"].float() for i in range(len(seqs))]

    # ---- (1) feature parity: per-token cosine HF vs pooling ----
    cos_means, cos_mins = [], []
    for a, b in zip(hf_hidden, pool_hidden):
        n = min(a.shape[0], b.shape[0])
        c = torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=-1)   # [n]
        cos_means.append(float(c.mean())); cos_mins.append(float(c.min()))
    print(f"\n## FEATURE PARITY (HF vs vLLM-pooling, per-token cosine)", flush=True)
    print(f"   mean cosine = {sum(cos_means)/len(cos_means):.4f}   "
          f"min-over-examples of per-token-min = {min(cos_mins):.4f}", flush=True)

    # ---- (2) head discrimination: AUC for both heads on both features ----
    heads = [("ours", load_value_head(args.head_ours, H, device))]
    if args.head_canon and os.path.exists(args.head_canon):
        heads.append(("canonical", load_value_head(args.head_canon, H, device)))
    else:
        print(f"# (no canonical baseline: {args.head_canon})", flush=True)
    print(f"\n## HEAD DISCRIMINATION (AUROC, max-over-response p_unsafe vs true label)", flush=True)
    for hname, head in heads:
        for fname, feats in (("HF", hf_hidden), ("pooling", pool_hidden)):
            a = auc(head_scores(head, feats, plens, device), labels)
            print(f"   {hname:9s} on {fname:7s} features: AUC = {a:.4f}", flush=True)
    print("\n(AUC ~0.5 = no discrimination; canonical>>ours => our weights are weak; "
          "parity<<1 => train/inference feature mismatch.)", flush=True)


if __name__ == "__main__":
    main()
