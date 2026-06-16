# SPDX-License-Identifier: Apache-2.0
"""
Train a value head from a labeled jsonl ({prompt, generation, score}) -- the missing CLI around
value_steer.train_probe. Domain-agnostic: `score` is P(undesirable) in [0,1] from any verifier
(safety judge, math grader, ...); the default recipe is the canonical safety one (focal +
TD-coherence, coh_weight 0.1).

Two phases keep the big model OUT of the head-training process and stream features from disk so
dataset size is bounded by disk, not GPU/RAM:

    # 1. extract: load the backbone as a vLLM pooling model (full GPU util), stream per-token
    #    POST-final-norm features (the EXACT inference forward the runners score) to an on-disk
    #    cache, then EXIT -- so the model is gone, never co-resident with anything else.
    python scripts/train_value_head.py --phase extract --model <m> --data data.jsonl \
        --out value_head.bin --cache-dir vh.featcache

    # 2. train: NO model loaded -- memmap the cache, train the tiny head over it for many epochs
    #    (one frozen-backbone forward total), calibrate, write value_head.bin + .meta.json.
    python scripts/train_value_head.py --phase train --out value_head.bin --cache-dir vh.featcache --calibrate

`--phase all` does both in one process (frees the engine between) for convenience on small data;
use the split phases for strict GPU isolation and big datasets.
"""

from __future__ import annotations

import os
import sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import gc
import json
import random

import torch
from torch.utils.data import DataLoader

from value_steer.train_probe import (
    ProbeDataset, FeatureCacheDataset, FeatureCollator, train_probe, save_probe_checkpoint,
)
from value_steer.value_probe import ValueHead, DEFAULT_SPEC

_IDENTITY = lambda feats, attn: feats   # noqa: E731 -- cached features ARE the feature


def _split_by_prompt(rows, val_frac, seed):
    """Split rows into (train, val) by UNIQUE PROMPT (no prompt in both -- avoids leakage when
    samples_per_prompt > 1 puts multiple rows per prompt)."""
    prompts = sorted({r["prompt"] for r in rows})
    random.Random(seed).shuffle(prompts)
    val_prompts = set(prompts[: int(len(prompts) * val_frac)])
    train = [r for r in rows if r["prompt"] not in val_prompts]
    val = [r for r in rows if r["prompt"] in val_prompts]
    return train, val


def _tokenize(tok, rows, max_len):
    """(prompt, generation) -> lists (sequences, prompt_lens, labels) via the chat template."""
    ds = ProbeDataset(tok, [r["prompt"] for r in rows], [r["generation"] for r in rows],
                      [float(r["score"]) for r in rows])
    seqs, plens, labels = [], [], []
    for it in ds:
        ids = it["input_ids"][:max_len] if max_len else it["input_ids"]
        seqs.append(ids)
        plens.append(min(it["prompt_len"], len(ids)))
        labels.append(it["label"])
    return seqs, plens, labels


def do_extract(args):
    """Load the pooling model, split + tokenize, stream features to <cache-dir>/{train,val}.

    NOTE: this is a PREFILL extraction and carries a train/inference feature mismatch (~0.97 cos vs
    the decode hidden VFD scores). For training, prefer scripts/decode_extract.py (decode-matched);
    `--phase train` here works on either cache. See docs/training-a-value-head.md."""
    import vllm_extract
    llm = vllm_extract.build_pooling_llm(args.model, gpu_memory_utilization=args.util,
                                         max_model_len=args.max_model_len)
    tok = llm.get_tokenizer()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        raise ValueError("tokenizer has no pad_token_id and no eos_token to fall back on")
    if getattr(tok, "chat_template", None) is None:
        raise ValueError("tokenizer has no chat_template; ProbeDataset needs apply_chat_template")

    rows = [json.loads(l) for l in open(args.data)]
    train_rows, val_rows = _split_by_prompt(rows, args.val_split, args.seed)
    extra = {"model": args.model, "domain": args.domain}
    for name, rs in (("train", train_rows), ("val", val_rows)):
        seqs, plens, labels = _tokenize(tok, rs, args.max_len)
        vllm_extract.write_feature_cache(llm, os.path.join(args.cache_dir, name),
                                         seqs, plens, labels, extra_meta=extra)
        print(f"# extracted {len(seqs)} {name} examples -> {args.cache_dir}/{name}", flush=True)
    return llm


def _loader(cache_subdir, batch_size, shuffle, *, num_workers, pin_memory):
    # Reading the on-disk feature memmap (random access under shuffle) is the training-phase
    # bottleneck -- the head compute is trivial -- so parallelize loading with worker processes
    # (each forks its own memmap view) + pinned memory for async H2D. persistent_workers avoids
    # re-spawning every epoch.
    ds = FeatureCacheDataset(cache_subdir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=FeatureCollator(),
                        num_workers=num_workers, pin_memory=pin_memory,
                        persistent_workers=num_workers > 0)
    return ds, loader


def do_train(args):
    """Memmap the cache, train the head over it, calibrate, write checkpoint + sidecar. No model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ld = dict(num_workers=args.num_workers, pin_memory=device == "cuda")
    train_ds, train_loader = _loader(os.path.join(args.cache_dir, "train"), args.batch_size, True, **ld)
    val_dir = os.path.join(args.cache_dir, "val")
    has_val = os.path.exists(os.path.join(val_dir, "meta.json")) and len(
        FeatureCacheDataset(val_dir)) > 0
    val_ds, val_loader = _loader(val_dir, args.batch_size, False, **ld) if has_val else (None, None)
    hidden = train_ds.H
    print(f"# train={len(train_ds)} val={len(val_ds) if val_ds else 0} hidden={hidden} device={device}",
          flush=True)

    head = ValueHead(hidden)
    train_probe(None, head, train_loader, epochs=args.epochs, lr=args.lr, loss_name=args.loss,
                use_td=not args.no_td, coh_weight=args.coh_weight, val_loader=val_loader,
                patience=args.patience, device=device, feature_fn=_IDENTITY,
                warmup_frac=args.warmup_frac, verbose=True)

    threshold = None
    if args.calibrate:
        if val_ds is None:
            print("# WARNING: --calibrate but val cache is empty; skipping calibration", flush=True)
        else:
            from calibrate_threshold import calibrate_from_dataset
            threshold = calibrate_from_dataset(val_ds, head, tau=args.cal_tau, device=device)
            print(f"# calibrated threshold (tau={args.cal_tau}) = {threshold:.4f}", flush=True)

    meta = {"domain": args.domain, "model": train_ds.meta.get("model"), "loss": args.loss,
            "coh_weight": args.coh_weight, "use_td": not args.no_td, "epochs": args.epochs,
            "n_train": len(train_ds),
            "recipe": "focal+td+coh0.1" if args.loss == "focal" else args.loss}
    wpath, spath = save_probe_checkpoint(args.out, head, feature_spec=DEFAULT_SPEC,
                                         threshold=threshold, meta=meta)
    print(f"[OK] wrote {wpath} + {spath}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["extract", "train", "all"], default="all")
    ap.add_argument("--model", default=os.environ.get("VALUE_STEER_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
                    help="backbone model (HF id or local path); env VALUE_STEER_MODEL overrides the default")
    ap.add_argument("--data", help="labeled jsonl (extract phase)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None, help="feature cache dir (default: <out>.featcache)")
    ap.add_argument("--domain", default="safety")
    ap.add_argument("--loss", default="focal", choices=["focal", "bce"])
    ap.add_argument("--coh-weight", type=float, default=0.1)
    ap.add_argument("--no-td", action="store_true", help="disable the TD-coherence term")
    # Defaults match the reference recipe that produced the canonical steering head.
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.1, help="linear warmup fraction (+decay); 0=constant LR")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=None, help="truncate each example to first N tokens")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for the cache reads")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--calibrate", action="store_true", help="calibrate + write threshold to the sidecar")
    ap.add_argument("--cal-tau", type=float, default=0.05)
    ap.add_argument("--util", type=float, default=0.85, help="vLLM gpu_memory_utilization for extraction")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.cache_dir = args.cache_dir or (args.out + ".featcache")

    if args.phase in ("extract", "all"):
        if not args.data:
            ap.error("--data is required for the extract phase")
        llm = do_extract(args)
    if args.phase == "all":
        del llm                       # best-effort free before head training (separate phases
        gc.collect()                  # give strict isolation; see the module docstring)
        torch.cuda.empty_cache()
    if args.phase in ("train", "all"):
        do_train(args)


if __name__ == "__main__":
    main()
