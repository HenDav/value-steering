# SPDX-License-Identifier: Apache-2.0
"""
Decode-matched feature extraction for value-head training (closes the train/inference gap).

The value head is SCORED at inference on the hidden the VFD runner produces during DECODE, which
differs from a prefill/pooling extraction (~0.97 cosine, measured). So instead of pooling-prefill
features, capture features the way they're actually scored: GENERATE responses with the VFD runner
(never-intervene) while VFD_DUMP_HIDDEN records the exact per-token decode hidden, judge-label the
fresh generations, and train on those. Features then == inference features (response tokens only,
which is the regime VFD scores).

Phases (separate processes -- gen needs the VFD model, label needs the Llama judge):
  --phase gen   : VFD-generate + capture decode hidden -> <cache>/feats.f16 + index.jsonl (label=-1)
                  + <cache>/gen.jsonl {index, prompt, generation}
  --phase label : judge gen.jsonl, split by prompt, write <cache>/{train,val}/ feature caches
Then: train_value_head.py --phase train --cache-dir <cache> ; then test.
"""

from __future__ import annotations

import os
import sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VFD_DUMP_HIDDEN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import json

import numpy as np
import torch

import dataset_loaders


def _runner(llm):
    eng = llm.llm_engine
    me = getattr(eng, "model_executor", None) or eng.engine_core.engine_core.model_executor
    dw = getattr(me, "driver_worker", None)
    w = getattr(dw, "worker", dw)
    return w.model_runner


def do_gen(args):
    """VFD-generate (never intervene) and stream the captured decode hidden (response tokens) to a
    flat feature cache + a gen.jsonl for judging."""
    from vllm import LLM, SamplingParams
    recs = dataset_loaders.load_prompts("safety", args.source, args.n)
    llm = LLM(model=args.model, worker_cls="value_steer.worker.ValueSteerWorker",
              additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": 8,
                                         "strict": True, "value_head_path": args.head}},
              enforce_eager=True, async_scheduling=False, gpu_memory_utilization=args.util,
              max_num_seqs=args.max_num_seqs, max_model_len=2048)
    runner = _runner(llm)

    # samples_per_prompt > 1: emit S responses per prompt (data augmentation -- diverse
    # continuations + labels). Each replica gets a DISTINCT seed so temperature sampling
    # actually diverges (same seed would reproduce the same tokens); the token-keyed pairing
    # below tolerates the rare collision (two identical samples share features, both kept).
    S = max(1, args.samples_per_prompt)
    exp = [r for r in recs for _ in range(S)]                     # each prompt repeated S times

    os.makedirs(args.cache_dir, exist_ok=True)
    H, total, idx, skipped = None, 0, 0, 0
    fb = open(os.path.join(args.cache_dir, "feats.f16"), "wb")
    ix = open(os.path.join(args.cache_dir, "index.jsonl"), "w")
    gj = open(os.path.join(args.cache_dir, "gen.jsonl"), "w")
    # Batched chat for speed; pair each output to its captured hidden by the committed-token
    # sequence (== the output tokens under never-intervene), which is robust to req-id/order
    # mismatches in the per-request dump dict.
    for i in range(0, len(exp), args.gen_chunk):
        grp = exp[i:i + args.gen_chunk]
        sps = [SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens, seed=args.seed + i + j)  # distinct per request
               for j in range(len(grp))]
        runner._dump_hidden.clear()
        outs = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in grp], sps)
        by_toks = {}
        for dump in runner._dump_hidden.values():
            by_toks.setdefault(tuple(d[0] for d in dump), dump)   # key: committed-token seq
        for r, o in zip(grp, outs):
            gtoks = list(o.outputs[0].token_ids)
            dump = by_toks.get(tuple(gtoks))
            if not gtoks or dump is None:                 # no capture matched this exact output
                skipped += 1
                continue
            feat = torch.stack([d[1] for d in dump]).to(torch.float16).numpy()   # [len, H]
            if H is None:
                H = int(feat.shape[1])
            feat.tofile(fb)
            ix.write(json.dumps({"index": idx, "offset": total, "length": feat.shape[0],
                                 "prompt_len": 0, "label": -1}) + "\n")
            gj.write(json.dumps({"index": idx, "prompt": r["prompt"],
                                 "generation": o.outputs[0].text}) + "\n")
            total += feat.shape[0]
            idx += 1
    fb.close(); ix.close(); gj.close()
    with open(os.path.join(args.cache_dir, "meta.json"), "w") as f:
        json.dump({"H": H, "total_rows": total, "count": idx, "dtype": "float16"}, f)
    print(f"[OK] decode-extracted {idx} examples ({total} response tokens; {skipped} skipped on "
          f"capture-misalign) -> {args.cache_dir}", flush=True)


def do_label(args):
    """Judge gen.jsonl, split by prompt, and materialize <cache>/{train,val} feature caches."""
    import value_verifiers  # noqa: F401 registers safety
    from value_steer.verifiers import get_verifier
    verifier = get_verifier("safety", judge_model=args.judge_model)

    with open(os.path.join(args.cache_dir, "meta.json")) as f:
        meta = json.load(f)
    H = int(meta["H"])
    flat = np.memmap(os.path.join(args.cache_dir, "feats.f16"), dtype=np.float16, mode="r",
                     shape=(int(meta["total_rows"]), H))
    index = [json.loads(l) for l in open(os.path.join(args.cache_dir, "index.jsonl"))]
    gens = [json.loads(l) for l in open(os.path.join(args.cache_dir, "gen.jsonl"))]
    scores = verifier.score_batch([g["prompt"] for g in gens], [g["generation"] for g in gens],
                                  [None] * len(gens))
    print(f"# judged {len(gens)} ({sum(s>=0.5 for s in scores)} undesirable)", flush=True)

    import random
    prompts = sorted({g["prompt"] for g in gens})
    random.Random(args.seed).shuffle(prompts)
    val_prompts = set(prompts[: int(len(prompts) * args.val_split)])

    def write_split(name, keep):
        d = os.path.join(args.cache_dir, name)
        os.makedirs(d, exist_ok=True)
        tot = 0
        with open(os.path.join(d, "feats.f16"), "wb") as fb, open(os.path.join(d, "index.jsonl"), "w") as ix:
            for rec, g, s in zip(index, gens, scores):
                if not keep(g["prompt"]):
                    continue
                rows = np.array(flat[rec["offset"]: rec["offset"] + rec["length"]])
                rows.tofile(fb)
                ix.write(json.dumps({"offset": tot, "length": rec["length"], "prompt_len": 0,
                                     "label": float(s)}) + "\n")
                tot += rec["length"]
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"H": H, "total_rows": tot, "count": "split", "model": args.model}, f)
        print(f"# wrote {name} cache ({tot} rows) -> {d}", flush=True)

    write_split("train", lambda p: p not in val_prompts)
    write_split("val", lambda p: p in val_prompts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["gen", "label"], required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--model", default=os.environ.get("VALUE_STEER_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
                    help="backbone model (HF id or local path); env VALUE_STEER_MODEL overrides the default")
    ap.add_argument("--head", default="", help="any value head (feature is head-independent)")
    ap.add_argument("--source", default="")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--samples-per-prompt", type=int, default=1,
                    help="generate (and label) this many continuations per prompt; each gets a distinct seed")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--util", type=float, default=0.45)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=256)
    ap.add_argument("--judge-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--val-split", type=float, default=0.1)
    args = ap.parse_args()
    if args.phase == "gen":
        do_gen(args)
    else:
        do_label(args)


if __name__ == "__main__":
    main()
