# SPDX-License-Identifier: Apache-2.0
"""
Decode-matched feature extraction for value-head training (closes the train/inference gap).

The value head is SCORED at inference on the hidden the VFD runner produces during DECODE, which
differs substantially from a prefill/pooling extraction. So instead of pooling-prefill
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
    # VFD needs a SINGLE KV-cache group. Models with mixed attention (e.g. Gemma's alternating
    # sliding/full) otherwise split into per-attention-type groups and hit the runner's fail-fast.
    # Disabling the hybrid KV-cache manager unifies them onto the full-attention spec (sliding
    # window becomes an in-kernel mask over full KV), which the candidate forward handles normally.
    extra = {"disable_hybrid_kv_cache_manager": True} if args.disable_hybrid_kv_cache else {}
    if args.force_arch:                    # e.g. load Gemma-4 as text-only Gemma4ForCausalLM (the
        extra["hf_overrides"] = {"architectures": [args.force_arch]}  # paper's arch; skips the vision wrapper
    # never-intervene capture draws the committed token from the natural distribution for ANY K, so
    # K=1 yields training features from the SAME distribution at ~K x fewer forward rows/step (the
    # specific samples differ via batch numerics, but the head learns the distribution, not tokens).
    llm = LLM(model=args.model, worker_cls="value_steer.worker.ValueSteerWorker",
              additional_config={"vfd": {"enabled": True, "threshold": 2.0,
                                         "num_candidates": args.num_candidates,
                                         "strict": True, "value_head_path": args.head,
                                         "single_stream": args.single_stream}},
              enforce_eager=not args.compiled, async_scheduling=False, gpu_memory_utilization=args.util,
              max_num_seqs=args.max_num_seqs, max_model_len=2048, **extra)
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
        sps = [SamplingParams(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
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
    # Reasoning models (Gemma-4 <|channel>thought..<channel|>, Qwen/DeepSeek <think>..</think>) emit
    # CoT before the answer. Judge the FINAL answer only (matches llm_safety.strip_thinking); the
    # captured trajectory still spans every decoded token, which is what VFD scores at inference.
    def _strip_thinking(t):
        import re
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
        t = re.sub(r"<\|channel>thought.*?<channel\|>", "", t, flags=re.DOTALL)
        return t.strip()
    gen_texts = [(_strip_thinking(g["generation"]) if args.strip_thinking else g["generation"]) for g in gens]
    scores = verifier.score_batch([g["prompt"] for g in gens], gen_texts, [None] * len(gens))
    print(f"# judged {len(gens)} ({sum(s>=0.5 for s in scores)} undesirable)", flush=True)
    # durable labeled-generations record (every output sample + its judge score) for post-hoc eval
    with open(os.path.join(args.cache_dir, "gen_scored.jsonl"), "w") as f:
        for g, s in zip(gens, scores):
            f.write(json.dumps({"index": g["index"], "prompt": g["prompt"],
                                "generation": g["generation"], "score": float(s)}) + "\n")

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


def do_calibrate(args):
    """Decode-matched conformal calibration: score the head over each labeled example's captured
    decode trajectory and call value_steer.calibration.posterior_threshold (the conformal +1
    quantile over the SAFE examples, tau=alpha) at each alpha. Writes the ĉ(alpha) curve to
    <cache>/thresholds.json. Reads a labeled cache (as written by --phase label)."""
    from value_steer.value_probe import load_value_head
    from value_steer.train_probe import FeatureCacheDataset
    from value_steer import calibration

    with open(os.path.join(args.cache_dir, "meta.json")) as f:
        H = int(json.load(f)["H"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = load_value_head(args.head, H, device)
    ds = FeatureCacheDataset(args.cache_dir)

    trajectories, safe_labels = [], []                     # score each captured trajectory ONCE
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            probs = head.p(item["features"].to(device))    # [Li] P(undesirable) in (0,1)
            trajectories.append(probs[item["prompt_len"]:].float().tolist())
            safe_labels.append(int(item["label"] >= 0.5))   # 1 = undesirable, 0 = safe/good
    n_safe = sum(1 for s in safe_labels if s == 0)

    alphas = [float(a) for a in args.alphas.split(",")]
    curve = [calibration.posterior_threshold(safe_labels, trajectories, tau=a) for a in alphas]
    out = {"alphas": alphas, "scalar": curve, "n": len(ds), "n_safe": n_safe,
           "model": args.model, "head": os.path.abspath(args.head)}
    with open(os.path.join(args.cache_dir, "thresholds.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Also fold the curve into the head's sidecar, so `--phase calibrate` leaves the head carrying
    # its ĉ(α) curve the way the published heads do (keeps `threshold` = a single default if present).
    sidecar_path = args.head + ".meta.json"
    sidecar = json.load(open(sidecar_path)) if os.path.exists(sidecar_path) else {}
    sidecar.setdefault("feature_spec", {"layer": "final", "norm": "post", "pooling": "none", "dtype": "fp32"})
    sidecar.setdefault("threshold", None)
    sidecar.setdefault("meta", {})["calibration"] = {
        "method": "decode-matched conformal posterior_threshold",
        "alphas": alphas, "thresholds": [round(c, 6) for c in curve],
        "n": len(ds), "n_safe": n_safe,
        "guarantee": "P(intervene | safe) <= alpha",
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[calibrate] n={len(ds)} n_safe={n_safe}  alphas={alphas}", flush=True)
    print("[calibrate] scalar c(alpha) = " + " ".join(f"{c:.4f}" for c in curve), flush=True)
    print(f"[calibrate] curve -> {sidecar_path} (meta.calibration) + {args.cache_dir}/thresholds.json", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["gen", "label", "calibrate"], required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--model", default=os.environ.get("VALUE_STEER_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
                    help="backbone model (HF id or local path); env VALUE_STEER_MODEL overrides the default")
    ap.add_argument("--head", default="", help="any value head (feature is head-independent)")
    ap.add_argument("--source", default="")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--samples-per-prompt", type=int, default=1,
                    help="generate (and label) this many continuations per prompt; each gets a distinct seed")
    ap.add_argument("--max-tokens", type=int, default=400)   # match llm_safety (--max_new_tokens 400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)         # match llm_safety (top_k=50, no top_p)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--util", type=float, default=0.45)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=256)
    ap.add_argument("--judge-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--single-stream", action="store_true",
                    help="gen: R=1 scratch reserve (pair with --max-num-seqs 1 for a true single-stream decode)")
    ap.add_argument("--num-candidates", type=int, default=8,
                    help="gen: VFD K. For never-intervene capture the natural-sample distribution is K-independent, so K=1 is fastest (~K x fewer forward rows) and yields an equivalent training distribution.")
    ap.add_argument("--compiled", action="store_true",
                    help="gen: compile + cudagraph the decode (enforce_eager=False) -- faster throughput and matches a compiled inference runtime.")
    ap.add_argument("--disable-hybrid-kv-cache", action="store_true",
                    help="gen: unify mixed-attention models (e.g. Gemma sliding/full) onto one KV-cache group so VFD's single-group surgery applies")
    ap.add_argument("--force-arch", default="",
                    help="gen: override the model's architecture (hf_overrides), e.g. Gemma4ForCausalLM to load Gemma-4 text-only")
    ap.add_argument("--strip-thinking", action="store_true",
                    help="label: strip reasoning-model CoT (Gemma <|channel>thought, Qwen <think>) before judging the answer")
    ap.add_argument("--alphas", default="0.05,0.25,0.45,0.65,0.85",
                    help="calibrate: comma-separated intervention error budgets (tau) for the c(alpha) curve")
    args = ap.parse_args()
    if args.phase == "gen":
        do_gen(args)
    elif args.phase == "label":
        do_label(args)
    else:
        do_calibrate(args)


if __name__ == "__main__":
    main()
