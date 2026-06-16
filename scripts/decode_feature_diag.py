# SPDX-License-Identifier: Apache-2.0
"""
Definitive train-vs-inference FEATURE check.

The value head is TRAINED on features from the vLLM *pooling* path (a prefill over the full
sequence) but SCORED at inference on the hidden the VFD runner computes during *decode* (its
candidate-forward, scratch-KV path). This measures whether those are the same tensor.

  1. Run VFD with threshold=2.0 (NEVER intervene -> commits the natural sampled token) and
     VFD_DUMP_HIDDEN=1, so the runner records the EXACT post-norm hidden it scored for each
     committed token (runner._dump_hidden).
  2. Pooling-extract the same (prompt + generated) token sequences -> per-token hidden.
  3. Align: decode-hidden[t] (gen token t) vs pooling-hidden[prompt_len + t]. Report cosine/relL2.

cos ~1 => train feature == inference feature (the weak head is a weights issue);
cos << 1 => the head is trained on a different tensor than it's scored on (the bug).
"""

from __future__ import annotations

import os
import sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VFD_DUMP_HIDDEN"] = "1"      # turn on the runner's hidden dump
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import json

import torch


def _runner(llm):
    eng = llm.llm_engine
    me = getattr(eng, "model_executor", None) or eng.engine_core.engine_core.model_executor
    dw = getattr(me, "driver_worker", None)
    w = getattr(dw, "worker", dw)
    return w.model_runner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True, help="any value head (feature is head-independent)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = []
    for line in open(args.data):
        p = json.loads(line).get("prompt", "").strip()
        if p:
            prompts.append(p)
        if len(prompts) >= args.n:
            break

    # ---- 1. VFD decode, never-intervene, dump the scored hidden ----
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, worker_cls="value_steer.worker.ValueSteerWorker",
              additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": 8,
                                         "strict": True, "value_head_path": args.head}},
              enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.45,
              max_num_seqs=16, max_model_len=2048)
    runner = _runner(llm)
    sp = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=args.max_tokens, seed=0)

    seqs, captured = [], []
    for p in prompts:
        runner._dump_hidden.clear()
        out = llm.generate([p], sp)[0]
        ptoks, gtoks = list(out.prompt_token_ids), list(out.outputs[0].token_ids)
        dumped = next(iter(runner._dump_hidden.values()), [])
        dtoks = [d[0] for d in dumped]
        dhid = [d[1] for d in dumped]                       # decode hidden per gen step (fp32 cpu)
        m = min(len(gtoks), len(dtoks))
        if dtoks[:m] != gtoks[:m]:
            print(f"# WARN: committed-token/output mismatch (m={m}); aligning on min", flush=True)
        seqs.append(ptoks + gtoks[:m])
        captured.append((len(ptoks), gtoks[:m], torch.stack(dhid[:m]) if dhid else None))

    import gc
    del llm; gc.collect(); torch.cuda.empty_cache()

    # ---- 2. pooling extraction of the same sequences ----
    import tempfile
    import vllm_extract
    from value_steer.train_probe import FeatureCacheDataset
    pllm = vllm_extract.build_pooling_llm(args.model, gpu_memory_utilization=0.4, max_model_len=2048)
    with tempfile.TemporaryDirectory() as cache:
        vllm_extract.write_feature_cache(pllm, cache, seqs, [0] * len(seqs), [0.0] * len(seqs))
        cds = FeatureCacheDataset(cache)
        pool = [cds[i]["features"].float() for i in range(len(seqs))]

    # ---- 3. compare decode-hidden[t] vs pooling-hidden[prompt_len + t + off] for off in {-1,0,+1}
    #         to disambiguate a genuine decode-vs-prefill gap from a position off-by-one. ----
    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    cos0, best_cos, best_off = [], [], {-1: 0, 0: 0, 1: 0}
    for (plen, gtoks, dhid), ph in zip(captured, pool):
        if dhid is None:
            continue
        for t in range(len(gtoks)):
            a = dhid[t]
            scored = {off: cos(a, ph[plen + t + off]) for off in (-1, 0, 1)
                      if 0 <= plen + t + off < ph.shape[0]}
            cos0.append(scored.get(0, float("nan")))
            bo = max(scored, key=scored.get)
            best_cos.append(scored[bo]); best_off[bo] += 1
    import statistics as st
    cos0 = [c for c in cos0 if c == c]
    print(f"\n## DECODE (inference) vs POOLING (train) hidden -- {len(cos0)} generated tokens",
          flush=True)
    print(f"   cosine @offset 0 (aligned): mean={st.mean(cos0):.4f} min={min(cos0):.4f}", flush=True)
    print(f"   cosine @best-of-3-offsets:  mean={st.mean(best_cos):.4f} min={min(best_cos):.4f}", flush=True)
    print(f"   best-offset histogram (which position matched best): {best_off}", flush=True)
    print("   -> if best-offset is mostly 0 and ~0.97: a REAL decode-vs-prefill gap;", flush=True)
    print("      if mostly +/-1 and jumps to ~0.999: it was my indexing (artifact).", flush=True)
    print("\n(cos~1 => train feature == inference feature -> weak head is weights, not features;\n"
          " cos<<1 => trained on a different tensor than scored on -> THE bug.)", flush=True)


if __name__ == "__main__":
    main()
