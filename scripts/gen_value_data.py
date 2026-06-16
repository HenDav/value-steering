# SPDX-License-Identifier: Apache-2.0
"""
Generate + verify-label training data for a value head (domain-pluggable).

Pipeline: prompts (per domain) -> vLLM generates responses -> a Verifier labels each
with P(undesirable) in [0,1] -> write the canonical jsonl that
scripts/train_value_head.py consumes: {index, prompt, generation, score, meta}.

Two phases (7B vLLM weights don't free cleanly in-process, so safety -- which loads a
judge model after generation -- runs them as SEPARATE processes):

    # safety (judge needs its own process so the generator's GPU mem is freed first)
    python scripts/gen_value_data.py --domain safety --phase gen    --model <m> --source <prompts.jsonl> --out data.jsonl
    python scripts/gen_value_data.py --domain safety --phase verify --judge-model NousResearch/Meta-Llama-3.1-8B-Instruct --out data.jsonl

    # a pure-CPU verifier (e.g. math, once implemented) can do everything in one call:
    python scripts/gen_value_data.py --domain math --phase all --model <m> --source <gsm8k.jsonl> --out data.jsonl

`gen` writes responses to `<out>.gen` (no score); `verify` reads that, scores, writes `<out>`.
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

import dataset_loaders


def _gen_path(out):
    return out + ".gen"


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def do_gen(args):
    """vLLM-generate responses; STREAM {index, prompt, generation, meta} rows to <out>.gen in
    chunks so RAM stays bounded for big prompt sets."""
    from vllm import LLM, SamplingParams
    recs = dataset_loaders.load_prompts(args.domain, args.source, args.n)
    print(f"# gen: domain={args.domain} prompts={len(recs)} "
          f"samples_per_prompt={args.samples_per_prompt} model={args.model}", flush=True)
    llm = LLM(model=args.model, gpu_memory_utilization=args.util, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed, n=args.samples_per_prompt)
    # Generate via the chat template (realistic assistant responses); the trainer re-applies the
    # same template, so prompt/response pairs round-trip consistently.
    gen_path = _gen_path(args.out)
    os.makedirs(os.path.dirname(gen_path) or ".", exist_ok=True)
    idx = 0
    with open(gen_path, "w") as fh:
        for grp in _chunks(recs, args.gen_chunk):
            outs = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in grp], sp)
            for r, o in zip(grp, outs):
                for comp in o.outputs:                  # samples_per_prompt completions
                    fh.write(json.dumps({"index": idx, "prompt": r["prompt"],
                                         "generation": comp.text, "meta": r["meta"]}) + "\n")
                    idx += 1
    print(f"[OK] wrote {idx} (prompt, generation) rows -> {gen_path}", flush=True)


def do_verify(args):
    """Score each generation with the domain verifier; STREAM the canonical labeled jsonl to
    <out> in chunks (bounded RAM for big sets)."""
    import value_verifiers  # noqa: F401 -- registers the safety verifier
    from value_steer.verifiers import get_verifier
    kwargs = {"judge_model": args.judge_model} if args.domain == "safety" else {}
    verifier = get_verifier(args.domain, **kwargs)

    n, pos = 0, 0
    with open(_gen_path(args.out)) as src, open(args.out, "w") as fh:
        for grp in _chunks_lines(src, args.gen_chunk):
            rows = [json.loads(l) for l in grp]
            scores = verifier.score_batch([r["prompt"] for r in rows],
                                          [r["generation"] for r in rows],
                                          [r.get("meta") for r in rows])
            for r, s in zip(rows, scores):
                fh.write(json.dumps({"index": r["index"], "prompt": r["prompt"],
                                     "generation": r["generation"], "score": float(s),
                                     "meta": r.get("meta", {})}) + "\n")
                n += 1
                pos += int(s >= 0.5)
    print(f"[OK] labeled {n} rows ({pos} undesirable / {n}) -> {args.out}", flush=True)


def _chunks_lines(fh, n):
    buf = []
    for line in fh:
        buf.append(line)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--phase", choices=["gen", "verify", "all"], default="all")
    ap.add_argument("--model", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--samples-per-prompt", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--judge-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--util", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gen-chunk", type=int, default=1024, help="prompts/rows per streamed chunk")
    args = ap.parse_args()

    if args.phase in ("gen", "all"):
        if not args.model or not args.source:
            ap.error("--model and --source are required for the gen phase")
        do_gen(args)
    if args.phase == "all" and args.domain == "safety":
        print("# NOTE: safety 'all' loads the judge in the same process as the generator; "
              "7B vLLM weights may not free -> prefer separate --phase gen / --phase verify runs.",
              flush=True)
    if args.phase in ("verify", "all"):
        do_verify(args)


if __name__ == "__main__":
    main()
