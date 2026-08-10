# SPDX-License-Identifier: Apache-2.0
"""Bit-exact behavior-preservation gate for VFD-plugin refactors.

Config-aware to cover the branches a core refactor touches; each config keeps its own saved
reference of the exact token ids the plugin emits:
  main     : eager, single-stream R=1 + batched R=32 @ threshold  (the deployment path)
  compiled : enforce_eager=False + single_stream=True             (cudagraph / _run_captured path)
  churn    : eager, 64 prompts @ max_num_seqs=16                  (continuous re-entry / mixed-step)

Batched and churn are GPU-nondeterministic *across* batch shape (M), but a same-M re-run is
bit-identical, so with the prompt set and shapes fixed the gate is bit-exact run-to-run. Workflow:
`REFTEST_MODE=save` once on the baseline commit, then `REFTEST_MODE=check` after each refactor step
-- identical token ids per config => behavior preserved; any mismatch => the refactor changed
decoding. Run all three configs to cover eager + cudagraph + reentry.

Env:
  VALUE_STEER_MODEL   backbone (HF id or local path)      [mistralai/Mistral-7B-Instruct-v0.3]
  VALUE_STEER_VHEAD   trained value head (.bin)           [required]
  SAFETY_PROMPTS      prompts jsonl, one {"prompt": ...}  [required]
  REFTEST_MODE        save | check                        [check]
  REFTEST_CONFIG      main | compiled | churn             [main]
  REFTEST_DIR         directory for the reference json    [.]
  REFTEST_THR         intervention threshold              [0.3571]
  VFD_K               candidates per step                 [8]
  VALUE_STEER_UTIL    vLLM gpu_memory_utilization         [0.6]
"""
import json
import os
import statistics as st

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from vllm import LLM, SamplingParams

MODE = os.environ.get("REFTEST_MODE", "check")
CONFIG = os.environ.get("REFTEST_CONFIG", "main")
MODEL = os.environ.get("VALUE_STEER_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HEAD = os.environ["VALUE_STEER_VHEAD"]
PROMPTS_FILE = os.environ["SAFETY_PROMPTS"]
REF = os.path.join(os.environ.get("REFTEST_DIR", "."), f"reftest_ref_{CONFIG}.json")
THR = float(os.environ.get("REFTEST_THR", "0.3571"))
K = int(os.environ.get("VFD_K", "8"))
UTIL = float(os.environ.get("VALUE_STEER_UTIL", "0.6"))
SP = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=200, seed=15)


def toks(o):
    return list(o.outputs[0].token_ids)


def msgs(ps):
    return [[{"role": "user", "content": p}] for p in ps]


def load_prompts(n):
    out = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)["prompt"].strip())
            if len(out) >= n:
                break
    if len(out) < n:
        raise SystemExit(f"SAFETY_PROMPTS has {len(out)} prompts, need {n} for config {CONFIG}")
    return out


def agg(seqs):
    """Coarse degeneration summary (mean length + count of >=10-token repeat runs) -- a human sanity
    readout only; the gate itself is exact token-id equality, not this aggregate."""
    real = 0
    for s in seqs:
        m = run = 1
        for i in range(1, len(s)):
            run = run + 1 if s[i] == s[i - 1] else 1
            m = max(m, run)
        if m >= 10:
            real += 1
    return {"ntok": round(st.mean(len(s) for s in seqs), 1) if seqs else 0, "real": real, "n": len(seqs)}


cfg = {"enabled": True, "threshold": THR, "num_candidates": K, "strict": True, "value_head_path": HEAD}
common = dict(
    model=MODEL,
    worker_cls="value_steer.worker.ValueSteerWorker",
    additional_config={"vfd": cfg},
    async_scheduling=False,
    gpu_memory_utilization=UTIL,
    max_model_len=2048,
    enable_chunked_prefill=False,
)

if CONFIG == "compiled":
    P = 32
    cfg["single_stream"] = True
    llm = LLM(enforce_eager=False, max_num_seqs=max(K, 8), **common)
    prompts = load_prompts(P)
    # warm up with ONE prompt: single_stream sizes scratch for 1 request, so a 2-prompt warmup
    # would batch 2 concurrent -> exceed the scratch blocks.
    llm.chat(msgs(prompts[:1]), SamplingParams(max_tokens=8, temperature=0.0))
    out = [toks(llm.chat([[{"role": "user", "content": p}]], SP)[0]) for p in prompts]  # driven single-stream
    streams = {"compiled-single": out}
elif CONFIG == "churn":
    P, MNS = 64, 16
    llm = LLM(enforce_eager=True, max_num_seqs=MNS, **common)
    prompts = load_prompts(P)
    llm.chat(msgs(prompts[:1]), SamplingParams(max_tokens=8, temperature=0.0))
    out = [toks(o) for o in llm.chat(msgs(prompts), SP)]  # P submitted @ MNS -> continuous re-entry
    streams = {f"churn-{P}@{MNS}": out}
else:  # main
    P = 32
    llm = LLM(enforce_eager=True, max_num_seqs=P, **common)
    prompts = load_prompts(P)
    llm.chat(msgs(prompts[:1]), SamplingParams(max_tokens=8, temperature=0.0))
    ss = [toks(llm.chat([[{"role": "user", "content": p}]], SP)[0]) for p in prompts]
    bt = [toks(o) for o in llm.chat(msgs(prompts), SP)]
    streams = {"single-stream-R1": ss, f"batched-R{P}": bt}

cur = {k: {"seqs": v, "agg": agg(v)} for k, v in streams.items()}
if MODE == "save":
    with open(REF, "w") as f:
        json.dump(cur, f)
    print(f"### REFTEST[{CONFIG}] SAVED -> {REF}  " + " ".join(f"{k}:{cur[k]['agg']}" for k in cur))
else:
    with open(REF) as f:
        ref = json.load(f)
    print(f"### REFTEST[{CONFIG}] thr={THR} k={K}")
    ok = True
    for k in cur:
        if k not in ref:
            print(f"###   {k}: MISSING from reference (was it saved with a different config/shape?)")
            ok = False
            continue
        m = sum(1 for a, b in zip(cur[k]["seqs"], ref[k]["seqs"]) if a == b)
        n = len(cur[k]["seqs"])
        print(f"###   {k}: {m}/{n} identical   cur={cur[k]['agg']} ref={ref[k]['agg']}")
        ok = ok and m == n
    print(f"### REFTEST[{CONFIG}] {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
