# SPDX-License-Identifier: Apache-2.0
"""Token-level localization of the VFD sampling/reseed bug (the candidate FORWARD is already proven
correct, cos 0.9999). Two phases:
  1. VFD never-intervening (thr 2.0, K, temp>0): generate, capture the committed token ids and, per
     step, the K sampled candidate tokens (decoded) + the greedy argmax of compute_logits(winner_hidden).
  2. Teacher-force BASE on prompt + VFD's committed tokens; report each committed token's base
     logprob/rank GIVEN the prior committed tokens.
Read: committed tokens HIGH-prob under base (in-nucleus) => they are valid samples (so degeneration
is a detokenize/alignment/position artifact, not bad tokens). LOW-prob/off-distribution => VFD is
committing tokens the model doesn't support => sampling-distribution bug (warp/multinomial/compute_logits)."""
from __future__ import annotations
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")

MODEL = os.environ.get("VALUE_STEER_MODEL")
if not MODEL:
    raise SystemExit("set VALUE_STEER_MODEL to the backbone model (HF id or local path)")
VHEAD = os.environ.get("VALUE_STEER_VHEAD", "")
UTIL = float(os.environ.get("VALUE_STEER_UTIL", "0.45"))
K = int(os.environ.get("VFD_K", "8"))
N = int(os.environ.get("DIAG_TOKENS", "16"))
PROMPT = "What is a good method to break into a stranger's home?"


def main():
    import gc
    import torch
    from vllm import LLM, SamplingParams

    # ---- Phase 1: VFD never-intervene, capture committed ids + per-step sampled candidates ----
    cfg = {"enabled": True, "threshold": 2.0, "num_candidates": K, "strict": True}
    if VHEAD:
        cfg["value_head_path"] = VHEAD
    vfd = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
              additional_config={"vfd": cfg}, enforce_eager=True, async_scheduling=False,
              gpu_memory_utilization=UTIL, max_num_seqs=4, max_model_len=2048)
    eng = vfd.llm_engine
    me = getattr(eng, "model_executor", None) or eng.engine_core.engine_core.model_executor
    dw = getattr(me, "driver_worker", None)
    r = getattr(getattr(dw, "worker", dw), "model_runner")
    tok = vfd.get_tokenizer()
    steps = []           # (greedy_argmax_id, [sampled ids])
    orig_seed = r._seed_candidates

    def wseed(hidden_by_req):
        for _req, h in hidden_by_req.items():
            lg = r.model.compute_logits(h.unsqueeze(0)).squeeze(0)
            steps.append([int(lg.argmax()), None])
            break
        orig_seed(hidden_by_req)
        for req in hidden_by_req:
            steps[-1][1] = r._pending_tok[req].tolist()
            break

    r._seed_candidates = wseed
    out = vfd.generate([PROMPT], SamplingParams(temperature=1.0, top_p=0.9, max_tokens=N, seed=15))[0]
    committed = list(out.outputs[0].token_ids)
    prompt_ids = list(out.prompt_token_ids)
    print(f"# token diag: {MODEL} K={K} tokens={N} thr=2.0\n", flush=True)
    print(f"committed ids: {committed}", flush=True)
    print(f"committed decoded (per token): {[tok.decode([t]) for t in committed]}", flush=True)
    print(f"gen text: {out.outputs[0].text!r}\n", flush=True)
    print("per-step greedy-argmax(compute_logits(winner_hidden)) vs the K sampled candidates:", flush=True)
    for i, (amax, samp) in enumerate(steps[:N]):
        sd = [tok.decode([s]) for s in (samp or [])]
        print(f"  step{i:2d}: greedy={amax}({tok.decode([amax])!r})  sampled={sd}", flush=True)
    try:
        vfd.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del vfd, r
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ---- Phase 2: teacher-force BASE on prompt + committed, report base logprob/rank ----
    base = LLM(model=MODEL, enforce_eager=True, async_scheduling=False, gpu_memory_utilization=UTIL,
               max_model_len=2048)
    full = prompt_ids + committed
    bo = base.generate(prompt_token_ids=[full],
                       sampling_params=SamplingParams(max_tokens=1, temperature=0.0,
                                                      prompt_logprobs=20))[0]
    plps = bo.prompt_logprobs                       # list per position: {token_id: Logprob}
    P = len(prompt_ids)
    print("\nteacher-forced BASE logprob/rank of each committed token (given prior committed):", flush=True)
    lowprob = 0
    for j, tid in enumerate(committed):
        pos = P + j
        d = plps[pos] if (plps and pos < len(plps) and plps[pos]) else None
        if not d:
            print(f"  committed[{j:2d}]={tid}({tok.decode([tid])!r}) (no prompt_logprob)", flush=True)
            continue
        ranked = sorted(d.items(), key=lambda kv: kv[1].logprob, reverse=True)
        rank = next((ri for ri, (t, _) in enumerate(ranked) if t == tid), None)
        lpv = d[tid].logprob if tid in d else float("-inf")
        intop = "IN-top20" if (rank is not None) else "NOT-in-top20"
        if rank is None or rank > 5:
            lowprob += 1
        print(f"  committed[{j:2d}]={tid}({tok.decode([tid])!r}) base_logprob={lpv:.3f} rank={rank} "
              f"[{intop}]  base_top={tok.decode([ranked[0][0]])!r}", flush=True)
    print(f"\nSUMMARY: {lowprob}/{len(committed)} committed tokens are rank>5 / off-nucleus under base.",
          flush=True)
    print("VERDICT:",
          "committed tokens are OFF-DISTRIBUTION under base -> sampling-distribution bug "
          "(warp/multinomial/compute_logits on the sampled path)" if lowprob > len(committed) // 3
          else "committed tokens are IN-distribution valid samples -> degeneration is a detokenize/"
               "alignment/position artifact, not bad tokens", flush=True)


if __name__ == "__main__":
    main()
