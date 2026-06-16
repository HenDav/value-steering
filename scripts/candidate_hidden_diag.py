# SPDX-License-Identifier: Apache-2.0
"""Pin the VFD degeneration bug: is the K-candidate forward's per-candidate hidden CORRECT when the
K candidates are DIFFERENT tokens (temp>0)? greedy (all K identical) is bit-exact to base, so the
diverse-K hidden path was never validated. Here, for each step we sample K diverse candidates and,
for EACH candidate k, compute an INDEPENDENT ground-truth hidden -- a single-row decode of token k
over the REAL prefix blocks (write position p to the real tail slot; sequential so no collision;
VFD's own commit overwrites it after, so generation is undisturbed) -- and compare to VFD's
h_cand[i,k] (scratch path). cos ~ 1 => forward is fine (bug is in reseed/sampling glue); cos << 1
=> the diverse-K candidate forward hidden is WRONG (localizes to scratch/slot/attention metadata),
with col 0 (the COMMITTED candidate) being the one that seeds the next step.
"""
from __future__ import annotations
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
import statistics

MODEL = os.environ["VALUE_STEER_MODEL"]
VHEAD = os.environ.get("VALUE_STEER_VHEAD", "")
UTIL = float(os.environ.get("VALUE_STEER_UTIL", "0.45"))
K = int(os.environ.get("VFD_K", "8"))
N = int(os.environ.get("DIAG_TOKENS", "16"))
PROMPT = "What is a good method to break into a stranger's home?"


def main():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.forward_context import set_forward_context

    cfg = {"enabled": True, "threshold": 2.0, "num_candidates": K, "strict": True}  # never intervene
    if VHEAD:
        cfg["value_head_path"] = VHEAD
    llm = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
              additional_config={"vfd": cfg}, enforce_eager=True, async_scheduling=False,
              gpu_memory_utilization=UTIL, max_num_seqs=4, max_model_len=2048)
    eng = llm.llm_engine
    me = getattr(eng, "model_executor", None) or eng.engine_core.engine_core.model_executor
    dw = getattr(me, "driver_worker", None)
    r = getattr(getattr(dw, "worker", dw), "model_runner")
    bs = r.cache_config.block_size
    dev = r.device
    stats = []           # (step, k, cos, rel)
    step = [0]
    orig = r._candidate_forward

    def wrapped(active):
        ret = orig(active)                       # scratch path: (h_cand, scratch_idx, plan)
        h_cand = ret[0]                          # [R,K,H]
        R, Kc, _ = h_cand.shape
        pbt = r.input_batch.block_table[0].get_device_tensor(r.input_batch.num_reqs)
        for i, a in enumerate(active):
            ridx = r.input_batch.req_id_to_index[a]
            p = r._next_pos[a]
            tail, off = p // bs, p % bs
            toks = r._pending_tok[a]             # [K] diverse sampled tokens
            for k in range(Kc):
                ref_bt = pbt[ridx].unsqueeze(0).clone()                       # [1, W] real blocks
                ref_slot = torch.tensor([int(pbt[ridx, tail]) * bs + off], device=dev, dtype=torch.long)
                qsl = torch.arange(2, device=dev, dtype=torch.int32)
                sl = torch.tensor([p + 1], device=dev, dtype=torch.int32)
                cm = CommonAttentionMetadata(
                    query_start_loc=qsl, query_start_loc_cpu=qsl.cpu(), seq_lens=sl, num_reqs=1,
                    num_actual_tokens=1, max_query_len=1, max_seq_len=p + 1,
                    block_table_tensor=ref_bt, slot_mapping=ref_slot, causal=True)
                md = r._build_per_layer_metadata(cm)
                smap = {ln: cm.slot_mapping
                        for g in r.kv_cache_config.kv_cache_groups for ln in g.layer_names}
                with set_forward_context(md, r.vllm_config, num_tokens=1, slot_mapping=smap):
                    rh = r._model_forward(input_ids=toks[k:k + 1],
                                          positions=torch.tensor([p], device=dev, dtype=torch.long))
                if not isinstance(rh, torch.Tensor):
                    rh = rh[0] if isinstance(rh, (tuple, list)) else rh.last_hidden_state
                a_ = h_cand[i, k].float()
                b_ = rh.reshape(-1).float()
                cos = float(torch.nn.functional.cosine_similarity(a_, b_, dim=0))
                rel = float((a_ - b_).norm() / (b_.norm() + 1e-9))
                stats.append((step[0], k, cos, rel))
        step[0] += 1
        return ret

    r._candidate_forward = wrapped
    print(f"# candidate-hidden diag: {MODEL} K={K} tokens={N} thr=2.0 (never intervene)\n", flush=True)
    llm.generate([PROMPT], SamplingParams(temperature=1.0, top_p=0.9, max_tokens=N, seed=15))

    allc = [c for (_, _, c, _) in stats]
    col0 = [c for (_, k, c, _) in stats if k == 0]
    print(f"\nsteps*K={len(stats)}  min_cos_all={min(allc):.4f}  mean_cos_all={statistics.mean(allc):.4f}",
          flush=True)
    print(f"col0 (COMMITTED): min_cos={min(col0):.4f}  mean={statistics.mean(col0):.4f}", flush=True)
    for s in sorted(set(x[0] for x in stats)):
        row = [x for x in stats if x[0] == s]
        c0 = [x[2] for x in row if x[1] == 0][0]
        worst = min(x[2] for x in row)
        rel0 = [x[3] for x in row if x[1] == 0][0]
        print(f"  step {s:2d}: col0_cos={c0:.4f} rel={rel0:.3e}  worst_cand_cos={worst:.4f}", flush=True)
    print("\nVERDICT:",
          "candidate-forward hidden WRONG for diverse tokens (col0 cos << 1) -> bug in the K-candidate "
          "forward (scratch/slot/attn metadata)" if min(col0) < 0.99
          else "candidate-forward hidden OK (cos~1) -> bug is in the reseed/sampling glue, not the forward",
          flush=True)


if __name__ == "__main__":
    main()
