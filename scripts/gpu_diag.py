# SPDX-License-Identifier: Apache-2.0
"""
GPU diagnostic for value-steer (run inside the SLURM a100 job).

Not a pytest -- a verbose, fail-soft probe that exercises every accelerator-bound seam
and PRINTS what actually happens (candidate shapes, p_unsafe, committed tokens, base vs
VFD divergence), so the first GPU contact is informative rather than a bare pass/fail.
Each section is independently guarded; a failure prints a traceback and the script
continues. Exit code is nonzero iff any section failed.

    VALUE_STEER_TEST_MODEL=facebook/opt-125m python scripts/gpu_diag.py
"""

from __future__ import annotations

import os

# In-process engine core (reach the live runner; avoid the forked-CUDA crash). Set
# before vLLM import.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# Use loopback IPv4 for the single-process distributed rendezvous: the node hostname
# resolves to IPv6 (c10d "Address family not supported"), which can stall init.
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")

import sys
import traceback
import faulthandler

# If any section stalls, dump every thread's stack every 90s so a hang is diagnosed
# in-line instead of via a wall-clock timeout.
faulthandler.dump_traceback_later(90, repeat=True, exit=False)

import torch

MODEL = os.environ.get("VALUE_STEER_TEST_MODEL", "facebook/opt-125m")
_failures: list[str] = []


def section(name):
    def deco(fn):
        print(f"\n{'='*72}\n# {name}\n{'='*72}", flush=True)
        try:
            fn()
            print(f"[OK] {name}", flush=True)
        except Exception:
            _failures.append(name)
            print(f"[FAIL] {name}", flush=True)
            traceback.print_exc()
        return fn
    return deco


class ConstHead:
    """Constant p_unsafe head; arch-agnostic, deterministic decisions."""
    def __init__(self, p):
        self._p = float(p)
    def p(self, h):
        return torch.full((h.shape[0],), self._p, device=h.device)
    def eval(self):
        return self


class ColPreferHead:
    """p_unsafe that increases with the per-candidate value-head score magnitude, used to
    make the winner deterministic. Here we instead key off a captured column preference by
    returning a fixed vector when called on [R*K, H]; set via .order (list over K)."""
    def __init__(self, K, prefer_col):
        self.K = K
        self.prefer_col = prefer_col
    def p(self, h):
        n = h.shape[0]
        K = self.K
        out = torch.ones(n, device=h.device)          # default unsafe
        # rows are flattened [R*K]; mark the preferred column safe (p=0)
        for i in range(n):
            if i % K == self.prefer_col:
                out[i] = 0.0
        return out
    def eval(self):
        return self


def _runner(llm):
    eng = llm.llm_engine
    me = getattr(eng, "model_executor", None)
    if me is None:
        core = getattr(eng, "engine_core", None)
        me = getattr(getattr(core, "engine_core", None), "model_executor", None)
    dw = getattr(me, "driver_worker", None)
    worker = getattr(dw, "worker", dw)
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError(f"no model_runner (executor={type(me).__name__})")
    return runner


_LLM_KW = dict(enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06)


def _free(llm):
    """Release a vLLM engine's GPU memory before building the next one. The in-process
    EngineCore holds the model until shut down; empty_cache returns the caching
    allocator's blocks to the driver so the next LLM's startup free-memory check passes."""
    import gc
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    from vllm import LLM, SamplingParams

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} model={MODEL}", flush=True)

    # --------------------------------------------------------------- #
    @section("abstention: p=0.0 forces EOS immediately")
    def _():
        llm = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
                  additional_config={"abstain": {"enabled": True, "threshold": 0.5, "strict": True}},
                  enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06)
        r = _runner(llm)
        r.value_head = ConstHead(0.0)
        out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8))
        toks = out[0].outputs[0].token_ids
        print("  eos_token_id:", r.eos_token_id, " emitted:", list(toks))
        assert len(toks) <= 1 and (len(toks) == 0 or toks[0] == r.eos_token_id), \
            f"expected <=1 EOS token, got {list(toks)}"
        _free(llm); del llm

    # --------------------------------------------------------------- #
    @section("abstention: p=1.0 does NOT fire (full length)")
    def _():
        llm = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
                  additional_config={"abstain": {"enabled": True, "threshold": 0.5, "strict": True}},
                  enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06)
        r = _runner(llm)
        r.value_head = ConstHead(1.0)
        out = llm.generate(["Count: one two three"], SamplingParams(max_tokens=8, ignore_eos=True))
        toks = out[0].outputs[0].token_ids
        print("  emitted n:", len(toks))
        assert len(toks) == 8, f"expected 8 tokens, got {len(toks)}"
        _free(llm); del llm

    # --------------------------------------------------------------- #
    # base (no steering) greedy reference for the VFD comparison
    base_tokens = {}
    @section("base model greedy reference")
    def _():
        llm = LLM(model=MODEL, enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06)
        out = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=6, temperature=0.0))
        base_tokens["greedy"] = list(out[0].outputs[0].token_ids)
        print("  base greedy tokens:", base_tokens["greedy"])
        _free(llm); del llm

    # --------------------------------------------------------------- #
    @section("VFD: scratch reserve + candidate forward shape + end-to-end run")
    def _():
        K = 4
        llm = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
                  additional_config={"vfd": {"enabled": True, "threshold": 0.5,
                                             "num_candidates": K, "strict": True}},
                  enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06,
                  max_num_seqs=8)
        r = _runner(llm)
        print("  scratch blocks reserved:", None if r._scratch_blocks is None
              else f"{len(r._scratch_blocks)} ids [{r._scratch_blocks[0]}..{r._scratch_blocks[-1]}]")
        r.value_head = ConstHead(0.0)        # all candidates safe -> first-safe = col 0

        # capture the candidate-forward hidden shape
        shapes = []
        orig = r._candidate_forward
        def wrapped(active):
            h_cand, scratch_idx, plan = orig(active)
            shapes.append(tuple(h_cand.shape))
            return h_cand, scratch_idx, plan
        r._candidate_forward = wrapped

        out = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=6, temperature=1.0, seed=0))
        toks = list(out[0].outputs[0].token_ids)
        print("  candidate_forward shapes [R,K,H]:", shapes)
        print("  VFD committed tokens:", toks)
        hidden = r.model_config.get_hidden_size()
        assert shapes, "candidate_forward never ran (stayed in prefill/fallback?)"
        for s in shapes:
            assert s[1] == K and s[2] == hidden, f"bad candidate shape {s} (K={K}, H={hidden})"
        assert len(toks) >= 1, "VFD produced no tokens"
        _free(llm); del llm

    # --------------------------------------------------------------- #
    @section("VFD: committed token tracks the head-preferred candidate column")
    def _():
        K = 4
        llm = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
                  additional_config={"vfd": {"enabled": True, "threshold": 0.5,
                                             "num_candidates": K, "strict": True}},
                  enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06,
                  max_num_seqs=8)
        r = _runner(llm)

        # Capture, per step, the pending candidate ids and which column the head prefers,
        # then verify the committed token equals that column's candidate.
        prefer = 2
        r.value_head = ColPreferHead(K, prefer)
        seen = {"pending": [], "committed": []}
        orig_sel = r._select
        def wrapped_select(active, h_cand, scratch_idx, plan):
            seen["pending"].append({a: r._pending_tok[a].tolist() for a in active})
            winners = orig_sel(active, h_cand, scratch_idx, plan)
            seen["committed"].append(dict(winners))
            return winners
        r._select = wrapped_select

        out = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=4, temperature=1.0, seed=0))
        toks = list(out[0].outputs[0].token_ids)
        print("  VFD(prefer col %d) committed tokens:" % prefer, toks)
        for step, (pend, comm) in enumerate(zip(seen["pending"], seen["committed"])):
            for a in comm:
                want = pend[a][prefer]
                got = comm[a]
                print(f"    step {step} req {a}: pending={pend[a]} prefer[{prefer}]={want} committed={got}")
                assert got == want, f"winner {got} != preferred-col candidate {want}"
        _free(llm); del llm

    # --------------------------------------------------------------- #
    @section("VFD candidate forward: scratch-path vs real-block reference (isolate)")
    def _():
        # For each step, recompute the candidate's hidden via the REAL prefix blocks (no
        # scratch) using the SAME metadata machinery, and compare to the scratch-path hidden.
        #   scratch != real  -> the scratch copy / attention-over-scratch is the bug.
        #   scratch == real   -> my CommonAttentionMetadata differs from a real decode's.
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.forward_context import set_forward_context
        K = 1
        vfd = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
                  additional_config={"vfd": {"enabled": True, "threshold": 2.0,
                                             "num_candidates": K, "strict": True}},
                  enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06,
                  max_num_seqs=8)
        r = vfd_runner = _runner(vfd)
        diffs = []
        orig = r._candidate_forward
        def wrapped(active):
            h_cand, scratch_idx, plan = orig(active)              # scratch path (already ran)
            R = len(active); bs = r.cache_config.block_size; dev = r.device
            toks = torch.cat([r._pending_tok[a] for a in active])
            pos = [r._next_pos[a] for a in active]
            positions = torch.tensor([p for p in pos for _ in range(K)], device=dev, dtype=torch.long)
            pbt = r.input_batch.block_table[0].get_device_tensor(r.input_batch.num_reqs)
            n = R * K
            ref_bt = torch.zeros((n, pbt.shape[1]), device=dev, dtype=pbt.dtype)
            ref_slot = []
            for i, a in enumerate(active):
                ridx = r.input_batch.req_id_to_index[a]
                p = pos[i]; off = p % bs; tail = p // bs
                for k in range(K):
                    ref_bt[i * K + k] = pbt[ridx]
                    ref_slot.append(int(pbt[ridx, tail]) * bs + off)
            qsl = torch.arange(n + 1, device=dev, dtype=torch.int32)
            sl = torch.tensor([p + 1 for p in pos for _ in range(K)], device=dev, dtype=torch.int32)
            ref_cm = CommonAttentionMetadata(
                query_start_loc=qsl, query_start_loc_cpu=qsl.cpu(), seq_lens=sl,
                num_reqs=n, num_actual_tokens=n, max_query_len=1, max_seq_len=int(max(pos)) + 1,
                block_table_tensor=ref_bt,
                slot_mapping=torch.tensor(ref_slot, device=dev, dtype=torch.long), causal=True)
            ref_md = r._build_per_layer_metadata(ref_cm)
            ref_slotmap = {ln: ref_cm.slot_mapping
                           for g in r.kv_cache_config.kv_cache_groups for ln in g.layer_names}
            with set_forward_context(ref_md, r.vllm_config, num_tokens=n, slot_mapping=ref_slotmap):
                ref_hs = r._model_forward(input_ids=toks, positions=positions)
            if not isinstance(ref_hs, torch.Tensor):
                ref_hs = ref_hs[0] if isinstance(ref_hs, (tuple, list)) else ref_hs.last_hidden_state
            d = float((h_cand.reshape(n, -1) - ref_hs.reshape(n, -1)).norm())
            diffs.append((pos[0], d, float(h_cand.reshape(n, -1).norm())))
            return h_cand, scratch_idx, plan
        r._candidate_forward = wrapped
        vfd.generate(["The quick brown fox jumps over the"],
                     SamplingParams(max_tokens=5, temperature=0.0))
        for p, d, hn in diffs:
            print(f"  pos {p}: ||scratch - real|| = {d:.4e}  (||scratch||={hn:.3e})")
        _free(vfd); del vfd

    # --------------------------------------------------------------- #
    @section("VFD vs base: per-step hidden divergence (localizer)")
    def _():
        # Capture the post-norm hidden that predicts each greedy token, from base and from
        # VFD(K=1), aligned by predicted position, and print the L2 diff. Pinpoints WHERE
        # the prefix first goes wrong: ~0 through step i then a jump at i+1 localizes the
        # bad step (bootstrap vs first steady vs later); a gradual creep means numerics.
        import torch as _t
        prompt = "The quick brown fox jumps over the"
        sp = SamplingParams(max_tokens=6, temperature=0.0)

        base = LLM(model=MODEL, enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06)
        br = _runner(base)
        base_h, base_tok = [], []
        _orig_st = br.sample_tokens
        def _w_st(go):
            st = getattr(br, "execute_model_state", None)
            if st is not None and getattr(st, "sample_hidden_states", None) is not None:
                base_h.append(st.sample_hidden_states[0].float().detach().cpu().clone())
            return _orig_st(go)
        br.sample_tokens = _w_st
        base_tok = list(base.generate([prompt], sp)[0].outputs[0].token_ids)
        _free(base); del base

        vfd = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
                  additional_config={"vfd": {"enabled": True, "threshold": 2.0,
                                             "num_candidates": 1, "strict": True}},
                  enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06,
                  max_num_seqs=8)
        vr = _runner(vfd)
        vfd_h = []
        _orig_seed = vr._seed_candidates
        def _w_seed(hidden_by_req):
            for _r, _h in hidden_by_req.items():
                vfd_h.append(_h.float().detach().cpu().clone())
            return _orig_seed(hidden_by_req)
        vr._seed_candidates = _w_seed
        vfd_tok = list(vfd.generate([prompt], sp)[0].outputs[0].token_ids)
        _free(vfd); del vfd

        print("  base tok:", base_tok)
        print("  vfd  tok:", vfd_tok)
        n = min(len(base_h), len(vfd_h))
        for i in range(n):
            d = float((vfd_h[i] - base_h[i]).norm())
            bn = float(base_h[i].norm())
            print(f"  step {i}: ||vfd_h - base_h|| = {d:.4e}  (||base_h||={bn:.3e})  "
                  f"predicts base={base_tok[i] if i < len(base_tok) else '-'} "
                  f"vfd={vfd_tok[i] if i < len(vfd_tok) else '-'}")

    # --------------------------------------------------------------- #
    @section("VFD greedy == base greedy (prefix correctness, token-for-token)")
    def _():
        # Prefix-correctness arbiter. With temperature=0 every candidate is the argmax and
        # threshold>1 makes all candidates 'safe' (first-safe = col 0 = argmax), so VFD must
        # reproduce base greedy decoding -- a wrong KV write position/prefix would diverge.
        #
        # K=1 is the strict correctness check: one query row = the SAME FlashAttention path
        # base decode uses, so it must match base token-for-token. K=4 additionally exercises
        # the batched candidate forward; any K=4-only divergence is the batched-decode kernel
        # numerics (benign for VFD's stochastic use), not a prefix bug -- reported, not fatal.
        prompt = "The quick brown fox jumps over the"
        sp = SamplingParams(max_tokens=10, temperature=0.0)

        base = LLM(model=MODEL, enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06)
        base_toks = list(base.generate([prompt], sp)[0].outputs[0].token_ids)
        _free(base); del base
        print("  base greedy:", base_toks)

        results = {}
        for K in (1, 4):
            vfd = LLM(model=MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
                      additional_config={"vfd": {"enabled": True, "threshold": 2.0,
                                                 "num_candidates": K, "strict": True}},
                      enforce_eager=True, async_scheduling=False, gpu_memory_utilization=0.06,
                      max_num_seqs=8)
            toks = list(vfd.generate([prompt], sp)[0].outputs[0].token_ids)
            _free(vfd); del vfd
            results[K] = toks
            match = "MATCH" if toks == base_toks else "DIVERGE@%d" % next(
                (i for i, (a, b) in enumerate(zip(base_toks, toks)) if a != b), len(base_toks))
            print(f"  vfd K={K} greedy: {toks}  -> {match}")

        # K=1 must match base exactly (single-row forward == base decode path).
        assert results[1] == base_toks, (
            f"VFD(K=1)-greedy diverged from base greedy -- real prefix bug:\n"
            f"    base ={base_toks}\n    vfdK1={results[1]}"
        )


if __name__ == "__main__":
    main()
    print(f"\n{'='*72}")
    if _failures:
        print("DIAG FAILURES:", _failures)
        sys.exit(1)
    print("DIAG ALL OK")
    sys.exit(0)
