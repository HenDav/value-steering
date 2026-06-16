# SPDX-License-Identifier: Apache-2.0
"""
GPU behavioral tests (written, NOT run here -- no accelerator in the build env).

These assert the features actually FIRE end-to-end, which the static contract checks
and CPU unit tests cannot: the runner hooks swallow exceptions in production, so "it
ran without error" is not "it worked." Each test forces a deterministic value head and
checks the *observable decode behavior*.

Gating: every test is skipped unless a CUDA device is present AND a small model is
named via $VALUE_STEER_TEST_MODEL (e.g. "facebook/opt-125m"). Run on a GPU box:

    VALUE_STEER_TEST_MODEL=facebook/opt-125m pytest tests/test_gpu_behavioral.py -q -m gpu

They double as the behavioral pass the compatibility agent records (versions.record_
validation(..., behavioral=True)) once green on a target vLLM version.
"""

import os

# Both runners are accessed in-process (the test forces a constant head on the live
# runner instance). The V1 multiprocess engine core would put the runner in another
# process (unreachable) AND, with CUDA already initialized in this process, vLLM's
# forked EngineCore dies ("Cannot re-initialize CUDA in forked subprocess"). Run the
# engine in-process so llm_engine.model_executor.driver_worker.worker.model_runner is
# the real runner. Must be set before vLLM is imported.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import pytest

pytestmark = pytest.mark.gpu

_MODEL = os.environ.get("VALUE_STEER_TEST_MODEL")


def _get_runner(llm):
    """Locate the value-steering model runner on an in-process V1 engine. The exact
    attribute path shifts across vLLM builds, so walk it defensively."""
    eng = llm.llm_engine
    me = getattr(eng, "model_executor", None)
    if me is None:
        core = getattr(eng, "engine_core", None)
        me = getattr(getattr(core, "engine_core", None), "model_executor", None)
    dw = getattr(me, "driver_worker", None)
    worker = getattr(dw, "worker", dw)
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError(
            f"could not locate model_runner (engine={type(eng).__name__}, "
            f"executor={type(me).__name__}, driver={type(dw).__name__})"
        )
    return runner


def _have_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


requires_gpu_model = pytest.mark.skipif(
    not (_have_gpu() and _MODEL),
    reason="needs CUDA + $VALUE_STEER_TEST_MODEL (small model)",
)


def _force_const_head(runner, p_value: float):
    """Replace the runner's value head with a constant p, so the decision is
    deterministic regardless of the (random) head weights."""
    import torch

    class _Const:
        def p(self, h):
            return torch.full((h.shape[0],), float(p_value), device=h.device)

        def eval(self):
            return self

    runner.value_head = _Const()


# --------------------------------------------------------------------------- #
# Abstention                                                                  #
# --------------------------------------------------------------------------- #
@requires_gpu_model
def test_abstention_forces_eos_when_value_low():
    """p=0.0 < c -> every decode step must emit EOS immediately (sequence stops at
    the first generated token)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"abstain": {"enabled": True, "threshold": 0.5, "strict": True}},
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.06,          # logits rows must be 1:1 with requests (no pipelining)
    )
    runner = _get_runner(llm)
    _force_const_head(runner, 0.0)                          # always below threshold

    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8))
    toks = out[0].outputs[0].token_ids
    eos = runner.eos_token_id
    # forced EOS -> at most one (EOS) token emitted
    assert len(toks) <= 1 and (len(toks) == 0 or toks[0] == eos)


@requires_gpu_model
def test_abstention_does_not_fire_when_value_high():
    """p=1.0 > c -> abstention must NOT fire; generation proceeds normally."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"abstain": {"enabled": True, "threshold": 0.5, "strict": True}},
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.06,
    )
    runner = _get_runner(llm)
    _force_const_head(runner, 1.0)

    out = llm.generate(["Count: one two three"], SamplingParams(max_tokens=8, ignore_eos=True))
    assert len(out[0].outputs[0].token_ids) == 8           # no early forced stop


# --------------------------------------------------------------------------- #
# VFD                                                                         #
# --------------------------------------------------------------------------- #
def _free_engine(llm):
    """Release a vLLM engine's GPU memory between in-process constructions (the V1
    EngineCore holds the model until shut down; empty_cache returns the blocks)."""
    import gc

    import torch
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


@requires_gpu_model
def test_vfd_runs_and_shifts_distribution():
    """VFD fires and the value head DRIVES the committed token: with a head that prefers a
    fixed non-zero candidate column, every committed token equals that column's sampled
    candidate (so the output is shifted away from the base argmax by the value filter).
    Exercises candidate-forward -> score -> first-safe select -> commit -> reseed."""
    import torch
    from vllm import LLM, SamplingParams

    K, prefer = 4, 2
    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 0.5, "num_candidates": K,
                                   "strict": True}},
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.06,
        max_num_seqs=8,
    )
    runner = _get_runner(llm)

    class _Prefer:                       # p=0 (safe) only for column `prefer` -> first-safe picks it
        def p(self, h):
            n = h.shape[0]
            out = torch.ones(n, device=h.device)
            out[torch.arange(n, device=h.device) % K == prefer] = 0.0
            return out

        def eval(self):
            return self

    runner.value_head = _Prefer()
    seen = {"pend": [], "comm": []}
    _orig = runner._select

    def _wsel(active, h_cand, scratch_idx, plan):
        seen["pend"].append({a: runner._pending_tok[a].tolist() for a in active})
        winners = _orig(active, h_cand, scratch_idx, plan)
        seen["comm"].append(dict(winners))
        return winners

    runner._select = _wsel
    llm.generate(["Hello, my name is"], SamplingParams(max_tokens=4, temperature=1.0, seed=0))
    assert seen["comm"], "VFD never committed a token (stuck in prefill/fallback)"
    for pend, comm in zip(seen["pend"], seen["comm"]):
        for a in comm:
            assert comm[a] == pend[a][prefer], (
                f"committed {comm[a]} != preferred-column candidate {pend[a][prefer]}"
            )


@requires_gpu_model
def test_vfd_candidate_forward_shapes():
    """_candidate_forward returns [R, K, H], AND VFD forced greedy reproduces base greedy
    token-for-token -- i.e. the single-forward KV prefix (scratch tail-copy + in-cache
    commit, no extra model forward) is maintained correctly across steps."""
    from vllm import LLM, SamplingParams

    K = 4
    prompt = "The quick brown fox jumps over the"
    sp = SamplingParams(max_tokens=8, temperature=0.0)

    base = LLM(model=_MODEL, enforce_eager=True, async_scheduling=False,
               gpu_memory_utilization=0.06)
    base_toks = list(base.generate([prompt], sp)[0].outputs[0].token_ids)
    _free_engine(base)
    del base

    # threshold>1 -> every candidate 'safe' -> first-safe = col 0; temperature 0 -> all
    # candidates are the argmax, so VFD commits the greedy token each step.
    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": K,
                                   "strict": True}},
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.06,
        max_num_seqs=8,
    )
    runner = _get_runner(llm)
    shapes = []
    _orig = runner._candidate_forward

    def _wrapped(active):
        h_cand, scratch_idx, plan = _orig(active)
        shapes.append(tuple(h_cand.shape))
        return h_cand, scratch_idx, plan

    runner._candidate_forward = _wrapped
    vfd_toks = list(llm.generate([prompt], sp)[0].outputs[0].token_ids)

    hidden = runner.model_config.get_hidden_size()
    assert shapes, "candidate forward never ran"
    for s in shapes:
        assert s[1] == K and s[2] == hidden, f"bad candidate shape {s} (K={K}, H={hidden})"
    assert vfd_toks == base_toks, (
        f"VFD greedy diverged from base greedy (prefix not maintained):\n"
        f"    base={base_toks}\n    vfd ={vfd_toks}"
    )


# --------------------------------------------------------------------------- #
# Batched / mixed / edge-case coverage (the formerly GPU-VALIDATE paths)      #
# --------------------------------------------------------------------------- #
@requires_gpu_model
def test_vfd_batched_and_mixed_matches_base():
    """R>1 concurrent requests of DIFFERENT lengths -> staggered prefill completion forces
    mixed prefill+decode steps (VFD falls back to base for those), then steady multi-request
    decode. Forced greedy, each request's VFD output must equal ITS base greedy output --
    validating per-request positions/scratch, the _seed_from_base row<->request mapping for
    R>1, the mixed-step fallback, and that the scratch reserve is sized K*max_num_seqs."""
    from vllm import LLM, SamplingParams

    K = 4
    prompts = ["The capital of France is",
               "Once upon a time, in a small village near the",
               "2 + 2 ="]
    sp = SamplingParams(max_tokens=8, temperature=0.0)

    base = LLM(model=_MODEL, enforce_eager=True, async_scheduling=False,
               gpu_memory_utilization=0.06)
    base_out = [list(o.outputs[0].token_ids) for o in base.generate(prompts, sp)]
    _free_engine(base)
    del base

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": K,
                                   "strict": True}},
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.06,
        max_num_seqs=8,
    )
    runner = _get_runner(llm)
    # batched/eager, no single_stream and no scratch_max_seqs override -> scratch_seqs == max_num_seqs
    assert runner._scratch_seqs == runner._max_num_seqs
    assert len(runner._scratch_blocks) == K * runner._scratch_seqs   # scratch reserve sizing
    vfd_out = [list(o.outputs[0].token_ids) for o in llm.generate(prompts, sp)]

    for i, (b, v) in enumerate(zip(base_out, vfd_out)):
        assert v == b, f"request {i} VFD greedy != base greedy:\n base={b}\n vfd ={v}"


@requires_gpu_model
def test_vfd_single_stream_scratch_reserve_is_K():
    """single_stream asserts peak concurrency 1, so the scratch reserve must shrink to K*1
    (NOT K*max_num_seqs). This is the memory optimization that makes high K affordable on the
    compile path (which forces max_num_seqs>=K): K=8,mns=16 drops from 128 blocks to 8."""
    from vllm import LLM, SamplingParams

    K = 8
    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": K,
                                   "strict": True, "single_stream": True}},
        async_scheduling=False, gpu_memory_utilization=0.06, max_num_seqs=16,  # mns>=K
    )
    runner = _get_runner(llm)
    assert runner._scratch_seqs == 1, f"single_stream should cap scratch_seqs to 1, got {runner._scratch_seqs}"
    assert len(runner._scratch_blocks) == K, (
        f"single_stream scratch reserve should be K={K}, got {len(runner._scratch_blocks)} "
        f"(max_num_seqs={runner._max_num_seqs}; old sizing would be {K * runner._max_num_seqs})"
    )
    # and it must still decode correctly with the minimal reserve (one request -> R=1)
    out = llm.generate(["The quick brown fox jumps over the"],
                       SamplingParams(max_tokens=8, temperature=0.0))
    assert len(list(out[0].outputs[0].token_ids)) == 8


@requires_gpu_model
def test_abstention_forces_eos_batched_mixed():
    """Abstention with R>1 requests of different lengths: even on mixed prefill+decode
    steps the logits rows stay 1:1 with requests (gpu_model_runner non-spec invariant), so
    p=0 forces EOS for EVERY request immediately."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"abstain": {"enabled": True, "threshold": 0.5, "strict": True}},
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.06,
    )
    runner = _get_runner(llm)
    _force_const_head(runner, 0.0)
    eos = runner.eos_token_id
    prompts = ["The capital of France is",
               "Once upon a time, in a small village near the",
               "2 + 2 ="]
    outs = llm.generate(prompts, SamplingParams(max_tokens=8))
    for i, o in enumerate(outs):
        toks = o.outputs[0].token_ids
        assert len(toks) <= 1 and (len(toks) == 0 or toks[0] == eos), \
            f"request {i} not gated to EOS: {list(toks)}"


@requires_gpu_model
def test_abstention_skips_ignore_eos_rows():
    """ignore_eos rows must be SKIPPED by abstention even when p<c (forcing EOS there would
    make an all -inf row -> NaN once the EOS mask is applied). p=0 + ignore_eos -> full
    length, not an early stop."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"abstain": {"enabled": True, "threshold": 0.5, "strict": True}},
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.06,
    )
    runner = _get_runner(llm)
    _force_const_head(runner, 0.0)                              # below threshold -> would gate
    out = llm.generate(["Count: one two three"],
                       SamplingParams(max_tokens=8, ignore_eos=True))
    assert len(out[0].outputs[0].token_ids) == 8               # ignore_eos row skipped


@requires_gpu_model
def test_vfd_rejects_logprobs():
    """VFD's single-forward output has no logprobs; a VFD-path request asking for them must
    raise (NotImplementedError propagates past execute_model's swallow with strict=True),
    not silently return None."""
    import pytest as _pytest
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": 4,
                                   "strict": True}},
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.06,
        max_num_seqs=8,
    )
    with _pytest.raises(Exception) as ei:
        llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0, logprobs=2))
    assert "logprob" in str(ei.value).lower()


# --------------------------------------------------------------------------- #
# Compile / CUDA-graph path (PIECEWISE)                                        #
# --------------------------------------------------------------------------- #
@requires_gpu_model
def test_vfd_compiled_matches_base_and_replays():
    """SINGLE-STREAM compile opt-in: with cudagraphs ON (enforce_eager=False) AND the explicit
    `single_stream=True` flag at max_num_seqs>=K, VFD's candidate backbone forward must route
    through captured PIECEWISE graphs AND still reproduce base greedy token-for-token.

    Asserts BOTH halves of "the feature FIRES, not just runs":
      * correctness: VFD greedy == base greedy (the single-forward KV prefix is maintained
        under graph replay -- the padded buffer copy and pad-row sink are correct);
      * the graph actually replayed: runner._vfd_replay_fired > 0 (else we'd be silently
        eager and the test would prove nothing about the compile path).

    WHY max_num_seqs=8 (not 1): the K-candidate forward is a uniform-decode batch of n=R*K
    rows, and vLLM only CAPTURES a uniform-decode graph at N rows when max_num_seqs>=N
    (gpu_model_runner._dummy_run: num_reqs=min(max_num_seqs,N) with 1 tok/req). At max_num_seqs=1
    the K=3 graph can't capture, so the compile path correctly falls back to eager (covered by
    test_vfd_compile_single_stream_capacity1_falls_back_to_eager). The speedup needs
    max_num_seqs>=K, which is also the regime where >1 CONCURRENT request would corrupt -- hence
    the single_stream opt-in (this test drives ONE request, so R=1 actual; correctness holds).
    Base also runs with cudagraphs so the token reference matches the compiled regime."""
    from vllm import LLM, SamplingParams

    K = 3                                    # n = 1*3 = 3 -> padded up to a captured size (4)
    prompt = "The quick brown fox jumps over the"
    sp = SamplingParams(max_tokens=8, temperature=0.0)

    base = LLM(model=_MODEL, async_scheduling=False, gpu_memory_utilization=0.06)
    base_toks = list(base.generate([prompt], sp)[0].outputs[0].token_ids)
    _free_engine(base)
    del base

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": K,
                                   "strict": True, "single_stream": True}},
        async_scheduling=False,             # cudagraphs ON (no enforce_eager)
        gpu_memory_utilization=0.06,
        max_num_seqs=8,                     # >= K so the K-candidate uniform-decode graph captures
    )
    runner = _get_runner(llm)
    assert runner._vfd_compile_ok, "model uses mrope/xdrope; compile path falls back to eager"
    vfd_toks = list(llm.generate([prompt], sp)[0].outputs[0].token_ids)   # ONE request -> R=1

    assert runner._vfd_replay_fired > 0, (
        "VFD never replayed a cudagraph (ran eager) -- compile path did not engage; "
        f"replay_fired={runner._vfd_replay_fired}"
    )
    assert vfd_toks == base_toks, (
        f"VFD greedy diverged from base under cudagraphs (padded-buffer path wrong):\n"
        f"    base={base_toks}\n    vfd ={vfd_toks}"
    )


@requires_gpu_model
def test_vfd_compile_single_stream_capacity1_falls_back_to_eager():
    """At max_num_seqs=1 with cudagraphs ON, the K-candidate uniform-decode graph CANNOT be
    captured (vLLM needs max_num_seqs>=N to capture an N-row decode graph; the ceiling is
    min(max_num_seqs*2,512)=2 < K). VFD must then SILENTLY and CORRECTLY fall back to eager:
    greedy still == base, and no graph replays (replay_fired==0). This documents that the
    'safe but unaccelerated' fallback is correct -- not a crash, not a mis-decode."""
    from vllm import LLM, SamplingParams

    K = 3
    prompt = "The quick brown fox jumps over the"
    sp = SamplingParams(max_tokens=8, temperature=0.0)

    base = LLM(model=_MODEL, async_scheduling=False, gpu_memory_utilization=0.06)
    base_toks = list(base.generate([prompt], sp)[0].outputs[0].token_ids)
    _free_engine(base)
    del base

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": K,
                                   "strict": True}},
        async_scheduling=False,             # cudagraphs ON
        gpu_memory_utilization=0.06,
        max_num_seqs=1,                     # ceiling 2 < K -> candidate graph can't capture
    )
    runner = _get_runner(llm)
    vfd_toks = list(llm.generate([prompt], sp)[0].outputs[0].token_ids)

    assert runner._vfd_replay_fired == 0, (
        "expected eager fallback at max_num_seqs=1 (K-row graph uncapturable), but a graph "
        f"replayed; replay_fired={runner._vfd_replay_fired}"
    )
    assert vfd_toks == base_toks, (
        f"VFD greedy diverged from base on the eager-fallback path:\n"
        f"    base={base_toks}\n    vfd ={vfd_toks}"
    )


@requires_gpu_model
def test_vfd_compile_batched_guard_raises():
    """V1 SAFETY CONTRACT: the compile/cudagraph path is single-stream only. Configuring it with
    batching (enforce_eager=False AND max_num_seqs>1) must FAIL FAST in __init__ -- not silently
    corrupt requests after the first. Batched serving must use enforce_eager=True."""
    import pytest as _pytest
    from vllm import LLM
    with _pytest.raises(Exception) as ei:
        LLM(model=_MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
            additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": 4,
                                       "strict": True}},
            async_scheduling=False, gpu_memory_utilization=0.06, max_num_seqs=2)  # compile + batch
    assert "single-request" in str(ei.value).lower() or "enforce_eager" in str(ei.value).lower()


@requires_gpu_model
def test_vfd_kv_ops_captured_match_base():
    """The per-layer KV surgery (scratch copy + winner commit) is replayed from CAPTURED
    CUDA graphs (capture_kv_ops, on by default). Asserts the capture FIRES
    (runner._kv_replayed > 0) AND that greedy still equals base token-for-token -- i.e. the
    captured in-place KV scatter (replayed against persistent index buffers) writes exactly
    the same KV as the eager loop. A wrong index buffer / stale capture would diverge here."""
    from vllm import LLM, SamplingParams

    K = 4
    prompt = "The quick brown fox jumps over the"
    sp = SamplingParams(max_tokens=8, temperature=0.0)

    base = LLM(model=_MODEL, enforce_eager=True, async_scheduling=False,
               gpu_memory_utilization=0.06)
    base_toks = list(base.generate([prompt], sp)[0].outputs[0].token_ids)
    _free_engine(base)
    del base

    llm = LLM(
        model=_MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 2.0, "num_candidates": K,
                                   "strict": True, "capture_kv_ops": True}},
        enforce_eager=True,            # capture of the KV ops is independent of the backbone
        async_scheduling=False,
        gpu_memory_utilization=0.06,
        max_num_seqs=8,
    )
    runner = _get_runner(llm)
    vfd_toks = list(llm.generate([prompt], sp)[0].outputs[0].token_ids)

    assert runner._capture_kv, "KV-op capture disabled itself (capture failed) -- see log"
    assert runner._kv_replayed > 0, (
        f"KV-op cudagraph never replayed (ran eager); replayed={runner._kv_replayed}"
    )
    assert vfd_toks == base_toks, (
        f"VFD greedy diverged from base with captured KV ops (bad capture/index buffer):\n"
        f"    base={base_toks}\n    vfd ={vfd_toks}"
    )


def _hidden_closeness_over_long_run(n_tokens=48):
    """Drive a long (multi-block) K=1 greedy generation and, at every step, compare VFD's
    scratch-path candidate hidden against a REAL-prefix-block reference forward (base-equivalent
    attention: full real block table, new token written to the real tail slot). Returns the list
    of per-step (position, cosine_similarity, relative_L2) plus whether the context crossed a KV
    block. The reference IS what base decode computes, so cosine~1 proves VFD's reconstructed
    attention equals base's up to float -- the correct long-context gate (token identity isn't,
    because near-ties flip). Greedy K=1: the reference token == the committed token, so the
    reference's write to the real tail slot is idempotent with the commit (no corruption)."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata

    cfg = {"enabled": True, "threshold": 2.0, "num_candidates": 1, "strict": True}
    llm = LLM(model=_MODEL, worker_cls="value_steer.worker.ValueSteerWorker",
              additional_config={"vfd": cfg}, enforce_eager=True, async_scheduling=False,
              gpu_memory_utilization=0.06, max_num_seqs=8)
    r = _get_runner(llm)
    bs = r.cache_config.block_size
    stats = []
    orig = r._candidate_forward

    def wrapped(active):
        ret = orig(active)                          # scratch path: (h_cand, scratch_idx, plan)
        h_cand = ret[0]
        dev = r.device
        toks = torch.cat([r._pending_tok[a] for a in active])
        pos = [r._next_pos[a] for a in active]
        positions = torch.tensor(pos, device=dev, dtype=torch.long)
        pbt = r.input_batch.block_table[0].get_device_tensor(r.input_batch.num_reqs)
        n = len(active)
        ref_bt = torch.zeros((n, pbt.shape[1]), device=dev, dtype=pbt.dtype)
        ref_slot = []
        for i, a in enumerate(active):
            ridx = r.input_batch.req_id_to_index[a]
            p = pos[i]
            ref_bt[i] = pbt[ridx]
            ref_slot.append(int(pbt[ridx, p // bs]) * bs + (p % bs))
        qsl = torch.arange(n + 1, device=dev, dtype=torch.int32)
        sl = torch.tensor([p + 1 for p in pos], device=dev, dtype=torch.int32)
        ref_cm = CommonAttentionMetadata(
            query_start_loc=qsl, query_start_loc_cpu=qsl.cpu(), seq_lens=sl, num_reqs=n,
            num_actual_tokens=n, max_query_len=1, max_seq_len=int(max(pos)) + 1,
            block_table_tensor=ref_bt,
            slot_mapping=torch.tensor(ref_slot, device=dev, dtype=torch.long), causal=True)
        ref_md = r._build_per_layer_metadata(ref_cm)            # cpl=0 -> normal decode (base path)
        ref_sm = {ln: ref_cm.slot_mapping
                  for g in r.kv_cache_config.kv_cache_groups for ln in g.layer_names}
        with set_forward_context(ref_md, r.vllm_config, num_tokens=n, slot_mapping=ref_sm):
            ref = r._model_forward(input_ids=toks, positions=positions)
        if not isinstance(ref, torch.Tensor):
            ref = ref[0] if isinstance(ref, (tuple, list)) else ref.last_hidden_state
        a_ = h_cand.reshape(n, -1)[0].float()
        b_ = ref.reshape(n, -1)[0].float()
        cos = float(torch.nn.functional.cosine_similarity(a_, b_, dim=0))
        rel = float((a_ - b_).norm() / (b_.norm() + 1e-9))
        stats.append((pos[0], cos, rel))
        return ret

    r._candidate_forward = wrapped
    llm.generate(["Count slowly and then describe a calm morning by the sea in detail:"],
                 SamplingParams(max_tokens=n_tokens, temperature=0.0, ignore_eos=True))
    _free_engine(llm)
    del llm
    crossed = any(p >= bs for p, _, _ in stats)
    return stats, crossed


@requires_gpu_model
def test_vfd_hidden_matches_real_prefix_long_context():
    """The CORRECT long-context gate (token identity isn't -- greedy flips near-ties): VFD's
    reconstructed attention (scratch tail-block + hand-built metadata, multi-block prefix) must
    equal the real-prefix-block decode (what base computes) up to FLOAT. Asserts that across a
    long multi-block run, every step's scratch-path hidden has cosine ~1 with the real-block
    reference -- proving the long-context divergence is benign float tie-breaking, not a KV bug."""
    stats, crossed = _hidden_closeness_over_long_run()
    assert stats, "candidate forward never ran"
    assert crossed, "context never crossed a KV block -- did not exercise the multi-block path"
    worst = min(c for _, c, _ in stats)
    worst_rel = max(rel for _, _, rel in stats)
    print(f"[hidden-closeness] steps={len(stats)} min_cos={worst:.6f} max_rel_L2={worst_rel:.4e}")
    assert worst > 0.999, f"scratch-path hidden diverges from real-block decode (min cos {worst})"
