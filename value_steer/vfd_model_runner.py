# SPDX-License-Identifier: Apache-2.0
"""
Value-Filtered Decoding for vLLM 0.19.1 (Paper 2), SCALAR head, no phase 1.

Method (matches llm_safety/llm_utils.py phase-2, vocab head intentionally omitted)
----------------------------------------------------------------------------------
The scalar head scores a token only AFTER it is consumed -- it needs h_t^{(k)}, the
hidden state produced by candidate k -- so candidates must be forwarded (unlike the
vocab head, which scores all next tokens from h_{t-1} in one projection but, per the
user, performs worse). Per decode step, for each request:

  1. sample K candidates from the WARPED LM distribution (temperature/top-p), NOT
     top-K argmax;                                   [matches dense_sample_chunk]
  2. forward the K candidates -> h_t^{(1..K)}        [THE one transformer pass]
  3. p_unsafe[k] = sigmoid(V̂(h_t^{(k)}))            [scalar head, fp32]
  4. safe = p_unsafe < threshold
       if any safe: commit the FIRST safe candidate  [keeps base dist among safe
                                                       tokens = minimal intervention]
       else:        commit argmin(p_unsafe)          [safest], or, if args_fallback,
                    argmin(p_unsafe + prob_weight * (-exp(logp)))  [ARGS objective]
  5. carry the winner's hidden state to seed the next step's candidates.

NO PHASE 1: the paper's optimistic "sample 1, keep if safe, else rollback" is dropped.
That removes KV-cache rollback entirely (no speculative commit, no undo) -- the single
thing that is genuinely painful on vLLM's paged cache. We always run the K-candidate
step; "first safe" recovers the same selective behavior phase 1 provided.

Single forward per token
-------------------------
The K-candidate forward IS the decode forward; there is no separate base forward, and
-- per an explicit project constraint -- the winner is committed WITHOUT a second model
pass. Candidates for step t are sampled at the END of step t-1 from compute_logits(winner
hidden) -- compute_logits is the LM head only (llama.py:582 / opt.py:412), one matmul,
not a pass. So exactly one transformer forward per emitted token, K query rows instead of
1 (nearly free in memory-bound decode).

Paged-KV mechanics (grounded against flash_attn.py @ v0.19.1)
-------------------------------------------------------------
FlashAttention writes a token's K/V to the flat slot in `slot_mapping`
(reshape_and_cache_flash, _custom_ops.py:2508) and, on read, locates position j of a
sequence at block_table[j // block_size], offset j % block_size (flash_attn_varlen_func,
seqused_k = seq_lens). So a candidate's new token at position p MUST live in the physical
block that serves block index p // block_size, at offset p % block_size -- the same block
that holds the request's real prefix tail for that block index. We therefore give each
candidate its OWN scratch block that is a COPY of the request's tail block (real positions
[tail_start, p-1]) and let the forward write the candidate's new token at offset
p % block_size. Attention then reads: full real prefix blocks [0, tail_idx) directly from
the request's real blocks, plus the partial tail block [tail_idx] from the candidate's
private scratch copy. K candidates of one request never collide (distinct scratch blocks),
and the real cache is never touched by the candidate forward.

Committing the winner is then a pure in-cache copy: the winner's new-token K/V already sits
in its scratch block at offset p % block_size; copy that one slot into the request's REAL
tail block at the same offset, per layer. No second forward. (See _commit_winner_kv.)

Feature contract (from DenseValueModel in the paper repo)
---------------------------------------------------------
  * feature = model.model(...).last_hidden_state -- final layer, POST-final-RMSNorm,
    per-token, the SAME tensor lm_head consumes. No pooling.
  * the head runs in fp32 with the hidden state cast to fp32 (backbone is bf16).
  * scalar head architecture (must match exactly to load the checkpoint):
      Linear(H,H) -> Tanh -> Linear(H,H) -> ReLU -> Linear(H,1)  (fp32, raw logit)
    p_unsafe = sigmoid(logit).

Scratch KV reservation (the worker-init integration seam)
---------------------------------------------------------
The worker cannot draw blocks from the scheduler's pool, so VFD GROWS the worker-side KV
tensors at init by one block per concurrent candidate (K * scratch_seqs, where scratch_seqs
is the assumed peak concurrency: 1 under single_stream, else max_num_seqs or an explicit
`scratch_max_seqs` cap) and hands those top block ids out as scratch (initialize_kv_cache
override below). The scheduler's pool
only ever spans [0, base_num_blocks), so it never allocates a scratch block -> the scratch
range collides with no live request. (Extra blocks come from the gpu_memory_utilization
headroom; lower K or max_num_seqs under a tight budget.)

GPU path status: VALIDATED on NVIDIA A100 (vLLM 0.19.1, torch 2.10.0+cu128, opt-125m,
FlashAttention v2). The single-forward path is correct end to end: scratch reserve,
candidate forward shape [R,K,H], value-filtered selection, and the in-cache winner commit
(no second model forward). Proven token-for-token AT SHORT CONTEXT: with a never-intervening
head + greedy sampling VFD reproduces base greedy decoding exactly (K=1 and K=4) -- so the KV
prefix (scratch tail-block copy + slot/position math + commit) is maintained correctly across
steps. CAVEAT (measured, not a bug): greedy==base is bit-exact only at short context. VFD's
hand-built CommonAttentionMetadata drives a different-but-valid FlashAttention kernel path than
base's native decode, so at long context (once the prefix crosses a KV block) greedy flips
near-ties -- e.g. opt-125m first flip at ~30 tokens picked base's runner-up at a 0.094-logprob
gap. This holds for K=1 and K=4 alike (not batched-attn noise); it is the same class as
switching attention backend / batch size / hardware. The KV mechanics stay correct (the
candidate hidden is bit-exact to a real-prefix-block decode, cos=1.0 over a 48-token multi-block
run); only greedy tie-breaking differs. A valid long-context gate is
hidden-state closeness, not token identity. Batched (R>1) and mixed prefill+decode steps are also validated
(tests/test_gpu_behavioral.py). Unsupported configs fail fast at the right point rather
than mis-decoding: spec decode, async scheduling, and multi-KV-cache-group models raise in
__init__/initialize_kv_cache; a VFD-path request asking for logprobs/prompt_logprobs raises
NotImplementedError. The PIECEWISE cudagraph path added on top of this is also validated on
A100 (see below). No silent GPU-VALIDATE gaps remain in the VFD path.

CUDA-graph / torch.compile (PIECEWISE) -- VALIDATED on A100
-----------------------------------------------------------
The candidate forward is a uniform-decode batch of n=R*K rows -- structurally a normal
decode batch vLLM already captures PIECEWISE graphs for. _candidate_forward dispatches into
those (via _vfd_cudagraph_dispatch): it pads n up to a captured size, writes the candidate
tokens/positions INTO the persistent self.input_ids/self.positions buffers the graph replays
against (cuda_graph.py does NOT copy inputs on replay -- it reads the captured addresses),
passes cudagraph_runtime_mode + batch_descriptor to set_forward_context, and slices the n
real rows back out. PIECEWISE keeps attention eager, so our per-layer slot_mapping still
drives the KV write; FULL is excluded at dispatch (it would capture attention with static
metadata, incompatible with VFD's per-step scratch tables) and any no-graph case (cudagraphs
off, enforce_eager, n too large, mrope/xdrope) falls back to the eager path -- always
correct, just unaccelerated. This removes the need for enforce_eager AT BATCH SIZE 1.
VALIDATED on A100 (vLLM 0.19.1, opt-125m): with cudagraphs ON, VFD greedy
reproduces base greedy token-for-token AND the graph actually replayed (_vfd_replay_fired>0)
-- including the padded-batch case (n straddling a capture size). See
tests/test_gpu_behavioral.py::test_vfd_compiled_matches_base_and_replays.

*** V1 SAFETY CONTRACT (single-stream-only compile path; eager is the serving default) ***
The compile/cudagraph speedup is correct AND reachable only for SINGLE-REQUEST decode, and the
two reasons pull opposite ways:
  (a) CORRECTNESS: at >1 CONCURRENT request (R>1) the candidate forward batches R*K rows -- a
      shape outside what vLLM compiled for plain decode -- and the COMPILED model corrupts every
      request after the first (request 0 stays correct; reproduced R=4 captured / R=8 fallback;
      see memory vfd-safety-eval). EAGER (enforce_eager=True) is correct for ALL R.
  (b) REPLAY/CAPTURE: the K-candidate forward is a uniform-decode batch of n=R*K rows; vLLM only
      captures a uniform-decode graph at N rows when max_num_seqs>=N (gpu_model_runner._dummy_run
      builds num_reqs=min(max_num_seqs,N) at 1 tok/req and asserts sum==N), and
      max_cudagraph_capture_size defaults to min(max_num_seqs*2,512). So at max_num_seqs=1 the
      candidate graph CANNOT capture (n=K>2) and the compile path falls back to eager -- the
      speedup is UNREACHABLE there (correct, just unaccelerated).
Replay needs max_num_seqs>=K; correctness needs R==1 actual concurrency -- and the engine can't
guarantee R==1 while max_num_seqs>=K. So the speedup is a SINGLE-REQUEST-LATENCY opt-in: set the
vfd flag single_stream=True AND max_num_seqs>=K AND drive exactly ONE request at a time (offline /
single-user / benchmarking). NOT safe for concurrent serving. Without single_stream, __init__
FAILS FAST when enforce_eager=False AND max_num_seqs>1. The DEFAULT supported path is EAGER
(correct for all R; the safety result runs here). The proper safe-by-construction fix (compile at
max_num_seqs=1) is to recast the candidate forward as ONE request with query_len=K via vLLM's
spec-decode tree machinery (depth-1 K-leaf tree) -- future work.

NOTE (load-bearing): the KV-write op reads the new-token slot from
forward_context.slot_mapping[layer_name], NOT from attn_metadata -- so _candidate_forward
MUST pass slot_mapping= to set_forward_context (see there), else candidate KV lands in a
stale slot and the hidden silently diverges from base.

Config: vllm_config.additional_config["vfd"]; per-request threshold via extra_args.
Spec decode must be OFF.
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from .scratch_alloc import ScratchAllocator, candidate_block_layout
from .steering_ops import select_vfd, warp_logits

# Shared scalar head + feature contract (same module abstention uses).
from .value_probe import ValueHead, load_value_head, request_threshold


class VFDModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        cfg = (vllm_config.additional_config or {}).get("vfd", {})
        self.vfd_enabled: bool = bool(cfg.get("enabled", True))
        self.K: int = int(cfg.get("num_candidates", 8))
        # Threshold: explicit cfg > the calibrated value in the checkpoint's .meta.json
        # sidecar (so a calibrated head drops in without re-passing the number) > 0.5.
        from .train_probe import resolve_threshold
        self.default_thr: float = resolve_threshold(cfg, default=0.5)
        self.args_fallback: bool = bool(cfg.get("args_fallback", False))
        self.prob_weight: float = float(cfg.get("prob_weight", 1.0))
        self.strict: bool = bool(cfg.get("strict", False))   # CI: re-raise, don't swallow
        # Opt-in: allow the compile/cudagraph speedup with max_num_seqs>1. The candidate
        # graph only CAPTURES when max_num_seqs >= the (padded) K-row batch (vLLM's
        # uniform-decode capture needs one request slot per candidate row -- see the guard
        # below), so the speedup is unreachable at max_num_seqs=1. But max_num_seqs>1 is also
        # the regime where >1 CONCURRENT request corrupts (R>1; see the docstring). Setting
        # this flag asserts "I will only ever drive ONE request at a time" (single-request
        # latency benchmarking); the engine cannot enforce that, so it stays opt-in and is
        # NOT safe for concurrent serving. Default serving path is eager (correct for all R).
        self.single_stream: bool = bool(cfg.get("single_stream", False))

        if vllm_config.speculative_config is not None and self.vfd_enabled:
            raise ValueError("VFD owns the decode forward; run with spec decode OFF.")
        # VFD's single-forward output path is synchronous (execute_model returns the
        # committed token directly); it has no async/pipelined branch. Fail fast rather
        # than silently mis-decode under async scheduling.
        if self.vfd_enabled and getattr(vllm_config.scheduler_config, "async_scheduling", False):
            raise ValueError(
                "VFD requires synchronous scheduling; launch with async_scheduling=False "
                "(its single-forward decode path has no async/pipelined output branch)."
            )
        # V1 SAFETY CONTRACT: the CUDA-graph/torch.compile decode path is correct for
        # SINGLE-REQUEST decode only. Two grounded facts force this (vLLM 0.19.1):
        #   (a) CORRECTNESS: at >1 CONCURRENT request (R>1) the compiled model corrupts every
        #       request after the first (request 0 stays correct; see the R-sweep in memory
        #       vfd-safety-eval). Eager mode is correct for ALL batch sizes.
        #   (b) REPLAY: the K-candidate forward is a uniform-decode batch of n=R*K rows. vLLM
        #       only CAPTURES a uniform-decode graph at num_tokens N when max_num_seqs >= N
        #       (gpu_model_runner._dummy_run: num_reqs=min(max_num_seqs, N), and it asserts
        #       sum(scheduled)==N with 1 tok/req), and max_cudagraph_capture_size defaults to
        #       min(max_num_seqs*2, 512). So at max_num_seqs=1 the candidate graph CANNOT be
        #       captured (n=K>1 > ceiling 2) -> the compile path silently falls back to eager
        #       and the speedup is UNREACHABLE.
        # (a) and (b) point opposite ways: replay needs max_num_seqs>=K, correctness needs R==1
        # actual concurrency -- the engine can't guarantee the latter when the former holds. So
        # the compile speedup is a SINGLE-REQUEST-LATENCY opt-in (set max_num_seqs>=K AND drive
        # one request at a time), gated behind `single_stream=True`. Without that opt-in we fail
        # fast on (not eager) AND max_num_seqs>1 rather than silently mis-decode batched requests.
        # The DEFAULT / supported serving path is eager (correct for all R; the safety win runs
        # here). max_num_seqs=1 + compile is allowed but just falls back to eager per (b).
        mns = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1) or 1)
        eager = bool(getattr(vllm_config.model_config, "enforce_eager", False))
        if self.vfd_enabled and (not eager) and mns > 1 and not self.single_stream:
            raise ValueError(
                "VFD's CUDA-graph/torch.compile decode path is validated for single-request "
                f"decode only, but max_num_seqs={mns} > 1 with enforce_eager=False would corrupt "
                "requests after the first. For batched serving set enforce_eager=True (correct for "
                "all batch sizes). To opt into the single-request-latency speedup, set the vfd "
                "config flag single_stream=True (you must then drive exactly ONE request at a time; "
                f"and max_num_seqs ({mns}) must be >= num_candidates ({self.K}) for the candidate "
                "graph to capture -- otherwise it falls back to eager)."
            )
        if self.vfd_enabled and (not eager) and self.single_stream and mns > 1 and mns < self.K:
            print(
                f"[VFD] single_stream WARNING: max_num_seqs={mns} < num_candidates K={self.K}, so "
                f"vLLM cannot capture the {self.K}-row candidate graph -- the compile path will "
                f"fall back to eager (correct, but no speedup). Set max_num_seqs >= K to get the "
                f"cudagraph speedup.", flush=True,
            )

        hidden = self.model_config.get_hidden_size()
        if (p := cfg.get("value_head_path")):
            self.value_head = load_value_head(p, hidden, device)
        else:
            self.value_head = ValueHead(hidden).to(device)
        self.value_head.eval()

        # Per req_id: the [K] candidate token ids to forward NEXT, sampled at the
        # end of the previous step from the winner's warped LM logits; and the
        # candidates' LM log-probs [K] (for the ARGS fallback objective).
        self._pending_tok: dict[str, torch.Tensor] = {}
        self._pending_logp: dict[str, torch.Tensor] = {}

        # Scratch reserve for the K-candidate forward: ONE real KV block per concurrent
        # candidate (each scratch block is a private copy of a request's tail block; see
        # the module docstring). The peak need is K * (concurrently-DECODING requests); we
        # allocate R*K per step and free them after the winner's KV is committed. The backing
        # blocks are reserved in initialize_kv_cache (worker-side growth, no scheduler coop).
        #
        # Reserve = K * scratch_seqs blocks, each one full KV block (e.g. 2 MiB for Mistral-7B
        # @ block_size 16, bf16). scratch_seqs is the assumed peak concurrency:
        #   * single_stream: peak is 1 by contract (one request at a time) -> K blocks, NOT
        #     K*max_num_seqs. This matters because the compile path forces max_num_seqs>=K, so
        #     the old sizing was K*max_num_seqs >= K**2 (e.g. K=40,mns=40 -> 1600 blocks = 3.1
        #     GiB) for a workload that only ever uses K. Now K=40 single_stream = 40 blocks = 80 MiB.
        #   * batched serving (default): up to max_num_seqs requests can decode at once, so the
        #     safe worst case is K*max_num_seqs. An operator who knows their real peak can cap it
        #     with the `scratch_max_seqs` config (clamped to [1, max_num_seqs]).
        # If a step ever needs more than reserved (e.g. single_stream contract violated with >1
        # concurrent request), ScratchAllocator.allocate raises "scratch exhausted" -- a LOUD
        # failure, strictly better than silently mis-decoding.
        self._max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 256))
        if self.single_stream:
            scratch_seqs = 1
        else:
            scratch_seqs = int(cfg.get("scratch_max_seqs", self._max_num_seqs))
        self._scratch_seqs = max(1, min(scratch_seqs, self._max_num_seqs))
        self._num_scratch_blocks = self.K * self._scratch_seqs
        self._scratch = ScratchAllocator(self._num_scratch_blocks)
        self._scratch_blocks: list[int] | None = None   # set in initialize_kv_cache
        # One extra scratch block, reserved as a sink for cudagraph PAD rows (see
        # _candidate_forward / _build_candidate_metadata): when n=R*K is padded up to a
        # captured graph size, the pad rows point their block_table/slot_mapping here so
        # their (discarded) attention can't touch any real or candidate KV.
        self._pad_sink_block: int | None = None

        # CUDA-graph compatibility (the candidate forward routes the backbone through
        # captured PIECEWISE graphs when possible; see _vfd_cudagraph_dispatch). The
        # captured input buffers are the 1-D self.input_ids/self.positions; models with
        # mrope/xdrope positions use a different (2-D) buffer the parent captured instead,
        # so VFD falls back to eager for those rather than mis-feeding positions.
        self._vfd_compile_ok = not (self.uses_mrope or self.uses_xdrope_dim > 0)
        self._vfd_replay_fired = 0      # observability: # steps that replayed a graph
        # Diagnostic: per-row count of steps with no "safe" candidate (-> argmin fallback).
        self._argmin_accum = torch.zeros((), dtype=torch.long, device=device)
        self._select_accum = 0
        # Diagnostic (env VFD_DUMP_HIDDEN): {req_id: [(committed_token_id, hidden_fp32_cpu), ...]}
        # Populated in _select; consumed by scripts/decode_extract.py to capture decode features.
        self._dump_hidden = {} if os.environ.get("VFD_DUMP_HIDDEN") else None

        # Per-phase timing (VFD_PROFILE=1): CUDA-event ms accumulated per phase across steady
        # steps, dumped every _prof_every steps. Used to TARGET further optimization at the
        # phase that actually dominates rather than guessing. Adds a sync per phase, so it is
        # opt-in only (off = zero overhead, the _phase context manager no-ops).
        self._debug = bool(os.environ.get("VFD_DEBUG"))
        # A/B toggle for the _next_pos re-anchor bugfix (default: fixed). VFD_NO_ANCHOR_FIX=1
        # restores the old buggy behavior (re-anchor ALL active) to reproduce the R>1 degeneration.
        self._anchor_reset_all = bool(os.environ.get("VFD_NO_ANCHOR_FIX"))
        self._prof_on = bool(os.environ.get("VFD_PROFILE"))
        self._prof: dict[str, float] = {}
        self._prof_steps = 0
        self._prof_every = int(os.environ.get("VFD_PROFILE_EVERY", "64"))

        # CUDA-graph capture of the per-layer KV surgery (copy + commit). Those loops are
        # ~95% kernel-launch overhead (VFD_PROFILE: ~2ms/step on Mistral-7B, data moved is
        # tiny); replaying the L-layer loop as ONE captured graph collapses ~64 launches to 1.
        # Captured lazily per exact row-count (zero padding waste). The per-op gather temp
        # scales with rows*block_size, so capture is capped to small batches (where launch
        # overhead dominates anyway); larger batches fall back to the eager loop. Persistent
        # index buffers are updated in place each step and the graph replays against them.
        self._capture_kv = bool(cfg.get("capture_kv_ops", True))
        self._kv_capture_max_rows = int(cfg.get("capture_kv_max_rows", 64))
        self._kv_graphs: dict = {}
        self._kv_replayed = 0           # observability: # captured-graph replays (FIRE check)
        self._cp_src = torch.empty(self._num_scratch_blocks, dtype=torch.long, device=device)
        self._cp_dst = torch.empty(self._num_scratch_blocks, dtype=torch.long, device=device)
        self._cm_real = torch.empty(self._max_num_seqs, dtype=torch.long, device=device)
        self._cm_scratch = torch.empty(self._max_num_seqs, dtype=torch.long, device=device)
        self._cm_off = torch.empty(self._max_num_seqs, dtype=torch.long, device=device)

        # Per req_id: the absolute position of the NEXT token VFD will generate. We track
        # this ourselves rather than read input_batch.num_computed_tokens, because at the
        # bootstrap step (right after the prefill super().execute_model) that counter has
        # not yet advanced past the prompt -- it would put the first generated token at
        # position 0 and overwrite the prompt's KV. Anchored at prompt length when a
        # request is taken over, then +1 per committed token.
        self._next_pos: dict[str, int] = {}

    # ============================================================== #
    # Seam A: reserve private scratch KV blocks at worker init.       #
    # The scheduler pool spans [0, base); we grow the worker-side KV  #
    # tensors by K*scratch_seqs blocks and keep the top ids private.  #
    # ============================================================== #
    def initialize_kv_cache(self, kv_cache_config) -> None:
        """Grow the worker-side KV tensors by `self._num_scratch_blocks + 1` blocks
        (`K * scratch_seqs`, +1 cudagraph pad sink) and reserve the new top block ids as VFD
        scratch, then build the cache as usual. scratch_seqs is the assumed peak concurrency
        (1 under single_stream, else max_num_seqs or the `scratch_max_seqs` cap).

        The scheduler's KVCacheManager is configured from the ORIGINAL `num_blocks`
        (worker.initialize_from_config sets cache_config.num_gpu_blocks before calling
        us), so it only ever hands out ids in [0, base). The extra block ids
        [base, base + K*scratch_seqs) exist only in the worker's tensors and are never
        scheduled -- safe, collision-free scratch.

        NOTE: the extra blocks are allocated beyond the profiled num_blocks (within the
        gpu_memory_utilization headroom) -- validated on A100 (batched test asserts the
        reserve is K*scratch_seqs). For a very large K*scratch_seqs or a tight memory budget,
        lower max_num_seqs / scratch_max_seqs / K (single_stream already caps it at K).
        block_size is assumed equal to the attention kernel block size (the common,
        single-group case)."""
        if not self.vfd_enabled:
            return super().initialize_kv_cache(kv_cache_config)

        import copy

        # Single KV-cache group only. The candidate forward builds ONE block table /
        # slot_mapping from group 0 and shares it across all attention layers; models with
        # multiple groups (e.g. alternating sliding/full attention, or hybrid attn+mamba)
        # would need per-group tables. Fail fast rather than silently scoring a wrong prefix.
        groups = getattr(kv_cache_config, "kv_cache_groups", [])
        if len(groups) > 1:
            raise NotImplementedError(
                f"VFD supports a single KV-cache group; this model has {len(groups)} "
                "(e.g. mixed attention types). Per-group candidate metadata is not wired."
            )

        base = int(kv_cache_config.num_blocks)
        # K*max_num_seqs candidate scratch blocks + 1 pad-sink block (cudagraph pad rows).
        extra = self._num_scratch_blocks + 1
        grown = copy.deepcopy(kv_cache_config)
        grown.num_blocks = base + extra
        for t in grown.kv_cache_tensors:
            # t.size == base * page_size_bytes for the layers sharing this tensor; grow
            # it by `extra` blocks' worth of bytes so the reshape yields base+extra blocks.
            if base == 0 or t.size % base != 0:
                raise NotImplementedError(
                    "VFD scratch reserve assumes kv_cache_tensor.size is a multiple of "
                    f"num_blocks (got size={t.size}, num_blocks={base}); cannot size the "
                    "private scratch region for this KV layout."
                )
            bytes_per_block = t.size // base
            t.size = bytes_per_block * (base + extra)

        super().initialize_kv_cache(grown)
        # First num_scratch_blocks are candidate scratch (the allocator's index space);
        # the top one is the pad sink (not handed to the allocator).
        self._scratch_blocks = list(range(base, base + self._num_scratch_blocks))
        self._pad_sink_block = base + self._num_scratch_blocks

    # ============================================================== #
    # Entry point. VFD REPLACES the decode forward (the K-candidate   #
    # forward is the step's only transformer pass), so the override   #
    # is execute_model, not a post-forward sample_tokens hook.        #
    #                                                                  #
    # Control flow:                                                    #
    #   * any request without pending candidates (first step / still  #
    #     prefilling) -> delegate to super() for a normal forward,     #
    #     then seed those requests from the base hidden.               #
    #   * otherwise (steady-state decode, all seeded) -> the           #
    #     K-candidate forward IS the step: candidate_forward -> select #
    #     (scores, commits winner KV, reseeds) -> assemble output.     #
    #                                                                  #
    # Mixed (some seeded, some not) steps fall back to the base forward #
    # -- VALIDATED on A100 (batched/different-length test). Async        #
    # scheduling is rejected in __init__ (no async output path).         #
    # ============================================================== #
    def execute_model(self, scheduler_output, *args, **kwargs):
        if not self.vfd_enabled:
            return super().execute_model(scheduler_output, *args, **kwargs)
        try:
            self._drop_finished(scheduler_output)
            # Decide the branch from scheduler_output (not the possibly-stale input_batch):
            # a step is steady-state VFD iff every scheduled request is a seeded pure decode
            # (exactly one scheduled token AND already taken over by VFD).
            sched = scheduler_output.num_scheduled_tokens               # req_id -> count
            scheduled = list(sched.keys())
            steady = bool(scheduled) and all(
                (r in self._pending_tok and sched[r] == 1) for r in scheduled
            )

            if not steady:
                # Prefill / bootstrap / mixed. Run the base forward (it writes prompt KV,
                # refreshes the persistent batch, and stashes the post-norm hidden), then
                # seed candidates from that hidden. CRUCIAL: the seeded candidates are
                # alternatives for the very position the base step would emit, so we must
                # CONSUME them THIS step -- run the candidate forward and emit the VFD
                # winner -- rather than letting base sample (which would take that position
                # and desync VFD by one). We return a ModelRunnerOutput so the engine skips
                # sample_tokens (engine/core.py: sample_tokens runs only if output is None).
                super().execute_model(scheduler_output, *args, **kwargs)
                active = self._active_req_ids()                  # fresh: super ran _update_states
                self._seed_from_base(active)
                if active and all(r in self._pending_tok for r in active):
                    self._reject_logprobs(active)
                    # Anchor each NEWLY-seeded request's generation position at its prompt
                    # length (first generated token sits right after the prompt). BUGFIX: only
                    # anchor requests we haven't seen -- this branch also fires on a "mixed" step
                    # when a NEW request joins an in-flight batch (continuous batching), and
                    # re-anchoring an ONGOING decoder would reset its position back to prompt-end,
                    # overwriting its generated KV and corrupting output (worse the more the batch
                    # churns -- the R>1 degeneration). Ongoing decoders keep their self-tracked
                    # _next_pos (set here once, then +1 per commit in _select).
                    for r in active:
                        if (not self._anchor_reset_all) and (r in self._next_pos):
                            continue
                        idx = self.input_batch.req_id_to_index[r]
                        self._next_pos[r] = int(self.input_batch.num_prompt_tokens[idx])
                    self._dbg("bootstrap: seeded+emitting", active,
                              "pos", [self._next_pos[r] for r in active])
                    self.execute_model_state = None              # consumed; skip base sampling
                    h_cand, scratch_idx, plan = self._candidate_forward(active)
                    winners = self._select(active, h_cand, scratch_idx, plan)
                    return self._build_output(active, winners)
                # Genuinely mixed (some still prefilling): let base emit this step and drop
                # any partial seeds so we only take over once the whole batch is decodable.
                for r in [x for x in self._pending_tok if x in active]:
                    self._pending_tok.pop(r, None)
                    self._pending_logp.pop(r, None)
                    self._next_pos.pop(r, None)
                self._dbg("mixed step -> base forward", active)
                return None                                      # engine calls sample_tokens

            # Steady-state decode: the candidate forward IS this step's forward. VFD bypasses
            # the base input-prep, so refresh the persistent batch (req set, block tables,
            # num_computed_tokens) and push the block table to device ourselves.
            self._update_states(scheduler_output)
            self.input_batch.block_table.commit_block_table(self.input_batch.num_reqs)
            active = self._active_req_ids()
            self._reject_logprobs(active)
            h_cand, scratch_idx, plan = self._candidate_forward(active)   # [R,K,H] + bookkeeping
            winners = self._select(active, h_cand, scratch_idx, plan)     # score+commit+reseed
            out = self._build_output(active, winners)
            self._prof_maybe_dump()
            return out
        except NotImplementedError:
            raise
        except Exception:
            if self.strict:
                raise
            # Production: never crash decoding -- fall back to a normal step.
            return super().execute_model(scheduler_output, *args, **kwargs)

    def _reject_logprobs(self, active_req_ids) -> None:
        """VFD's single-forward output (_build_output) emits one committed token id per
        request with no logprobs/prompt_logprobs. If a VFD-path request asks for them, fail
        loudly (NotImplementedError propagates past execute_model's swallow) rather than
        silently returning None. Extend _build_output to lift this."""
        for r in active_req_ids:
            sp = self.requests[r].sampling_params
            if getattr(sp, "logprobs", None) or getattr(sp, "prompt_logprobs", None):
                raise NotImplementedError(
                    f"VFD does not produce logprobs/prompt_logprobs (request {r}); disable "
                    "them for VFD requests or extend _build_output."
                )

    def _dbg(self, *parts) -> None:
        if self._debug:
            print("[VFD]", *parts, flush=True)

    @contextlib.contextmanager
    def _phase(self, name: str):
        """Time a steady-step phase with CUDA events when VFD_PROFILE=1; else a no-op.
        Accumulates ms into self._prof[name]; the caller dumps periodically (_prof_maybe_dump)."""
        if not self._prof_on:
            yield
            return
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            yield
        finally:
            e.record()
            torch.cuda.synchronize()
            self._prof[name] = self._prof.get(name, 0.0) + s.elapsed_time(e)

    def _prof_maybe_dump(self) -> None:
        """Print mean per-phase ms over the last _prof_every steady steps, then reset."""
        if not self._prof_on:
            return
        self._prof_steps += 1
        if self._prof_steps % self._prof_every:
            return
        n = self._prof_every
        total = sum(self._prof.values()) or 1.0
        parts = " ".join(
            f"{k}={self._prof[k] / n:.3f}ms({100 * self._prof[k] / total:.0f}%)"
            for k in sorted(self._prof, key=lambda k: -self._prof[k])
        )
        print(f"[VFD-PROFILE] steps={n} mean/step: {parts}", flush=True)
        self._prof.clear()

    # ============================================================== #
    # Seam 1 (concrete): sample K candidates from the warped LM dist. #
    #   compute_logits is the LM head only -> cheap, not a forward.   #
    # ============================================================== #
    @torch.inference_mode()
    def _seed_candidates(self, hidden_by_req: dict[str, torch.Tensor]) -> None:
        for req_id, h in hidden_by_req.items():
            logits = self.model.compute_logits(h.unsqueeze(0)).squeeze(0)   # [vocab]
            sp = self.requests[req_id].sampling_params
            temp = getattr(sp, "temperature", None)
            temp = 1.0 if temp is None else float(temp)
            if temp <= 0.0:
                # Greedy request: every candidate is the argmax token, so VFD commits the
                # greedy token (matching base greedy decode token-for-token). vLLM treats
                # temperature==0 as greedy; mirror that here instead of warping by it.
                tok = int(logits.argmax())
                self._pending_tok[req_id] = torch.full(
                    (self.K,), tok, device=logits.device, dtype=torch.long
                )
                self._pending_logp[req_id] = torch.zeros(self.K, device=logits.device)
                continue
            warped = self._warp(logits, req_id)                              # temp/top-p
            probs = F.softmax(warped, dim=-1)
            toks = torch.multinomial(probs, self.K, replacement=True)        # [K]
            self._pending_tok[req_id] = toks
            self._pending_logp[req_id] = torch.log(probs[toks].clamp_min(1e-12))

    def _warp(self, logits: torch.Tensor, req_id: str) -> torch.Tensor:
        """Apply the request's temperature/top-p so candidates are drawn from the
        same distribution vLLM would sample. TODO(gpu): for exact parity, reuse the
        request's vLLM logits-processor stack (top_k/min_p/penalties) instead of this
        temperature+top_p subset."""
        sp = self.requests[req_id].sampling_params
        temperature = float(getattr(sp, "temperature", 1.0) or 1.0)
        top_p = float(getattr(sp, "top_p", 1.0) or 1.0)
        return warp_logits(logits, temperature, top_p)

    def _vfd_cudagraph_dispatch(self, n: int):
        """Decide whether the n=R*K candidate forward can replay a captured graph.

        The candidate forward is a pure uniform-decode batch (every row query_len==1) of n
        rows -- structurally a normal decode batch vLLM already captures PIECEWISE graphs for.
        We dispatch into those via the parent's CudagraphDispatcher, which rounds n UP to the
        nearest captured size and returns the matching BatchDescriptor (its num_tokens is the
        padded width). Returns (CUDAGraphMode, BatchDescriptor).

        FULL is EXCLUDED (invalid_modes): a FULL graph captures attention with static metadata
        buffers, incompatible with VFD's per-step dynamic scratch block-tables/slot_mapping --
        replaying into it would mis-decode. With FULL excluded, dispatch returns PIECEWISE
        (attention stays eager -> our slot_mapping still drives the KV write) or, when no graph
        fits (cudagraphs off, enforce_eager, n > max capture size) or the model uses mrope/
        xdrope positions, NONE -> the caller runs today's eager path. This exclusion is the
        single located guard against ever entering a FULL graph; no separate raise is needed
        because the NONE fallback is always correct (just unaccelerated)."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor
        if not self._vfd_compile_ok:
            return CUDAGraphMode.NONE, BatchDescriptor(n)
        return self.cudagraph_dispatcher.dispatch(
            num_tokens=n, uniform_decode=True, invalid_modes={CUDAGraphMode.FULL}
        )

    # ============================================================== #
    # Seam 2 (VALIDATED on A100): forward the pending K candidates per #
    # request as R*K decode rows. Each row shares the request's full  #
    # prefix blocks and adds ONE private scratch block (a copy of the #
    # request's tail block) into which the row's new-token KV is       #
    # written. Standard paged decode via _model_forward -- backend-   #
    # agnostic for the read; the scratch copy is the only KV surgery. #
    # ============================================================== #
    @torch.inference_mode()
    def _candidate_forward(self, active_req_ids: list[str]):
        R, K = len(active_req_ids), self.K
        n = R * K
        device = self.device

        # Flatten candidates to [n] and their positions (each candidate sits at the
        # next position VFD will generate for its request -- tracked in self._next_pos).
        toks = torch.cat([self._pending_tok[r] for r in active_req_ids])        # [n]
        seq_lens_per_req = [self._next_pos[r] for r in active_req_ids]          # context len
        positions = torch.tensor(
            [sl for sl in seq_lens_per_req for _ in range(K)],
            device=device, dtype=torch.long,
        )                                                                       # [n]

        # CUDA-graph dispatch: route the backbone through a captured PIECEWISE graph when one
        # fits (n padded up to a captured size), else eager (NONE -> n_pad == n). Omitting
        # this (the old behavior) defaulted cudagraph_runtime_mode to NONE, forcing the
        # backbone to run op-by-op every step -- the sole reason VFD required enforce_eager.
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        cg_mode, batch_desc = self._vfd_cudagraph_dispatch(n)
        n_pad = int(batch_desc.num_tokens)                                      # == n if NONE

        with self._phase("meta"):
            cm, scratch_idx, plan = self._build_candidate_metadata(
                active_req_ids, seq_lens_per_req, n_pad
            )
        try:
            # Make each candidate's scratch block a copy of the request's real tail
            # block, so attention reads the real prefix tail; the forward then writes
            # the candidate's new-token KV at offset p % block_size. (Pad rows are not in
            # `plan` -- they read/write the dedicated pad sink and are sliced off below.)
            with self._phase("copy"):
                self._copy_real_tail_to_scratch(plan)
            with self._phase("meta"):
                attn_metadata = self._build_per_layer_metadata(cm)
            # The KV-write op (unified_kv_cache_update) reads the new-token slot from
            # forward_context.slot_mapping[layer_name], NOT from attn_metadata. Pass our
            # candidate slot_mapping per layer, else the candidate KV lands in a stale slot
            # and attention reads garbage at the new token's position (single-group case;
            # all attention layers share the one slot tensor). Under PIECEWISE attention is
            # eager, so this still drives the KV write on the graph path.
            slot_mapping_by_layer = {
                ln: cm.slot_mapping
                for g in self.kv_cache_config.kv_cache_groups
                for ln in g.layer_names
            }
            if cg_mode != CUDAGraphMode.NONE:
                # PIECEWISE-replay path -- VALIDATED on A100: greedy==base under
                # cudagraphs with replay actually firing, padded case included
                # (test_vfd_compiled_matches_base_and_replays).
                # The PIECEWISE graph replays against the persistent input buffers at the
                # addresses fixed at capture (cuda_graph.py does NOT copy inputs on replay).
                # So write candidate inputs INTO self.input_ids/self.positions and pass the
                # SAME slices the parent captured (buf[:n_pad] shares buf's base data_ptr).
                # Pad rows [n:n_pad] get token 0 / position 0; their (discarded) attention
                # reads the pad sink. _update_states does not touch these buffers, so VFD
                # owns them for this step.
                self.input_ids.gpu[:n].copy_(toks)         # int32 buffer <- long ids (cast)
                self.input_ids.gpu[n:n_pad].zero_()
                self.positions[:n].copy_(positions)
                self.positions[n:n_pad].zero_()
                in_ids = self.input_ids.gpu[:n_pad]
                in_pos = self.positions[:n_pad]
            else:
                in_ids, in_pos = toks, positions
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=n_pad,
                                     cudagraph_runtime_mode=cg_mode,
                                     batch_descriptor=batch_desc,
                                     slot_mapping=slot_mapping_by_layer), self._phase("forward"):
                hs = self._model_forward(input_ids=in_ids, positions=in_pos)    # [n_pad, H]
            if not isinstance(hs, torch.Tensor):                                # some models wrap
                hs = hs[0] if isinstance(hs, (tuple, list)) else hs.last_hidden_state
            if cg_mode != CUDAGraphMode.NONE:
                self._vfd_replay_fired += 1     # observability: replay actually engaged
            return hs[:n].view(R, K, -1), scratch_idx, plan
        except Exception:
            # On failure free scratch here; on success _select frees it after commit
            # (the winner's KV must still be readable from scratch at commit time).
            self._scratch.free(scratch_idx)
            raise

    def _build_candidate_metadata(self, active_req_ids, seq_lens_per_req, n_pad=None):
        """Construct CommonAttentionMetadata for R*K single-token decode rows (padded to
        n_pad rows on the cudagraph path) plus the per-candidate commit plan.

        Row layout per candidate (request i, candidate k, position p = seq_lens[i]):
          * block_table[row] = real prefix blocks [0, p//block_size) + the candidate's
            private scratch block at index p//block_size (the tail block);
          * slot_mapping[row] = scratch_block * block_size + (p % block_size) -- where the
            forward writes this candidate's new-token K/V;
          * seq_lens[row] = p + 1 (prefix + the new token).

        Returns (common_metadata, scratch_idx, plan). `plan` carries the tensors needed to
        (a) copy each request's real tail block into the scratch block before the forward
        and (b) commit the winner's slot afterward, both keyed by the flattened row index.

        VALIDATED on A100 (single KV-cache group): the slot/block math and position
        tracking are correct -- VFD greedy reproduces base greedy token-for-token. block_size
        vs kernel-block-size alignment is asserted in _kv_validate (raises if they differ)."""
        from vllm.v1.attention.backend import CommonAttentionMetadata

        R, K = len(active_req_ids), self.K
        n = R * K
        device = self.device
        block_size = self.cache_config.block_size

        scratch_idx = self._scratch.allocate(n)                # [n] indices into _scratch_blocks
        scratch_block_ids = self._scratch_block_ids()          # reserved real blocks

        # Group 0's block table ([num_reqs, max_blocks_per_req]). Single KV-cache group is
        # guaranteed here -- initialize_kv_cache raises for multi-group models. Read the CPU
        # MIRROR (get_cpu_tensor), not get_device_tensor().to("cpu"): the latter is a per-step
        # D2H sync that stalls the CPU on the prior GPU step. The CPU mirror is authoritative and
        # current -- execute_model called commit_block_table (CPU->GPU) just before us -- so this
        # reads it sync-free and lets the CPU build metadata while the GPU finishes the last step.
        prefix_bt_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        bt_dtype = prefix_bt_cpu.dtype
        prefix_blocks = [
            prefix_bt_cpu[self.input_batch.req_id_to_index[r]].tolist() for r in active_req_ids
        ]
        # Pure index math (CPU-unit-tested in tests/test_scratch_alloc.py): block table +
        # slot mapping + per-candidate commit plan. Keeping it vLLM/torch-free is what lets
        # the high-risk slot/block arithmetic be a regression test without a GPU.
        layout = candidate_block_layout(
            seq_lens_per_req, prefix_blocks, scratch_block_ids, scratch_idx, K, block_size
        )

        bt = torch.tensor(layout["block_table"], device=device, dtype=bt_dtype)
        slot_mapping = torch.tensor(layout["slot_mapping"], device=device, dtype=torch.long)
        seq_lens = torch.tensor(
            [sl + 1 for sl in seq_lens_per_req for _ in range(K)],
            device=device, dtype=torch.int32,
        )                                                                      # prefix + new token

        # cudagraph padding: extend the forward-consumed metadata (block_table / slot_mapping
        # / seq_lens) from n real rows to n_pad with sink rows. Each pad row is a length-1
        # sequence on the dedicated pad-sink block (seq_len 1, slot = sink*block_size); its
        # attention reads/writes only the sink, never a real or candidate block, and its
        # hidden row is discarded in _candidate_forward. The commit `plan` stays length n --
        # pad rows never win, never copy, never commit.
        n_pad = n if n_pad is None else int(n_pad)
        if n_pad > n:
            sink = self._pad_sink_block
            if sink is None:
                raise NotImplementedError(
                    "cudagraph pad sink not reserved: initialize_kv_cache must run before "
                    "the padded candidate forward (it reserves the +1 sink block)."
                )
            pad = n_pad - n
            width = bt.shape[1]
            bt = torch.cat(
                [bt, torch.full((pad, width), sink, device=device, dtype=bt_dtype)], 0
            )
            slot_mapping = torch.cat(
                [slot_mapping,
                 torch.full((pad,), sink * block_size, device=device, dtype=torch.long)], 0
            )
            seq_lens = torch.cat(
                [seq_lens, torch.ones(pad, device=device, dtype=torch.int32)], 0
            )
        qsl = torch.arange(n_pad + 1, device=device, dtype=torch.int32)        # one query/row

        cm = CommonAttentionMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl.cpu(),
            seq_lens=seq_lens,
            num_reqs=n_pad,
            num_actual_tokens=n_pad,
            max_query_len=1,
            max_seq_len=int(max(seq_lens_per_req)) + 1,
            block_table_tensor=bt,
            slot_mapping=slot_mapping,
            causal=True,
        )
        plan = {
            "scratch_blk": torch.tensor(layout["scratch_blk"], device=device, dtype=torch.long),
            "real_tail_blk": torch.tensor(layout["real_tail_blk"], device=device, dtype=torch.long),
            "offset": torch.tensor(layout["offset"], device=device, dtype=torch.long),
            "needs_copy": torch.tensor(layout["needs_copy"], device=device, dtype=torch.bool),
        }
        self._dbg("candidate meta: positions", seq_lens_per_req,
                  "tail_blocks", layout["real_tail_blk"][::K] if K else layout["real_tail_blk"],
                  "offsets", layout["offset"][::K] if K else layout["offset"],
                  "block_size", block_size)
        return cm, scratch_idx, plan

    def _build_per_layer_metadata(self, cm):
        """Run the attention-metadata builder for each attention group and map the
        result onto every layer in that group. Single-KV-group / single-attn-group is
        the only case VFD allows (initialize_kv_cache raises for multi-group models)."""
        attn_metadata: dict = {}
        for kv_gid, groups in enumerate(self.attn_groups):
            for group in groups:
                builder = group.get_metadata_builder()
                md = builder.build(common_prefix_len=0, common_attn_metadata=cm)
                for layer_name in group.layer_names:
                    attn_metadata[layer_name] = md
        return attn_metadata

    def _scratch_block_ids(self) -> list[int]:
        """Real KV-cache block ids reserved for scratch candidate KV, set by the
        initialize_kv_cache override (worker-side growth of the KV tensors). Raises only
        if called before the cache was initialized (an invariant guard, not a gap)."""
        ids = self._scratch_blocks
        if ids is None:
            raise NotImplementedError(
                "scratch KV blocks not reserved yet: initialize_kv_cache must run before "
                "the candidate forward (it grows the worker KV tensors by K*max_num_seqs "
                "blocks and records their ids)."
            )
        return ids

    # -------------------------------------------------------------- #
    # KV-cache layout helper. FlashAttention stores each layer's cache as a logical
    # (2, num_blocks, block_size, num_kv_heads, head_size) tensor (kv_cache.unbind(0)
    # in the backend); indexing this logical view honors the backend stride order
    # automatically, so a direct slot copy needs no manual stride handling. Other
    # layouts (e.g. hybrid attn+mamba's (num_blocks, 2, ...)) are not supported here.
    # -------------------------------------------------------------- #
    def _kv_validate(self, kv_cache):
        """Validate the FlashAttention (2, num_blocks, block_size, num_kv_heads, head_size)
        layout and return the raw tensor. Callers index the BLOCK dim while keeping dim 0
        (the k/v axis), so one op moves both key and value -- half the kernel launches of
        copying key_cache and value_cache separately (the KV surgery is launch-bound; see
        VFD_PROFILE)."""
        block_size = self.cache_config.block_size
        if not (isinstance(kv_cache, torch.Tensor) and kv_cache.dim() == 5
                and kv_cache.shape[0] == 2):
            raise NotImplementedError(
                "VFD scratch KV surgery supports the FlashAttention "
                "(2, num_blocks, block_size, num_kv_heads, head_size) layout only; got "
                f"{getattr(kv_cache, 'shape', type(kv_cache))!r}. Wire the equivalent "
                "cache-copy for this backend."
            )
        if kv_cache.shape[2] != block_size:
            raise NotImplementedError(
                f"kernel block size ({kv_cache.shape[2]}) != cache_config.block_size "
                f"({block_size}); VFD slot/block math assumes they match (single-group)."
            )
        return kv_cache

    def _run_captured(self, key, run) -> None:
        """Manual mid-decode cudagraph capture of the in-place KV scatter -- VALIDATED on A100
        (test_vfd_kv_ops_captured_match_base: greedy==base + _kv_replayed>0; profile: copy+commit
        3.31ms -> 0.90ms vs the original per-op loop). On capture failure it degrades to the
        eager loop (re-raises under strict).

        Run `run` (a pure-device per-layer KV op reading persistent index buffers) under a
        lazily-captured CUDA graph, replayed by `key` on later steps. The capture pass itself
        executes `run` once, so this step's KV write happens whether we capture or replay. On
        any capture failure: re-raise under strict (tests catch it), else disable capture and
        run eager so production never breaks. The index buffers must already hold this step's
        values; `run` must contain NO host syncs (no .item/.cpu) or capture will error."""
        g = self._kv_graphs.get(key)
        if g is not None:
            g.replay()
            self._kv_replayed += 1
            return
        try:
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run()                       # warmup (idempotent copy) so capture is clean
                run()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()                       # capture pass -> also performs THIS step's op
            self._kv_graphs[key] = g
            self._kv_replayed += 1
        except Exception:
            self._capture_kv = False        # degrade permanently; don't thrash captures
            self._kv_graphs.clear()
            if self.strict:
                raise
            run()                           # ensure this step's op still lands

    @torch.inference_mode()
    def _copy_real_tail_to_scratch(self, plan) -> None:
        """For each candidate whose new token shares a block with real prefix tokens
        (offset > 0), copy the request's real tail block into the candidate's scratch
        block, per layer, so attention reads the correct prefix tail. Block-aligned
        candidates (offset == 0) start a fresh block and need no copy."""
        nrows = plan["scratch_blk"].shape[0]
        if self._capture_kv and nrows <= self._kv_capture_max_rows:
            # Capturable variant: FIXED shape (all n rows, no mask). No-copy rows become
            # self-copies (src == dst == scratch block) -- a harmless re-write, since the
            # forward overwrites the new-token slot and attention is bounded by seq_len.
            # Masking would make the row count vary per step and break graph capture.
            src = torch.where(plan["needs_copy"], plan["real_tail_blk"], plan["scratch_blk"])
            self._cp_src[:nrows].copy_(src)
            self._cp_dst[:nrows].copy_(plan["scratch_blk"])

            def run():
                s_, d_ = self._cp_src[:nrows], self._cp_dst[:nrows]
                for kv_cache in self.kv_caches:
                    kv = self._kv_validate(kv_cache)
                    kv[:, d_] = kv[:, s_]
            self._run_captured(("copy", nrows), run)
            return
        # Eager path (capture off or batch too large): mask to the rows that need a copy.
        mask = plan["needs_copy"]
        if not bool(mask.any()):
            return
        src = plan["real_tail_blk"][mask]        # [m] real tail blocks
        dst = plan["scratch_blk"][mask]          # [m] scratch blocks
        for kv_cache in self.kv_caches:
            kv = self._kv_validate(kv_cache)
            kv[:, dst] = kv[:, src]              # whole-block copy, key+value in one op

    # ============================================================== #
    # Seam 3 (concrete): first-safe / safest selection, commit, reseed#
    # ============================================================== #
    @torch.inference_mode()
    def _select(self, active_req_ids: list[str], h_cand: torch.Tensor,
                scratch_idx, plan) -> dict[str, int]:
        R, K, H = h_cand.shape
        with self._phase("score"):
            p_unsafe = self.value_head.p(h_cand.reshape(R * K, H)).reshape(R, K)  # [R,K]
            thresholds = torch.tensor(
                [self._threshold(r) for r in active_req_ids],
                device=p_unsafe.device,
                dtype=p_unsafe.dtype,
            )
            logp = (
                torch.stack([self._pending_logp[r] for r in active_req_ids])
                if self.args_fallback
                else None
            )
            winner_col = select_vfd(                          # [R] long; first-safe / argmin / ARGS
                p_unsafe,
                thresholds,
                args_fallback=self.args_fallback,
                prob_weight=self.prob_weight,
                logp=logp,
            )
            # Diagnostic (sync-free, device-accumulated): how often NO candidate is "safe", so
            # selection falls to argmin(p_unsafe) -- which on harmful prompts chains the head's
            # "safest" off-distribution token -> degeneration. High argmin-rate => threshold too
            # strict for the prompt distribution (needs calibration), not a candidate-count issue.
            no_safe = (p_unsafe < thresholds.view(-1, 1)).any(dim=1).logical_not().sum()
            self._argmin_accum += no_safe
            self._select_accum += R

        # Gather winner token + winner hidden ON DEVICE, then ONE .tolist() D2H for the tokens
        # (the engine output needs py ints) -- vs the old 2R per-row int(winner_col[i]) /
        # int(pending_tok[r][col]) host syncs. winner_hidden stays on device (no sync).
        R_ = len(active_req_ids)
        pend = torch.stack([self._pending_tok[r] for r in active_req_ids])      # [R, K]
        wtok = pend.gather(1, winner_col.view(-1, 1)).squeeze(1).tolist()       # [R] py ints (1 D2H)
        wh = h_cand[torch.arange(R_, device=h_cand.device), winner_col]         # [R, H] device gather
        winners = {r: wtok[i] for i, r in enumerate(active_req_ids)}
        winner_hidden = {r: wh[i] for i, r in enumerate(active_req_ids)}

        # DIAGNOSTIC (off unless VFD_DUMP_HIDDEN set): record the EXACT post-norm hidden the value
        # head was scored on for each committed token, so it can be compared to the training-time
        # (pooling) feature for the same tokens -- answers "train feature == inference feature?".
        if self._dump_hidden is not None:
            for i, r in enumerate(active_req_ids):
                self._dump_hidden.setdefault(r, []).append(
                    (int(wtok[i]), wh[i].detach().to(torch.float32).cpu()))

        # Commit the winner's KV into each request's real tail slot (pure in-cache copy,
        # NO second forward). Load-bearing for the single-forward scheme: without it the
        # next candidate forward would see a prefix missing this token's KV.
        with self._phase("commit"):
            self._commit_winner_kv(active_req_ids, winner_col, plan)
        self._scratch.free(scratch_idx)        # candidate KV consumed; release the blocks

        for r in active_req_ids:               # this step filled _next_pos[r]; advance by one
            self._next_pos[r] += 1
        with self._phase("seed"):
            self._seed_candidates(winner_hidden)   # candidates for the NEXT step
        return winners

    # -------------------------------------------------------------- #
    def _threshold(self, req_id: str) -> float:
        return request_threshold(
            self.requests[req_id].sampling_params, "vfd_threshold", self.default_thr
        )

    # ============================================================== #
    # Seam B (VALIDATED on A100): commit winner K/V WITHOUT a model   #
    # forward. The candidate forward already wrote every candidate's  #
    # new-token K/V into its scratch block at offset p % block_size;  #
    # the winner's K/V is therefore already in the cache. Commit =    #
    # copy that one slot from the scratch block into the request's    #
    # REAL tail block at the same offset, per layer. The real tail    #
    # block already holds positions [tail_start, p-1]; this adds p.   #
    # ============================================================== #
    @torch.inference_mode()
    def _commit_winner_kv(self, active_req_ids, winner_col, plan) -> None:
        K = self.K
        R = len(active_req_ids)
        device = self.device
        # Build the winner row indices ON DEVICE (rows = i*K + winner_col[i]); no per-row
        # int(winner_col[i]) host sync. winner_col is [R] long from select_vfd.
        rows = torch.arange(R, device=device, dtype=torch.long) * K + winner_col.to(torch.long)
        scratch_blk = plan["scratch_blk"][rows]    # [R]
        real_blk = plan["real_tail_blk"][rows]     # [R]
        offset = plan["offset"][rows]              # [R]
        # VALIDATED on A100: the winner's real target block IS allocated the step it's
        # generated (same-step commit; the guard below never fired across the greedy/multi-step
        # runs). Kept as an invariant guard: if a future backend/scheduler allocated it a step
        # late, this surfaces loudly instead of corrupting block 0 (would need deferred commit).
        base = self._scratch_blocks[0] if self._scratch_blocks else None
        if self._debug:
            self._dbg("commit:", "real_blk", real_blk.tolist(), "scratch_blk", scratch_blk.tolist(),
                      "offset", offset.tolist(), "scratch_base", base)
        if base is not None and bool((real_blk >= base).any()):   # one .any() sync; outside capture
            raise RuntimeError(
                "VFD commit target points into the scratch range -- winner's real block "
                "was not allocated by the scheduler this step (commit-timing: deferral "
                f"needed). real_blk={real_blk.tolist()} scratch_base={base}"
            )
        if self._capture_kv and R <= self._kv_capture_max_rows:
            self._cm_real[:R].copy_(real_blk)
            self._cm_scratch[:R].copy_(scratch_blk)
            self._cm_off[:R].copy_(offset)

            def run():
                r_, s_, o_ = self._cm_real[:R], self._cm_scratch[:R], self._cm_off[:R]
                for kv_cache in self.kv_caches:
                    kv = self._kv_validate(kv_cache)
                    kv[:, r_, o_] = kv[:, s_, o_]
            self._run_captured(("commit", R), run)
            return
        for kv_cache in self.kv_caches:
            kv = self._kv_validate(kv_cache)
            kv[:, real_blk, offset] = kv[:, scratch_blk, offset]   # key+value in one op

    # ----------------------- lifecycle / output -------------------- #
    def _active_req_ids(self) -> list[str]:
        ids = self.input_batch.req_ids
        return [ids[i] for i in range(self.input_batch.num_reqs) if ids[i] is not None]

    def _drop_finished(self, scheduler_output) -> None:
        """Clear candidate state for requests that finished/aborted this step, so
        _pending_tok / _pending_logp don't leak across the run."""
        finished = set(getattr(scheduler_output, "finished_req_ids", None) or [])
        live = set(self._active_req_ids())
        for d in (self._pending_tok, self._pending_logp, self._next_pos):
            for r in [r for r in d if r in finished or r not in live]:
                d.pop(r, None)

    @torch.inference_mode()
    def _seed_from_base(self, req_ids: list[str]) -> None:
        """Seed candidates for newly-decodable requests from the base forward's last
        hidden (stashed by super().execute_model in execute_model_state). The row<->request
        mapping mirrors abstention's (sample_hidden_states rows are 1:1 with requests in
        non-spec decode) -- VALIDATED on A100 for R>1 by the batched greedy==base test."""
        st = getattr(self, "execute_model_state", None)
        if st is None or getattr(st, "sample_hidden_states", None) is None:
            return
        h = st.sample_hidden_states                       # [num_reqs, H], post-norm
        ids = self._active_req_ids()
        hidden_by_req = {r: h[i] for i, r in enumerate(ids) if r in req_ids}
        self._seed_candidates(hidden_by_req)

    def _build_output(self, active_req_ids: list[str], winners: dict[str, int]):
        """Assemble a ModelRunnerOutput with one committed token per request. Minimal by
        design: logprobs/prompt_logprobs are rejected upstream (_reject_logprobs) and async
        scheduling is rejected in __init__, so this only carries sampled_token_ids."""
        from vllm.v1.outputs import ModelRunnerOutput
        ids = self._active_req_ids()
        return ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[winners[r]] for r in ids],
            logprobs=None,
            prompt_logprobs_dict={},
        )


# ---------------------------------------------------------------------------
# Wiring: VFD replaces the decode forward, so point the worker's runner class at
# VFDModelRunner before the worker builds it (plugin/--worker-cls/patch). Launch:
#   vllm serve <model> --additional-config '{"vfd": {
#       "enabled": true, "num_candidates": 8, "threshold": 0.5,
#       "args_fallback": false, "prob_weight": 1.0,
#       "value_head_path": "/path/to/value_head.pt"}}'
#   (no --speculative_config)
