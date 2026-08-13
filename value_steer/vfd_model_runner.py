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
correct, just unaccelerated. Capture works at ALL batch sizes -- R==1 AND R>1 concurrent;
the R>1 graph needs max_num_seqs>=R*K (vLLM captures the n=R*K uniform-decode graph only
when max_num_seqs>=that size), above which the candidate forward falls back to the eager path.
VALIDATED on A100 (vLLM 0.19.1, opt-125m): with cudagraphs ON, VFD greedy
reproduces base greedy token-for-token AND the graph actually replayed (_vfd_replay_fired>0)
-- including the padded-batch case (n straddling a capture size). See
tests/test_gpu_behavioral.py::test_vfd_compiled_matches_base_and_replays.

*** COMPILED BATCHED DECODE (enforce_eager=False, max_num_seqs>1, R>1) -- SUPPORTED & CORRECT ***
Compiled batched VFD decode is correct for ALL R -- there is no single-stream restriction. Two
fixes make it correct, both gated on self._model_compiled so the EAGER token stream stays
byte-for-byte unchanged:
  (a) fast_build=True for the candidate FA-metadata build (self._candidate_fast_build) sidesteps
      FlashAttention-3's persistent AOT scheduler_metadata buffer that the n=R*K candidate batch
      otherwise overflows (a crash at flash_attn.py:507).
  (b) the candidate forward populates the runner's persistent input_ids/positions buffers even on
      the eager-fallback (cg_mode=NONE) path -- the in-place torch.compile'd model reads THOSE
      buffers, not the tensors passed to _model_forward; without it reqs after the first decoded
      against stale buffer contents (garbled "request 0" content).
There is NO fail-fast guard: __init__ no longer raises for (enforce_eager=False AND max_num_seqs>1).
single_stream is NOT a correctness gate -- it now ONLY sizes the scratch reserve (K blocks under
single_stream, else K*max_num_seqs). Batched CUDA-graph CAPTURE is enabled too: the candidate
forward is captured at R==1 AND R>1 whenever its n=R*K rows fit a captured graph. Capturing the
R>1 graph needs max_num_seqs>=R*K (vLLM captures an N-row uniform-decode graph only when
max_num_seqs>=N; ceiling = min(max_num_seqs*2, 512)), above which it falls back to the still-
inductor-compiled eager path. Measured faster than eager across R. EAGER (enforce_eager=True) stays
correct for all R and is the serving default.

NOTE (load-bearing): the KV-write op reads the new-token slot from
forward_context.slot_mapping[layer_name], NOT from attn_metadata -- so _candidate_forward
MUST pass slot_mapping= to set_forward_context (see there), else candidate KV lands in a
stale slot and the hidden silently diverges from base.

Config: vllm_config.additional_config["vfd"]; per-request threshold via extra_args.
Spec decode must be OFF.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from .scratch_alloc import ScratchAllocator, candidate_block_layout
from .steering_ops import select_vfd

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
        # single_stream is NOT a correctness gate -- compiled batched serving is correct for all R
        # (see the docstring). It ONLY sizes the scratch reserve: when set, peak concurrency is
        # assumed to be 1, so the reserve is K blocks instead of K*max_num_seqs (a big saving, since
        # the compile path forces max_num_seqs>=K). Leave it False for concurrent serving; set it
        # only when you truly drive one request at a time (offline / single-user / benchmarking).
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
        # torch.compile / CUDA-graph decode. Correct for ALL batch sizes (eager and compiled) after
        # the two compiled-batched fixes documented at self._model_compiled below. The candidate
        # forward is cudagraph-CAPTURED whenever the n=R*K rows fit a captured graph -- at R==1 AND R>1
        # (faster than eager, validated correct at R=2/4/8/16). vLLM captures a uniform-decode
        # graph at N rows only when max_num_seqs>=N (gpu_model_runner._dummy_run) and
        # max_cudagraph_capture_size defaults to min(max_num_seqs*2, 512), so capturing the R>1 graph
        # needs max_num_seqs>=R*K; above the ceiling the candidate forward falls back to the (still
        # inductor-compiled) eager path. See _vfd_cudagraph_dispatch.
        eager = bool(getattr(vllm_config.model_config, "enforce_eager", False))
        # Under torch.compile (not eager) the FA backend allocates a PERSISTENT AOT
        # scheduler_metadata buffer sized for the engine's max batch; VFD's n=R*K candidate
        # forward overflows it (flash_attn.py build() line ~507). fast_build=True disables AOT
        # scheduling for the candidate metadata (the spec-decode path FA provides for exactly
        # this few-iteration dynamic shape), sidestepping that buffer. Eager has no such buffer,
        # so keep AOT there (preserves the validated eager token stream).
        self._candidate_fast_build = not eager
        # Whether the model is torch.compiled in place (enforce_eager=False). The compiled forward
        # reads input_ids/positions from the runner's persistent buffers, so the candidate forward
        # must populate them (see _candidate_forward_impl); eager reads the passed args directly.
        self._model_compiled = not eager
        # COMPILED BATCHED decode (enforce_eager=False, max_num_seqs>1, R>1 concurrent) is SUPPORTED.
        # Two fixes made it correct (2026-08, vLLM 0.19.1, Mistral-7B, h100; validated EOS 13/16 ==
        # eager, vs a broken 1/16): (1) fast_build=True for the candidate FA metadata build
        # (self._candidate_fast_build) sidesteps FA3's persistent AOT scheduler_metadata buffer, which
        # the n=R*K candidate batch otherwise overflows (flash_attn.py:507 crash); (2) the candidate
        # forward populates the runner's persistent input_ids/positions buffers on the eager-fallback
        # NONE path (see _candidate_forward_impl) -- the in-place-compiled model reads those buffers,
        # not the passed args, so without this reqs 1+ decoded against stale buffer state. The
        # candidate forward is cudagraph-captured at any R whose n=R*K fits a captured graph (see
        # _vfd_cudagraph_dispatch), falling back to the inductor-compiled eager path above the ceiling.
        # single_stream now only sizes the scratch reserve (K vs K*max_num_seqs).

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


        # CUDA-graph capture of the per-layer KV surgery (copy + commit). Those loops are
        # dominated by kernel-launch overhead (data moved is tiny); replaying the L-layer loop
        # as ONE captured graph collapses the per-layer launches to 1.
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
        # Deferred commits (block-boundary fix): when a winner's new token is the FIRST token of a
        # fresh KV block (offset 0) whose real block the scheduler has NOT allocated yet (block-table
        # entry still 0/padding), committing to it would write into physical block 0 and corrupt
        # position 0. Instead we RETAIN the winner's new-token K/V (per layer) here and flush it into
        # the real block on the next step, once the scheduler has allocated it (it always is by then).
        # {req_id: (tail_idx, offset, [per-layer K/V slot tensor])}
        self._deferred: dict[str, tuple] = {}
        # [B]-reentry single-writer fix: on a not-steady step VFD calls super().execute_model() to
        # prefill the newcomers, but that base forward ALSO forwards every ongoing decoder at
        # position num_computed -- a slot whose token VFD has NOT emitted yet -- and writes GARBAGE
        # K/V there (base samples from a stale input row). On an ordinary position VFD's candidate
        # commit overwrites it; at a block boundary VFD DEFERS, so base's garbage survives into the
        # prefix (the R>1 batched/compiled corruption: request 0 fine, joiners corrupt). Fix: before
        # that super() call, record the ongoing decoders here; the _prepare_inputs override sinks
        # their decode-row slot_mapping to PAD_SLOT_ID(-1) so base SKIPS the K/V write for them.
        # VFD's candidate forward then becomes the SOLE writer of every decoder's K/V.
        self._sink_slots_for: set[str] = set()

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
    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        # Delegate to vLLM (populates blk_table.slot_mapping.gpu via compute_slot_mapping), then
        # sink the ongoing decoders' decode-row slots to PAD_SLOT_ID(-1). _get_slot_mappings reads
        # that same buffer as a VIEW *after* this returns (gpu_model_runner 0.19.1: _prepare_inputs
        # ~L3858 precedes _get_slot_mappings ~L3958), and reshape_and_cache_flash skips slot<0 (the
        # padding contract, backends/utils.PAD_SLOT_ID=-1). So base's forward still runs for these
        # rows but WRITES NO K/V -- leaving VFD's candidate forward the sole writer. Only fires on a
        # not-steady step (execute_model sets _sink_slots_for right before super().execute_model()).
        out = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        if self.vfd_enabled and self._sink_slots_for:
            offs = np.concatenate([[0], np.cumsum(np.asarray(num_scheduled_tokens))])
            req_ids = self.input_batch.req_ids
            rows = [i for i, r in enumerate(req_ids) if r in self._sink_slots_for]
            if rows:
                for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups):
                    sm = self.input_batch.block_table[gid].slot_mapping.gpu
                    for i in rows:
                        sm[int(offs[i]):int(offs[i + 1])] = -1     # PAD_SLOT_ID -> no K/V write
            self._sink_slots_for = set()
        return out

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
                #
                # SINGLE-WRITER: the requests ALREADY decoding (in _pending_tok) must not have their
                # K/V written by this base forward -- it would forward them at position num_computed
                # (a slot VFD hasn't emitted) and write garbage. Mark them so the _prepare_inputs
                # override (invoked inside super().execute_model()) sinks their slots to -1. The
                # newcomers (prefilling P) are NOT in _pending_tok yet, so base still writes their
                # prompt K/V. VFD's candidate forward below is then the sole writer for the decoders.
                self._sink_slots_for = {r for r in scheduled if r in self._pending_tok}
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
                        if r in self._next_pos:
                            continue
                        idx = self.input_batch.req_id_to_index[r]
                        self._next_pos[r] = int(self.input_batch.num_prompt_tokens[idx])
                    self.execute_model_state = None              # consumed; skip base sampling
                    self._flush_deferred(active)                 # block-boundary deferred K/V
                    h_cand, scratch_idx, plan = self._candidate_forward(active)
                    winners = self._select(active, h_cand, scratch_idx, plan)
                    return self._build_output(active, winners)
                # Genuinely mixed (some still prefilling): let base emit this step and drop
                # any partial seeds so we only take over once the whole batch is decodable.
                mixed_drop = [x for x in self._pending_tok if x in active]
                for r in mixed_drop:
                    self._pending_tok.pop(r, None)
                    self._pending_logp.pop(r, None)
                    # KEEP _next_pos: this return-None path leaves execute_model_state intact, so the
                    # engine samples ONE base token for each ongoing decoder this step, advancing its
                    # true length by one. Popping _next_pos here made the next bootstrap step re-anchor
                    # the decoder to PROMPT length (position reset -> KV corruption at boundary-adjacent
                    # positions -- the R>1 batched residual). Mirror the base emit with +1 instead.
                    if r in self._next_pos:
                        self._next_pos[r] += 1
                return None                                      # engine calls sample_tokens

            # Steady-state decode: the candidate forward IS this step's forward. VFD bypasses
            # the base input-prep, so refresh the persistent batch (req set, block tables,
            # num_computed_tokens) and push the block table to device ourselves.
            self._update_states(scheduler_output)
            self.input_batch.block_table.commit_block_table(self.input_batch.num_reqs)
            active = self._active_req_ids()
            self._reject_logprobs(active)
            self._flush_deferred(active)                                  # block-boundary deferred K/V
            h_cand, scratch_idx, plan = self._candidate_forward(active)   # [R,K,H] + bookkeeping
            winners = self._select(active, h_cand, scratch_idx, plan)     # score+commit+reseed
            out = self._build_output(active, winners)
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
            # Use the request's OWN generator (as vLLM's sampler does), not the global torch RNG:
            # a bare torch.multinomial advances shared global state, so with R>1 each request's
            # candidates depend on the other requests in the batch (and on their order) -- a
            # cross-request coupling absent from base decode. The per-request generator isolates them.
            gen = getattr(self.requests[req_id], "generator", None)
            toks = torch.multinomial(probs, self.K, replacement=True, generator=gen)   # [K]
            self._pending_tok[req_id] = toks
            self._pending_logp[req_id] = torch.log(probs[toks].clamp_min(1e-12))

    def _warp(self, logits: torch.Tensor, req_id: str) -> torch.Tensor:
        """Apply the request's temperature + top_k/top_p so candidates are drawn from EXACTLY
        vLLM's sampling distribution. We call vLLM's own apply_top_k_top_p rather than the pure
        warp_logits helper: warp_logits' nucleus boundary convention (keep-crossing-token) differs
        from vLLM's (remove-from-bottom cumsum<=1-p) by one token, so it drew from a slightly
        BROADER nucleus -> off-vLLM-nucleus tokens -> long-run degeneration (newline/word loops).
        Grounded against pinned vLLM 0.19.1: apply_top_k_top_p(logits[B,V], k|None, p|None)."""
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        sp = self.requests[req_id].sampling_params
        temperature = float(getattr(sp, "temperature", 1.0) or 1.0)
        out = logits / max(temperature, 1e-5)
        top_p = float(getattr(sp, "top_p", 1.0) or 1.0)
        top_k = int(getattr(sp, "top_k", 0) or 0)
        k = None if top_k <= 0 else torch.tensor([top_k], device=out.device)
        p = None if top_p >= 1.0 else torch.tensor([top_p], device=out.device, dtype=out.dtype)
        if k is None and p is None:
            return out
        return apply_top_k_top_p(out.unsqueeze(0).clone(), k, p).squeeze(0)

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
        # Cudagraph-capture the candidate forward whenever a captured PIECEWISE graph fits the n=R*K
        # rows -- at R==1 AND R>1. VALIDATED correct + faster at R=2/4/8/16 (greedy texts coherent per
        # request, no cross-request bleed). The dispatcher rounds n up to a
        # captured size and returns NONE when none fits (n > max_cudagraph_capture_size = min(mns*2,
        # 512)), so large batches fall back to the eager path automatically. Capturing the R>1 graph
        # requires max_num_seqs >= n=R*K (vLLM captures an N-row uniform-decode graph only when
        # max_num_seqs >= N); size max_num_seqs to the concurrency you want captured (scratch = K*mns).
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
        return self._candidate_forward_impl(active_req_ids)

    def _candidate_forward_impl(self, active_req_ids: list[str]):
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

        cm, scratch_idx, plan = self._build_candidate_metadata(
            active_req_ids, seq_lens_per_req, n_pad
        )
        try:
            # Make each candidate's scratch block a copy of the request's real tail
            # block, so attention reads the real prefix tail; the forward then writes
            # the candidate's new-token KV at offset p % block_size. (Pad rows are not in
            # `plan` -- they read/write the dedicated pad sink and are sliced off below.)
            self._copy_real_tail_to_scratch(plan)
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
            if self._model_compiled:
                # LOAD-BEARING for compiled decode: the model is compiled IN PLACE
                # (self.model.compile() in gpu_model_runner), so it reads input_ids/positions from
                # these PERSISTENT buffers, NOT from the tensors passed to _model_forward -- on BOTH
                # the PIECEWISE-replay path (the graph reads the fixed capture addresses; cuda_graph.py
                # does NOT copy inputs on replay) AND the eager-fallback NONE path (the in-place
                # compiled forward still reads the buffers). So write the candidate inputs into
                # self.input_ids/self.positions and pass the persistent slices. Passing fresh args on
                # the NONE path left requests after the first decoding against STALE buffer state --
                # the R>1 compiled-batched corruption (req0 correct, reqs 1+ garbled). Validated:
                # this fix gives EOS 13/16 == eager, vs 1/16 broken. Pad rows [n:n_pad] get token/pos 0
                # (discarded; attention reads the pad sink). _update_states does not touch these
                # buffers, so VFD owns them for this step.
                self.input_ids.gpu[:n].copy_(toks)         # int32 buffer <- long ids (cast)
                self.input_ids.gpu[n:n_pad].zero_()
                self.positions[:n].copy_(positions)
                self.positions[n:n_pad].zero_()
                in_ids = self.input_ids.gpu[:n_pad]
                in_pos = self.positions[:n_pad]
            else:
                # enforce_eager: the model reads the passed args directly (no compiled buffer read).
                in_ids, in_pos = toks, positions
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=n_pad,
                                     cudagraph_runtime_mode=cg_mode,
                                     batch_descriptor=batch_desc,
                                     slot_mapping=slot_mapping_by_layer):
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
            # Per-REQUEST (winner-independent: all K candidates of a request share the tail block)
            # CPU copies so the block-boundary deferral check in _commit_winner_kv stays SYNC-FREE
            # (no per-step D2H). real_tail_blk/offset are identical across a request's K rows -> [::K].
            "rtb_cpu": list(layout["real_tail_blk"][::K]),
            "off_cpu": list(layout["offset"][::K]),
        }
        return cm, scratch_idx, plan

    def _build_per_layer_metadata(self, cm):
        """Run the attention-metadata builder for each attention group and map the
        result onto every layer in that group. Single-KV-group / single-attn-group is
        the only case VFD allows (initialize_kv_cache raises for multi-group models)."""
        attn_metadata: dict = {}
        for kv_gid, groups in enumerate(self.attn_groups):
            for group in groups:
                builder = group.get_metadata_builder()
                md = builder.build(common_prefix_len=0, common_attn_metadata=cm,
                                   fast_build=self._candidate_fast_build)
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
        copying key_cache and value_cache separately (the KV surgery is launch-bound)."""
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

    def _kv_tensors(self):
        """Yield each layer's validated FlashAttention KV tensor, in layer order. The single
        prelude every per-layer KV-surgery loop shares (copy/commit/flush); sync-free, so it is
        safe inside a captured `run()` closure."""
        for kv_cache in self.kv_caches:
            yield self._kv_validate(kv_cache)

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
                for kv in self._kv_tensors():
                    kv[:, d_] = kv[:, s_]
            self._run_captured(("copy", nrows), run)
            return
        # Eager path (capture off or batch too large): mask to the rows that need a copy.
        mask = plan["needs_copy"]
        if not bool(mask.any()):
            return
        src = plan["real_tail_blk"][mask]        # [m] real tail blocks
        dst = plan["scratch_blk"][mask]          # [m] scratch blocks
        for kv in self._kv_tensors():
            kv[:, dst] = kv[:, src]              # whole-block copy, key+value in one op

    # ============================================================== #
    # Seam 3 (concrete): first-safe / safest selection, commit, reseed#
    # ============================================================== #
    @torch.inference_mode()
    def _select(self, active_req_ids: list[str], h_cand: torch.Tensor,
                scratch_idx, plan) -> dict[str, int]:
        R, K, H = h_cand.shape
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
        # head was scored on for each committed token, so decode-matched training features can be
        # captured (scripts/decode_extract.py) -- the tensor the head is scored on at inference.
        if self._dump_hidden is not None:
            for i, r in enumerate(active_req_ids):
                self._dump_hidden.setdefault(r, []).append(
                    (int(wtok[i]), wh[i].detach().to(torch.float32).cpu()))

        # Commit the winner's KV into each request's real tail slot (pure in-cache copy,
        # NO second forward). Load-bearing for the single-forward scheme: without it the
        # next candidate forward would see a prefix missing this token's KV.
        self._commit_winner_kv(active_req_ids, winner_col, plan)
        self._scratch.free(scratch_idx)        # candidate KV consumed; release the blocks

        for r in active_req_ids:               # this step filled _next_pos[r]; advance by one
            self._next_pos[r] += 1
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
        # BLOCK-BOUNDARY DEFERRAL: a winner whose real tail block is still 0 (padding) is the first
        # token of a fresh block the scheduler allocates one step LATE (measured on Mistral-7B: the
        # block is absent at pos 16/32/48... and present the next step). Committing now would copy
        # into physical block 0 -> corrupts position 0 and loses the new token's K/V -> cache drift
        # -> degeneration. Retain the winner's K/V and flush it once the block exists (next step).
        # SYNC-FREE boundary detection. real_tail_blk is per-REQUEST and winner-independent (all K
        # candidates of a request share the tail block), and it was built from the CPU block-table
        # mirror -- so read it from CPU (plan["rtb_cpu"]) instead of a per-step bool(GPU.any()) D2H.
        # (The old GPU .any() ran every commit and serialized the CPU<->GPU pipeline -> ~3-5x slower.)
        rtb_cpu = plan["rtb_cpu"]                   # per-request real tail block, on CPU
        unalloc = [b == 0 for b in rtb_cpu]         # pure CPU -> NO per-step sync
        if any(unalloc):                            # rare (block boundaries only)
            keep = [not u for u in unalloc]
            sb, of = scratch_blk.tolist(), plan["off_cpu"]   # sb: winner scratch (rare sync); of: CPU
            for i in range(R):
                if not keep[i]:
                    r = active_req_ids[i]
                    tail_idx = self._next_pos[r] // self.cache_config.block_size
                    self._deferred[r] = (tail_idx, of[i],
                                         [kv[:, sb[i], of[i]].clone() for kv in self._kv_tensors()])
            sel = torch.tensor([i for i in range(R) if keep[i]], device=device, dtype=torch.long)
            scratch_blk, real_blk, offset = scratch_blk[sel], real_blk[sel], offset[sel]
            rtb_cpu = [rtb_cpu[i] for i in range(len(keep)) if keep[i]]
            R = int(sel.numel())
            if R == 0:
                return
        # VALIDATED on A100: the winner's real target block IS allocated the step it's generated
        # (same-step commit; this guard never fired). Kept as an invariant guard -- now a pure-CPU
        # check (rtb_cpu) so it costs no sync; surfaces loudly if a scheduler ever allocates late.
        base = self._scratch_blocks[0] if self._scratch_blocks else None
        if base is not None and any(b >= base for b in rtb_cpu):
            raise RuntimeError(
                "VFD commit target points into the scratch range -- winner's real block "
                "was not allocated by the scheduler this step (commit-timing: deferral "
                f"needed). real_blk={rtb_cpu} scratch_base={base}"
            )
        if self._capture_kv and R <= self._kv_capture_max_rows:
            self._cm_real[:R].copy_(real_blk)
            self._cm_scratch[:R].copy_(scratch_blk)
            self._cm_off[:R].copy_(offset)

            def run():
                r_, s_, o_ = self._cm_real[:R], self._cm_scratch[:R], self._cm_off[:R]
                for kv in self._kv_tensors():
                    kv[:, r_, o_] = kv[:, s_, o_]
            self._run_captured(("commit", R), run)
            return
        for kv in self._kv_tensors():
            kv[:, real_blk, offset] = kv[:, scratch_blk, offset]   # key+value in one op

    @torch.inference_mode()
    def _flush_deferred(self, active_req_ids) -> None:
        """Write any block-boundary-deferred winner K/V into its real block, now that the scheduler
        has allocated it. Called at the start of a step BEFORE _copy_real_tail_to_scratch, so the
        block holds the boundary token when the next candidate forward copies the tail block."""
        if not self._deferred:
            return
        active = set(active_req_ids)
        bt = self.input_batch.block_table[0].get_cpu_tensor()
        for r in list(self._deferred):
            if r not in active:
                self._deferred.pop(r, None)
                continue
            tail_idx, off, bufs = self._deferred[r]
            ridx = self.input_batch.req_id_to_index[r]
            blk = int(bt[ridx, tail_idx]) if tail_idx < bt.shape[1] else 0
            if blk == 0:                       # still not allocated -- keep waiting (shouldn't happen)
                continue
            for kv, buf in zip(self._kv_tensors(), bufs):
                kv[:, blk, off] = buf
            self._deferred.pop(r, None)

    def _update_states(self, scheduler_output):
        """After vLLM refreshes input_batch (block tables now reflect any block the scheduler just
        allocated), flush block-boundary-deferred winner K/V. This runs INSIDE super().execute_model
        on mixed/bootstrap steps too, so the deferred token's K/V is in the real cache BEFORE the base
        forward reads it -- without this, a mixed step (a new request joining a batch mid-ramp) reads
        the still-missing boundary position and corrupts the cache (the R>1 batched residual)."""
        ret = super()._update_states(scheduler_output)
        if self.vfd_enabled and self._deferred:
            self._flush_deferred(self._active_req_ids())
        return ret

    # ----------------------- lifecycle / output -------------------- #
    def _active_req_ids(self) -> list[str]:
        ids = self.input_batch.req_ids
        return [ids[i] for i in range(self.input_batch.num_reqs) if ids[i] is not None]

    def _drop_finished(self, scheduler_output) -> None:
        """Clear candidate state for requests that finished/aborted this step, so
        _pending_tok / _pending_logp don't leak across the run."""
        finished = set(getattr(scheduler_output, "finished_req_ids", None) or [])
        live = set(self._active_req_ids())
        for d in (self._pending_tok, self._pending_logp, self._next_pos, self._deferred):
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
        # Only seed requests that are NOT already seeded. An ONGOING decoder that hits this
        # not-steady path mid-generation (because a NEW request joined and prefilled in the same
        # step) still holds the correct candidates seeded from its own last _select. Re-seeding it
        # here from sample_hidden_states is (a) redundant and (b) WRONG: with a prefill sharing the
        # batch the row<->request 1:1 mapping no longer holds, so h[i] can be a different request's
        # (or an off-by-one) hidden -> the ongoing decoder re-emits its previous token (the R>1
        # stutter). Skip already-pending requests; only seed the genuinely new ones.
        hidden_by_req = {r: h[i] for i, r in enumerate(ids)
                         if r in req_ids and r not in self._pending_tok}
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
