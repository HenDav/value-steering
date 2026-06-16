# SPDX-License-Identifier: Apache-2.0
"""
Version-compatibility harness for the value-steering plugin (abstention + VFD).

WHY THIS IS NOT "does it run"
-----------------------------
The runner hooks deliberately swallow exceptions and no-op on contract mismatch so
production never crashes (try/except in sample_tokens; the num_rows!=num_reqs guard).
That means a check that only asserts "output was produced" goes GREEN on a vLLM
version where the feature has silently DIED. So this harness does two things:

  1. STATIC contract checks -- assert the exact internal surfaces the thin runner
     adapters bind to still exist with the expected shapes/signatures. Needs only
     `import vllm` (no GPU). Grounded against the pinned tree below.
  2. BEHAVIORAL checks -- run the runners in STRICT mode (re-raise, don't swallow)
     and assert the feature actually FIRED: abstention forces EOS when V̂ < c; VFD
     shifts the next-token distribution vs the base model. Needs a GPU + small model.

Run both per vLLM version in CI. If a bump renames a field or shifts a semantic, a
static check fails loudly here instead of the feature going quietly inert in prod.

PINNED: contracts below are verified against vLLM 0.19.1. When bumping, update the
expected signatures here in lockstep -- this file IS the compatibility spec.
"""

from __future__ import annotations

import inspect

PINNED = "0.19.1"


class ContractError(AssertionError):
    pass


# ======================================================================= #
# STATIC CONTRACT CHECKS  (import vllm only; no GPU)                       #
# Each asserts one surface the adapters in {abstention,vfd}_model_runner   #
# and value_probe depend on. Keep this list == the adapters' attack surface#
# ======================================================================= #

def check_execute_model_state():
    """ExecuteModelState must carry logits + sample_hidden_states (the abstention
    feature) and be a NamedTuple (so ._fields introspection works)."""
    from vllm.v1.worker.gpu_model_runner import ExecuteModelState
    if not hasattr(ExecuteModelState, "_fields"):
        raise ContractError("ExecuteModelState is no longer a NamedTuple")
    need = {"logits", "sample_hidden_states", "hidden_states", "scheduler_output"}
    missing = need - set(ExecuteModelState._fields)
    if missing:
        raise ContractError(f"ExecuteModelState missing fields: {missing}")


def check_runner_seams():
    """The two override points + the state attribute + the candidate-forward seam."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    for m in ("sample_tokens", "execute_model", "_model_forward"):
        if not callable(getattr(GPUModelRunner, m, None)):
            raise ContractError(f"GPUModelRunner.{m} missing/not callable")
    # sample_tokens(self, grammar_output): the in-place-logits hook depends on this
    params = list(inspect.signature(GPUModelRunner.sample_tokens).parameters)
    if params[:2] != ["self", "grammar_output"]:
        raise ContractError(f"sample_tokens signature changed: {params}")
    # execute_model_state attribute is set in __init__ to None
    src = inspect.getsource(GPUModelRunner.__init__)
    if "execute_model_state" not in src:
        raise ContractError("GPUModelRunner.__init__ no longer sets execute_model_state")


def check_input_batch_and_request():
    """Row<->request mapping + per-row prefill-eligibility signals that abstention
    uses: num_reqs/req_ids, CachedRequestState.sampling_params, the input-batch
    prompt/computed arrays, and scheduler_output.num_scheduled_tokens."""
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
    ann = getattr(CachedRequestState, "__annotations__", {})
    if "sampling_params" not in ann:
        raise ContractError("CachedRequestState.sampling_params missing")
    src = inspect.getsource(InputBatch)
    for attr in ("num_reqs", "req_ids", "num_computed_tokens_cpu", "num_prompt_tokens"):
        if attr not in src:
            raise ContractError(f"InputBatch.{attr} not found")
    from vllm.v1.core.sched.output import SchedulerOutput
    if "num_scheduled_tokens" not in getattr(SchedulerOutput, "__annotations__", {}):
        raise ContractError("SchedulerOutput.num_scheduled_tokens missing")


def check_config_surfaces():
    """additional_config (our config channel), get_hidden_size/get_vocab_size,
    SamplingParams.extra_args (per-request thresholds)."""
    from vllm.config import VllmConfig
    from vllm.config.model import ModelConfig
    from vllm.sampling_params import SamplingParams
    if "additional_config" not in getattr(VllmConfig, "__annotations__", {}) \
            and "additional_config" not in inspect.getsource(VllmConfig):
        raise ContractError("VllmConfig.additional_config missing")
    for m in ("get_hidden_size", "get_vocab_size"):
        if not callable(getattr(ModelConfig, m, None)):
            raise ContractError(f"ModelConfig.{m} missing")
    if "extra_args" not in inspect.getsource(SamplingParams):
        raise ContractError("SamplingParams.extra_args missing")


def check_forward_context():
    """set_forward_context import + the slot_mapping plumbing the VFD candidate forward
    depends on. The KV-WRITE op reads the new-token slot from forward_context.slot_mapping
    (a per-layer dict), NOT from attn_metadata -- so set_forward_context MUST accept a
    slot_mapping kwarg and ForwardContext MUST carry it. If this drifts, VFD writes candidate
    KV to a stale slot and the hidden silently diverges from base (a bug behavioral caught
    only via the greedy-equivalence test; assert it statically here)."""
    from vllm.forward_context import ForwardContext, set_forward_context
    params = inspect.signature(set_forward_context).parameters
    for need in ("attn_metadata", "vllm_config", "num_tokens", "slot_mapping"):
        if need not in params:
            raise ContractError(
                f"set_forward_context missing param {need!r}: {list(params)}")
    if "slot_mapping" not in getattr(ForwardContext, "__annotations__", {}):
        raise ContractError("ForwardContext no longer carries slot_mapping")


def check_kv_write_slot_mapping_source():
    """The attention KV-write op (unified_kv_cache_update) takes its destination slot from
    forward_context.slot_mapping. VFD relies on this to route candidate KV to its scratch
    slot via set_forward_context(slot_mapping=...)."""
    import vllm.model_executor.layers.attention.attention as attn_mod
    src = inspect.getsource(attn_mod)
    if "unified_kv_cache_update" not in src or "forward_context.slot_mapping" not in src:
        raise ContractError(
            "attention KV-write no longer reads forward_context.slot_mapping")


def check_model_forward_signature():
    """VFD's candidate forward calls _model_forward(input_ids=..., positions=...)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    params = list(inspect.signature(GPUModelRunner._model_forward).parameters)
    if params[:3] != ["self", "input_ids", "positions"]:
        raise ContractError(f"_model_forward signature changed: {params}")


def check_block_table_api():
    """VFD reads input_batch.block_table[0].get_device_tensor(num_reqs) and pushes the
    device block table itself via commit_block_table(num_reqs)."""
    from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
    if not callable(getattr(MultiGroupBlockTable, "__getitem__", None)):
        raise ContractError("MultiGroupBlockTable.__getitem__ missing (group access)")
    for m in ("get_device_tensor", "commit_block_table"):
        if not callable(getattr(BlockTable, m, None)):
            raise ContractError(f"BlockTable.{m} missing")


def check_runner_input_prep_and_binding():
    """VFD's steady step refreshes batch state via _update_states; the scratch reserve
    overrides initialize_kv_cache; the in-cache commit/copy index self.kv_caches, which
    bind_kv_cache aliases to the attention layers' kv_cache (so the surgery is visible to
    attention)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    for m in ("_update_states", "initialize_kv_cache"):
        if not callable(getattr(GPUModelRunner, m, None)):
            raise ContractError(f"GPUModelRunner.{m} missing")
    from vllm.v1.worker.utils import bind_kv_cache
    bparams = list(inspect.signature(bind_kv_cache).parameters)
    if "runner_kv_caches" not in bparams or "forward_context" not in bparams:
        raise ContractError(f"bind_kv_cache signature changed: {bparams}")


def check_model_runner_output_fields():
    """VFD._build_output constructs ModelRunnerOutput(req_ids, req_id_to_index,
    sampled_token_ids, ...)."""
    from vllm.v1.outputs import ModelRunnerOutput
    ann = getattr(ModelRunnerOutput, "__annotations__", {})
    missing = {"req_ids", "req_id_to_index", "sampled_token_ids"} - set(ann)
    if missing:
        raise ContractError(f"ModelRunnerOutput missing fields: {missing}")


def check_common_attention_metadata_fields():
    """VFD._build_candidate_metadata constructs CommonAttentionMetadata with these fields."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    need = {"query_start_loc", "query_start_loc_cpu", "seq_lens", "num_reqs",
            "num_actual_tokens", "max_query_len", "max_seq_len", "block_table_tensor",
            "slot_mapping", "causal"}
    missing = need - set(getattr(CommonAttentionMetadata, "__annotations__", {}))
    if missing:
        raise ContractError(f"CommonAttentionMetadata missing fields: {missing}")


def check_reshape_and_cache_flash():
    """Winner-KV commit (VFD) -- exact parameter order matters."""
    from vllm import _custom_ops as ops
    got = list(inspect.signature(ops.reshape_and_cache_flash).parameters)
    need = ["key", "value", "key_cache", "value_cache",
            "slot_mapping", "kv_cache_dtype", "k_scale", "v_scale"]
    if got[:len(need)] != need:
        raise ContractError(f"reshape_and_cache_flash signature changed: {got}")


def check_kv_cache_layout():
    """FlashAttention cache shape (2, num_blocks, block_size, num_kv_heads, head_size)
    + stride-order accessor (VFD slice surgery, if used, must honor the layout)."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    shape = FlashAttentionBackend.get_kv_cache_shape(
        num_blocks=8, block_size=16, num_kv_heads=4, head_size=64
    )
    if len(shape) != 5 or shape[0] != 2 or shape[1:] != (8, 16, 4, 64):
        raise ContractError(f"FlashAttention kv cache shape changed: {shape}")
    if not callable(getattr(FlashAttentionBackend, "get_kv_cache_stride_order", None)):
        raise ContractError("FlashAttentionBackend.get_kv_cache_stride_order missing")


# Abstention needs only the first four; VFD needs all seven.
ABSTENTION_CONTRACTS = [
    check_execute_model_state, check_runner_seams,
    check_input_batch_and_request, check_config_surfaces,
]
VFD_EXTRA_CONTRACTS = [
    check_forward_context, check_reshape_and_cache_flash, check_kv_cache_layout,
    # Deeper runner attack surface -- the internals the VFD candidate forward / commit
    # touch beyond the original three. These catch drift (esp. the slot_mapping plumbing)
    # statically, i.e. without a GPU, that previously only the behavioral suite would hit.
    check_kv_write_slot_mapping_source, check_model_forward_signature,
    check_block_table_api, check_runner_input_prep_and_binding,
    check_model_runner_output_fields, check_common_attention_metadata_fields,
]


def run_static(which: str = "all") -> bool:
    import vllm

    from .versions import is_validated, validated_versions
    v = getattr(vllm, "__version__", "?")
    status = "validated" if is_validated(v) else "NOT in validated set"
    print(f"vLLM {v} (pinned contract: {PINNED}) -- {status}")
    if not is_validated(v):
        print(f"  validated versions: {sorted(validated_versions())}")
        print("  contract checks below still run; unvalidated != broken, but untested.")
    checks = list(ABSTENTION_CONTRACTS)
    if which in ("all", "vfd"):
        checks += VFD_EXTRA_CONTRACTS
    ok = True
    for fn in checks:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            ok = False
            print(f"  FAIL  {fn.__name__}: {e!r}")
    return ok


# The GPU behavioral suite (run the runner in strict mode with a stub constant head and assert the
# feature actually fires) lives in tests/test_gpu_behavioral.py, not here -- this module is the
# static, no-GPU contract gate. See CONTRIBUTING.md for how the two fit together.


_USAGE = """\
value-steer-compat -- vLLM version-contract checks for value-steer

Usage:
  value-steer-compat [all|abstain|vfd]   run the static contract checks (needs `import vllm`)
  value-steer-compat record [note...]    run static checks and record pass/fail for the live vLLM
  value-steer-compat -h | --help         show this message

Static checks need only an importable vLLM; the GPU behavioral suite runs separately
(see CONTRIBUTING.md). Install the runtime with `pip install "value-steer[vllm]"`.
"""


def _cli():
    import sys
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(_USAGE)
        sys.exit(0)
    try:
        import vllm  # noqa: F401  -- fail clean if the runtime isn't installed
    except ModuleNotFoundError:
        print("vLLM is not installed. Install the runtime with: pip install \"value-steer[vllm]\"")
        sys.exit(2)
    if args and args[0] == "record":
        # value-steer-compat record [note...]  -> runs static, records pass/fail for the
        # live vLLM version. (Full validation also needs the GPU behavioral pass; the
        # agent calls versions.record_validation(..., behavioral=True) directly then.)
        from .versions import current_vllm_version, record_validation
        v = current_vllm_version()
        if v is None:
            print("vllm not importable; cannot record")
            sys.exit(1)
        green = run_static("all")
        rec = record_validation(
            v, "pass" if green else "fail", static=green, note=" ".join(args[1:])
        )
        print(f"recorded {v}: {rec}")
        sys.exit(0 if green else 1)
    which = args[0] if args else "all"
    sys.exit(0 if run_static(which) else 1)


if __name__ == "__main__":
    _cli()
