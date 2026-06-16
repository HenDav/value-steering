# SPDX-License-Identifier: Apache-2.0
"""
Dynamic-abstention for vLLM 0.19.1 (Paper 1), SAMPLING-SITE integration.

Supersedes abstention_proposer.py. The proposer route rode `extract_hidden_states`
and fed the head an aux (intermediate, pre-final-norm) hidden state, which does not
match the shared feature contract. This variant reads `sample_hidden_states` -- the
final-layer POST-norm hidden, the exact tensor lm_head consumes and the exact tensor
VFD scores -- so abstention and VFD now share BOTH the head (value_probe.ValueHead)
AND the feature (value_probe.FeatureSpec).

Mechanism
---------
In V1, execute_model() stashes ExecuteModelState carrying `logits` and
`sample_hidden_states`; sample_tokens() then samples from `logits`
(gpu_model_runner.py:4101 / 4123-4167). Abstention is a logits rewrite applied in
place before delegating:

    vhat = value_head.p(sample_hidden_states)     # [num_reqs], post-norm feature
    abstain = vhat < c                            # quit when value is low
    logits[abstain] = -inf; logits[abstain, EOS] = 0   # sampler emits EOS -> stop

No candidate forward (abstention only needs h_t, not h_{t+1}), no aux-layer plumbing,
no KV surgery. One forward per token, unchanged from the base runner.

Direction note: `.p()` is a calibrated probability whose meaning comes from the
abstention training labels. This assumes the probe predicts "keep going" (quit when
LOW). If your probe predicts P(should-stop), flip to `vhat > c`.

Config: vllm_config.additional_config["abstain"]; per-request threshold via
SamplingParams.extra_args["abstain_threshold"]. Requires speculative decoding OFF
(under spec, logits rows are draft+bonus positions, not 1:1 with requests, so the
hook no-ops). Chunked prefill IS handled: non-spec logits rows are 1:1 with requests
(gpu_model_runner.py:2047), and partial-prefill rows -- whose sampled tokens vLLM
discards -- are skipped per-row.

Status: SAMPLING-SITE SCAFFOLD. Fully implemented against the pinned APIs; remaining
validation (EOS actually fires end-to-end) is the GPU behavioral check in compat_check.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from .steering_ops import force_eos_rows

# Shared scalar head + feature contract (same module VFD uses).
from .value_probe import ValueHead, load_value_head, request_threshold


class AbstentionModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        cfg = (vllm_config.additional_config or {}).get("abstain", {})
        self.abstain_enabled: bool = bool(cfg.get("enabled", True))
        self.default_c: float = float(cfg.get("threshold", 0.5))
        self.strict: bool = bool(cfg.get("strict", False))   # CI: re-raise, don't swallow

        hidden = self.model_config.get_hidden_size()
        if (p := cfg.get("value_head_path")):
            self.value_head = load_value_head(p, hidden, device)
        else:
            self.value_head = ValueHead(hidden).to(device)
        self.value_head.eval()

        # EOS id(s): hf_config.eos_token_id may be int, list, or None.
        mc = self.model_config
        eos = getattr(mc.hf_config, "eos_token_id", None)
        if eos is None:
            eos = getattr(mc.hf_text_config, "eos_token_id", None)
        if eos is None:
            raise ValueError("Could not resolve eos_token_id for abstention.")
        self.eos_token_id: int = eos[0] if isinstance(eos, (list, tuple)) else int(eos)

    # ------------------------------------------------------------------ #
    # Hook: force EOS in logits where V̂ < c, then delegate.             #
    # ------------------------------------------------------------------ #
    def sample_tokens(self, grammar_output):
        st = self.execute_model_state
        if st is None or not self.abstain_enabled:
            return super().sample_tokens(grammar_output)
        try:
            self._apply_abstention(st)   # mutates st.logits in place
        except Exception:
            if self.strict:
                raise
            # Production: abstention must never crash decoding; use base logits.
        return super().sample_tokens(grammar_output)

    @torch.inference_mode()
    def _apply_abstention(self, st) -> None:
        logits = st.logits                    # [num_rows, vocab]
        h = st.sample_hidden_states           # [num_rows, H], post-norm == lm_head input

        num_reqs = self.input_batch.num_reqs
        num_rows = logits.shape[0]
        # Non-spec decode: logits_indices == query_start_loc[1:] - 1, i.e. exactly
        # one row per request in input_batch order (gpu_model_runner.py:2047). So
        # row i <-> request i, and num_rows == num_reqs even on MIXED prefill+decode
        # steps. The only case this breaks is speculative decoding (rows are draft+
        # bonus positions, not 1:1 with requests), which abstention does not support.
        if num_rows != num_reqs:
            if self.strict:
                raise AssertionError(
                    f"num_rows ({num_rows}) != num_reqs ({num_reqs}): abstention "
                    "requires speculative decoding OFF (its logits rows are not 1:1 "
                    "with requests)."
                )
            return

        vhat = self.value_head.p(h)           # [num_rows] in (0, 1)

        sched = st.scheduler_output.num_scheduled_tokens   # req_id -> count
        computed = self.input_batch.num_computed_tokens_cpu  # per req index
        prompt = self.input_batch.num_prompt_tokens          # per req index
        req_ids = self.input_batch.req_ids

        sps = [self.requests[req_ids[i]].sampling_params for i in range(num_reqs)]
        c = torch.tensor(
            [request_threshold(sp, "abstain_threshold", self.default_c) for sp in sps],
            device=logits.device,
            dtype=vhat.dtype,
        )
        abstain = vhat < c                    # quit when value is low (see direction note)

        # Keep only rows that emit a KEPT token this step and won't have EOS
        # suppressed downstream:
        #   * partial-prefill rows -- vLLM computes their logits "for simplicity"
        #     then DISCARDS the sampled token (gpu_model_runner.py:2041-2044). A row
        #     is generating iff its prompt is fully consumed by end of this step.
        #   * ignore_eos rows -- forcing -inf except EOS becomes an all -inf row once
        #     the EOS mask is applied (NaN). min_tokens has the same hazard until the
        #     minimum is met; left to the operator (abstention + min_tokens conflict).
        for i, sp in enumerate(sps):
            if not bool(abstain[i]):
                continue
            generating = computed[i] + sched.get(req_ids[i], 0) >= prompt[i]
            if (not generating) or getattr(sp, "ignore_eos", False):
                abstain[i] = False

        if not bool(abstain.any()):
            return

        force_eos_rows(logits, abstain, self.eos_token_id)  # in-place; see steering_ops


# ---------------------------------------------------------------------------
# Wiring: like VFD, point the worker's runner class at AbstentionModelRunner before
# the worker builds it (plugin / --worker-cls / patch). No method-string dispatch.
# Launch (spec decode off):
#   vllm serve <model> --additional-config '{"abstain": {
#       "enabled": true, "threshold": 0.5,
#       "value_head_path": "/path/to/value_head.pt"}}'
#
# Shares value_probe.ValueHead and the post-norm feature with vfd_model_runner.py.
# Both intervene at the same sampling site on the same feature with the same head;
# abstention gates to EOS, VFD filters K candidates.
