# SPDX-License-Identifier: Apache-2.0
"""
--worker-cls entry point: select the value-steering model runner from config.

Launch (one of):
  vllm serve <model> --worker-cls value_steer.worker.ValueSteerWorker \
      --additional-config '{"abstain": {"value_head_path": "...", "threshold": 0.5}}'
  vllm serve <model> --worker-cls value_steer.worker.ValueSteerWorker \
      --additional-config '{"vfd": {"value_head_path": "...", "threshold": 0.5}}' \
      # (no --speculative_config; VFD owns the decode forward)

The base Worker builds self.model_runner inside init_device(); we let it, then
replace the instance with our subclass when abstain/vfd is enabled. Both runners
subclass GPUModelRunner, so the swap is transparent to the rest of the worker. This
uses the supported ParallelConfig.worker_cls surface -- no monkeypatching of vLLM
internals, which is what keeps it robust across version bumps (the compat harness
asserts the few internals the runners do touch).

Rebuilding the runner discards the base instance built by super().init_device();
that is cheap because the model is not loaded until the separate load_model() call.
"""

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker


def _wants(cfg: dict, key: str) -> bool:
    # Enabled if the section is present and not explicitly disabled.
    section = cfg.get(key)
    if section is None:
        return False
    return bool(section.get("enabled", True)) if isinstance(section, dict) else bool(section)


class ValueSteerWorker(Worker):
    def init_device(self):
        super().init_device()
        cfg = self.vllm_config.additional_config or {}
        want_abstain = _wants(cfg, "abstain")
        want_vfd = _wants(cfg, "vfd")
        if want_abstain and want_vfd:
            raise ValueError(
                "Enable only one of {abstain, vfd} -- they are separate decode modes."
            )
        if want_abstain or want_vfd:
            from .versions import warn_if_unvalidated
            warn_if_unvalidated()   # serve-time RuntimeWarning if vLLM is untested
        if want_abstain:
            from .abstention_model_runner import AbstentionModelRunner
            self.model_runner = AbstentionModelRunner(self.vllm_config, self.device)
        elif want_vfd:
            from .vfd_model_runner import VFDModelRunner
            self.model_runner = VFDModelRunner(self.vllm_config, self.device)
        # else: leave the stock runner in place (no steering).
