# SPDX-License-Identifier: Apache-2.0
"""
Shared value-probe infrastructure for both papers.

One scalar value head + one feature contract, imported by:
  * abstention_model_runner.AbstentionModelRunner (Paper 1) -- gate at sampling site
  * vfd_model_runner.VFDModelRunner               (Paper 2) -- filter K candidates

Architecture and contract are taken from the VFD paper repo (DenseValueModel),
the only one of the two whose training code is verified:
  * feature  = model.model(...).last_hidden_state -- final decoder layer,
               POST-final-RMSNorm, per token, the SAME tensor lm_head consumes;
  * the head runs in fp32 with the hidden state cast to fp32 (backbone is bf16);
  * head     = Linear(H,H) -> Tanh -> Linear(H,H) -> ReLU -> Linear(H,1), raw logit;
  * probability = sigmoid(logit).

The head is LABEL-AGNOSTIC at inference: the runtime sees only a calibrated probe.
What the probability means -- P(should quit) for abstention vs P(unsafe) for VFD --
lives entirely in the training labels and in the caller's threshold logic, not here.
So both papers share this module; only their callers differ.

NOTE: the abstention checkpoint must be trained with (or re-loaded into) THIS head;
its training code was not available to verify, so unifying the code assumes a
matching architecture. If Paper 1 used a different head, retrain with this one or
keep a per-caller override.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FeatureSpec:
    """The hidden state the head was trained on -- the single source of truth for
    the contract. Both runners are written to extract exactly this (final post-norm
    hidden); this records it so a future change has one obvious place to update and
    check against. (Documentation, not a runtime assertion.)"""
    layer: str = "final"     # final decoder layer
    norm: str = "post"       # AFTER the final RMSNorm (== lm_head input)
    pooling: str = "none"    # per-token, no pooling
    dtype: str = "fp32"      # head input cast to fp32


DEFAULT_SPEC = FeatureSpec()


class ValueHead(nn.Module):
    """Exact replica of the VFD repo's DenseValueModel.value_head. fp32, raw logit."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(torch.float32)

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        # h: [..., H] -> [...]. Cast to fp32 to match training. No inference_mode
        # decorator here so the same head is reusable for training the probe; the
        # inference callers (runners) already run under torch.inference_mode().
        return self.net(h.to(torch.float32)).squeeze(-1)

    def p(self, h: torch.Tensor) -> torch.Tensor:
        # Calibrated probability in (0, 1). Caller decides the direction of the
        # threshold comparison (quit-if-low vs intervene-if-high).
        return torch.sigmoid(self.logit(h))


def load_value_head(path: str, hidden_size: int, device) -> ValueHead:
    """Build the shared head and load a checkpoint into it.

    `path` must hold the state dict of the `value_head` SUBMODULE (Sequential keys
    "0.weight", "0.bias", "2.weight", ...) -- i.e. torch.save(model.value_head.
    state_dict()), not the whole DenseValueModel. load_state_dict is strict, so an
    architecture mismatch raises here rather than silently mis-scoring."""
    head = ValueHead(hidden_size).to(device)
    head.net.load_state_dict(torch.load(path, map_location=device))
    head.eval()
    return head


def request_threshold(sampling_params, key: str, default: float) -> float:
    """Per-request threshold from SamplingParams.extra_args[key], else default.
    sampling_params may be None (CachedRequestState.sampling_params is | None)."""
    ea = getattr(sampling_params, "extra_args", None) or {}
    return float(ea.get(key, default))
