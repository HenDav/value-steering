# SPDX-License-Identifier: Apache-2.0
"""
Pure-tensor decode-policy ops for the value-steering plugin.

Zero vLLM dependency -- only torch. The runners (abstention, VFD) keep their vLLM
plumbing (reading hidden states, batch/request bookkeeping) but delegate the actual
DECISION logic here, so it is unit-testable on CPU with no GPU, no model, and no
vLLM install. Keep policy here; keep plumbing in the runners.
"""

from __future__ import annotations

import torch


def warp_logits(
    logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0
) -> torch.Tensor:
    """Temperature + nucleus (top-p) warping over the last dim; supports [V] or
    [..., V]. Matches what vLLM sampling applies so VFD candidates are drawn from
    the same distribution. Does NOT cover top_k / min_p / penalties -- for full
    parity reuse vLLM's logits-processor stack (TODO at the call site)."""
    t = max(float(temperature), 1e-5)
    logits = logits / t
    if top_p < 1.0:
        s, idx = torch.sort(logits, descending=True, dim=-1)
        cdf = torch.cumsum(torch.softmax(s, dim=-1), dim=-1)
        cut = cdf > top_p
        cut[..., 1:] = cut[..., :-1].clone()  # keep the first token crossing top_p
        cut[..., 0] = False
        s = s.masked_fill(cut, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, s)
    return logits


def select_vfd(
    p_unsafe: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    args_fallback: bool = False,
    prob_weight: float = 1.0,
    logp: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row value-filtered candidate selection (matches llm_safety phase-2).

    p_unsafe   [R, K]  predicted unsafety of each candidate (after consuming it)
    thresholds [R]     per-row safety threshold c
    logp       [R, K]  candidate LM log-probs (required iff args_fallback)
    returns    [R]     winner column index per row (long)

    Rule: if any candidate is safe (p_unsafe < c) pick the FIRST safe one (lowest
    column -> preserves the base distribution among safe tokens); else argmin(p_unsafe),
    or with args_fallback argmin(p_unsafe - prob_weight * exp(logp))."""
    R, K = p_unsafe.shape
    safe = p_unsafe < thresholds.view(R, 1)                       # [R, K]
    cols = torch.arange(K, device=p_unsafe.device).view(1, K).expand(R, K)
    sentinel = torch.full_like(cols, K)
    first_safe = torch.where(safe, cols, sentinel).min(dim=1).values  # [R], == K if none
    has_safe = first_safe < K
    if args_fallback:
        if logp is None:
            raise ValueError("args_fallback=True requires logp")
        fallback = (p_unsafe + prob_weight * (-torch.exp(logp))).argmin(dim=1)
    else:
        fallback = p_unsafe.argmin(dim=1)
    return torch.where(has_safe, first_safe, fallback)            # [R] long


def force_eos_rows(
    logits: torch.Tensor, abstain_mask: torch.Tensor, eos_token_id: int
) -> torch.Tensor:
    """In-place: for each row where abstain_mask is True, set the whole row to -inf
    and EOS to 0.0 so the sampler emits EOS. Returns the forced row indices.

    Callers must have already excluded rows that would suppress EOS (ignore_eos /
    unmet min_tokens), else the row becomes all -inf -> NaN after the EOS mask."""
    rows = torch.nonzero(abstain_mask, as_tuple=False).squeeze(-1)
    if rows.numel():
        logits[rows] = float("-inf")
        logits[rows, eos_token_id] = 0.0
    return rows
