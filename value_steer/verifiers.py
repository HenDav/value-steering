# SPDX-License-Identifier: Apache-2.0
"""
Pluggable verifiers for value-head training data (domain-agnostic).

A *verifier* labels a (prompt, generation) pair with **P(undesirable) in [0, 1]**
-- 0.0 = good (safe / correct / passing), 1.0 = bad. This single convention is what
makes the value head domain-agnostic: the loss, feature, head, training loop, and
calibration are unchanged across domains; only the verifier differs. And because the
head predicts P(undesirable), VFD's selection (commit the candidate with the LOWEST
value -- "first-safe: p < threshold", else argmin) needs no per-domain edits.

This module is PURE (stdlib only -- no vLLM, no transformers, no sympy), so it imports
on a training/calibration box without the heavy stack and stays in `import value_steer`.
Verifiers that need heavy deps (the safety LLM judge, a math grader, a code sandbox)
register themselves from `scripts/` via `register(...)`; the math/code stubs here mark
the extension points. To add a domain: implement a Verifier, `register("<name>", factory)`,
and add a prompt source in `scripts/dataset_loaders.py`.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Verifier(Protocol):
    """Maps (prompt, generation) -> P(undesirable) in [0, 1] (0=good, 1=bad)."""

    name: str

    def score(self, prompt: str, generation: str, meta: dict | None = None) -> float:
        ...

    def score_batch(
        self, prompts: list[str], generations: list[str], metas: list[dict] | None = None
    ) -> list[float]:
        ...


class _BaseVerifier:
    """Mixin: a default `score_batch` that loops over `score` (override for real batching)."""

    name: str = "base"

    def score_batch(self, prompts, generations, metas=None):
        metas = metas if metas is not None else [None] * len(prompts)
        return [self.score(p, g, m) for p, g, m in zip(prompts, generations, metas)]


class MathVerifier(_BaseVerifier):
    """Math/olympiad correctness. STUB: not implemented in this pass.

    When implemented, WRAP AN EXISTING GRADING LIBRARY (e.g. HuggingFace `math_verify`,
    or an lm-eval grader) rather than hand-rolling answer extraction + symbolic
    equivalence. Ground truth arrives in `meta["answer"]`; return 0.0 if the
    generation's final answer matches, else 1.0."""

    name = "math"

    def score(self, prompt: str, generation: str, meta: dict | None = None) -> float:
        raise NotImplementedError(
            "math verifier not implemented -- wrap an existing grading library "
            "(e.g. math_verify) rather than hand-rolling; meta['answer'] holds ground truth"
        )


class CodeVerifier(_BaseVerifier):
    """Code correctness. STUB: not implemented (needs an isolated execution sandbox)."""

    name = "code"

    def score(self, prompt: str, generation: str, meta: dict | None = None) -> float:
        raise NotImplementedError(
            "code verifier needs a sandbox to execute untrusted model code against unit "
            "tests -- not built"
        )


# --------------------------------------------------------------------------- #
# Registry. Pure verifiers self-register here; impure ones (safety judge) call #
# register(...) from scripts/ on import, keeping this module dep-light.        #
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[..., Verifier]] = {}


def register(name: str, factory: Callable[..., Verifier]) -> None:
    """Register a verifier factory under `name` (idempotent overwrite)."""
    _REGISTRY[name] = factory


def get_verifier(name: str, **kwargs) -> Verifier:
    """Construct the verifier registered under `name`. `safety` is registered by
    importing `scripts/value_verifiers.py` first (it needs transformers)."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        hint = ""
        if name == "safety":
            hint = " -- import scripts/value_verifiers.py first to register the safety judge"
        raise ValueError(f"unknown verifier {name!r}; registered: {known}{hint}")
    return _REGISTRY[name](**kwargs)


register("math", MathVerifier)
register("code", CodeVerifier)
