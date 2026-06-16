# SPDX-License-Identifier: Apache-2.0
"""
Per-domain prompt/problem sources for value-head data generation.

Each domain yields a uniform stream of {"prompt": str, "meta": dict}. `meta` carries
whatever the domain's verifier needs: {} for safety (the judge is reference-free),
{"answer": ...} for math (ground truth), tests for code, etc. Adding a domain = one
branch here plus a Verifier (see value_steer.verifiers).

Light deps only (stdlib json). The math branch is a documented extension point that
lands together with the math verifier.
"""

from __future__ import annotations

import json


def _load_jsonl_prompts(path, n):
    """Dedup `prompt` strings from a jsonl file, up to n (None = all)."""
    seen, out = set(), []
    with open(path) as fh:
        for line in fh:
            p = json.loads(line).get("prompt", "").strip()
            if p and p not in seen:
                seen.add(p)
                out.append({"prompt": p, "meta": {}})
            if n is not None and len(out) >= n:
                break
    return out


def load_prompts(domain: str, source: str, n: int | None = None) -> list[dict]:
    """Return [{"prompt", "meta"}] for `domain`, reading from `source`.

    safety: `source` is a jsonl with a "prompt" field (the harmful-HH set); meta={}.
    math:   not implemented -- lands with the math verifier; expected to read a
            {problem, answer} source (e.g. GSM8K) and yield meta={"answer": ...}.
    """
    if domain == "safety":
        return _load_jsonl_prompts(source, n)
    if domain == "math":
        raise NotImplementedError(
            "math prompt source not implemented -- add a loader that yields "
            "{'prompt': problem, 'meta': {'answer': gold}} (e.g. from GSM8K) alongside the "
            "math verifier in value_steer.verifiers"
        )
    raise ValueError(f"unknown domain {domain!r} (known: safety, math)")
