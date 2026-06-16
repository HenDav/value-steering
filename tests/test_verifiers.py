# SPDX-License-Identifier: Apache-2.0
"""
CPU unit tests for the pluggable verifier interface (pure; stdlib only, no vLLM/transformers).

Run:  pytest test_verifiers.py -q

The math/code verifiers are located stubs in this pass; these tests pin that the registry
selects them and the stub fires (rather than silently mislabeling), plus the Verifier
score_batch default. Math equivalence tests land with the math implementation.
"""

import pytest

from value_steer.verifiers import (
    CodeVerifier,
    MathVerifier,
    Verifier,
    _BaseVerifier,
    get_verifier,
    register,
)


def test_math_verifier_is_a_located_stub():
    v = get_verifier("math")
    assert v.name == "math"
    with pytest.raises(NotImplementedError):
        v.score("p", "g", {"answer": "42"})


def test_code_verifier_is_a_located_stub():
    v = get_verifier("code")
    assert v.name == "code"
    with pytest.raises(NotImplementedError):
        v.score("p", "g")


def test_unknown_verifier_raises_valueerror():
    with pytest.raises(ValueError):
        get_verifier("bogus")


def test_score_batch_default_loops_over_score():
    class _Const(_BaseVerifier):
        name = "const"

        def score(self, prompt, generation, meta=None):
            return 1.0 if generation == "bad" else 0.0

    v = _Const()
    out = v.score_batch(["a", "b", "c"], ["ok", "bad", "ok"])
    assert out == [0.0, 1.0, 0.0]
    # metas default to None and are passed through positionally
    assert v.score_batch(["a"], ["bad"], [{"x": 1}]) == [1.0]


def test_registry_roundtrip_and_protocol():
    class _Dummy(_BaseVerifier):
        name = "dummy"

        def score(self, prompt, generation, meta=None):
            return 0.0

    register("dummy_test", _Dummy)
    v = get_verifier("dummy_test")
    assert isinstance(v, Verifier)          # runtime_checkable Protocol
    assert v.score("x", "y") == 0.0


def test_math_and_code_classes_register_under_names():
    assert isinstance(get_verifier("math"), MathVerifier)
    assert isinstance(get_verifier("code"), CodeVerifier)
