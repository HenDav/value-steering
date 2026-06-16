# SPDX-License-Identifier: Apache-2.0
"""CPU tests for value_steer.versions (registry query/record/derive). No vLLM needed.

Run:  pytest tests/test_versions.py -q
"""

import json

import pytest

from value_steer import versions as V


def _write(path, data):
    path.write_text(json.dumps(data))
    return str(path)


# --------------------------------------------------------------------------- #
# query                                                                       #
# --------------------------------------------------------------------------- #
def test_packaged_registry_has_pin():
    # the shipped registry validates the version we built against
    assert "0.19.1" in V.validated_versions()
    assert V.is_validated("0.19.1")
    assert not V.is_validated("9.9.9")


def test_only_pass_results_count(tmp_path):
    p = _write(tmp_path / "r.json", {
        "0.19.1": {"result": "pass"},
        "0.20.0": {"result": "fail"},
    })
    assert V.validated_versions(p) == {"0.19.1"}
    assert V.is_validated("0.19.1", p)
    assert not V.is_validated("0.20.0", p)


# --------------------------------------------------------------------------- #
# record                                                                      #
# --------------------------------------------------------------------------- #
def test_record_roundtrip(tmp_path):
    p = str(tmp_path / "r.json")
    V.record_validation("0.19.2", "pass", static=True, behavioral=True, note="agent", path=p)
    assert V.is_validated("0.19.2", p)
    rec = json.loads((tmp_path / "r.json").read_text())["0.19.2"]
    assert rec["result"] == "pass" and rec["behavioral"] is True and "recorded" in rec


def test_record_fail_is_not_validated(tmp_path):
    p = str(tmp_path / "r.json")
    V.record_validation("0.21.0", "fail", static=False, path=p)
    assert not V.is_validated("0.21.0", p)


def test_record_bad_result_raises(tmp_path):
    with pytest.raises(ValueError):
        V.record_validation("0.19.2", "maybe", path=str(tmp_path / "r.json"))


# --------------------------------------------------------------------------- #
# derived specifier                                                           #
# --------------------------------------------------------------------------- #
def test_specifier_caps_at_next_minor(tmp_path):
    p = _write(tmp_path / "r.json", {
        "0.19.1": {"result": "pass"},
        "0.19.3": {"result": "pass"},
    })
    assert V.supported_specifier(p) == ">=0.19.1,<0.20"


def test_specifier_advances_with_new_minor(tmp_path):
    p = _write(tmp_path / "r.json", {
        "0.19.1": {"result": "pass"},
        "0.20.0": {"result": "pass"},
    })
    assert V.supported_specifier(p) == ">=0.19.1,<0.21"


def test_specifier_empty_when_none_validated(tmp_path):
    p = _write(tmp_path / "r.json", {"0.20.0": {"result": "fail"}})
    assert V.supported_specifier(p) == ""


# --------------------------------------------------------------------------- #
# runtime warning                                                             #
# --------------------------------------------------------------------------- #
def test_warn_on_unvalidated(tmp_path):
    p = _write(tmp_path / "r.json", {"0.19.1": {"result": "pass"}})
    with pytest.warns(RuntimeWarning):
        assert V.warn_if_unvalidated("9.9.9", path=p) is False
    assert V.warn_if_unvalidated("0.19.1", path=p) is True
