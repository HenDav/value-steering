# SPDX-License-Identifier: Apache-2.0
"""
Validated vLLM-version registry: the source of truth for which versions value-steer
has been checked against.

The compatibility agent OWNS this file. Its loop: run `compat_check` (static, and on a
GPU the behavioral pass) against a vLLM version, then `record_validation(version, ...)`
on green. The pin in pyproject.toml is a coarse pip-level gate derived from this set
(supported_specifier); the EXACT-version check here is the truth -- so even an in-range
but untested version triggers a runtime warning. Validation leads; the pin trails.

No vLLM import at module top, so this is importable on a training/calibration box.
"""

from __future__ import annotations

import datetime
import json
import warnings
from pathlib import Path

_DEFAULT_PATH = Path(__file__).with_name("validated_versions.json")


def _load(path=None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict, path=None) -> None:
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def validated_versions(path=None) -> set[str]:
    """Versions whose recorded result is 'pass'."""
    return {v for v, rec in _load(path).items() if rec.get("result") == "pass"}


def is_validated(version: str, path=None) -> bool:
    return version in validated_versions(path)


def record_validation(
    version: str,
    result: str = "pass",
    *,
    static: bool | None = None,
    behavioral: bool | None = None,
    note: str = "",
    path=None,
) -> dict:
    """Agent helper: record a version's validation result (pass/fail) and write it back.
    Writes to the packaged registry (or `path`); use against a source/editable checkout."""
    if result not in ("pass", "fail"):
        raise ValueError("result must be 'pass' or 'fail'")
    data = _load(path)
    rec = {
        "result": result,
        "static": static,
        "behavioral": behavioral,
        "note": note,
        "recorded": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    data[version] = rec
    _save(data, path)
    return rec


def current_vllm_version() -> str | None:
    try:
        import vllm
        return getattr(vllm, "__version__", None)
    except Exception:
        return None


def warn_if_unvalidated(version: str | None = None, path=None) -> bool:
    """Return True if `version` (default: the installed vLLM) is validated; else emit a
    RuntimeWarning and return False. Called at serve time by the worker."""
    v = version or current_vllm_version()
    if v is None:
        return False
    if is_validated(v, path):
        return True
    warnings.warn(
        f"vllm {v} is not in value-steer's validated set "
        f"({sorted(validated_versions(path))}). Run `value-steer-compat` to check "
        f"contracts before relying on steering; proceeding at your own risk.",
        RuntimeWarning,
        stacklevel=2,
    )
    return False


def supported_specifier(path=None) -> str:
    """Derive a coarse PEP 440 specifier from the validated set for pyproject:
    '>=<min>,<<major>.<minor+1>' over the passing versions. This is a convenience
    summary -- it may span untested intermediate versions, which is why the runtime
    warning checks the EXACT version. Regenerate the pyproject pin from this on release."""
    parsed = [t for t in (_parse(v) for v in validated_versions(path)) if len(t) >= 2]
    if not parsed:
        return ""
    lo, hi = min(parsed), max(parsed)
    lo_s = ".".join(str(x) for x in lo)
    return f">={lo_s},<{hi[0]}.{hi[1] + 1}"


def _parse(v: str) -> tuple[int, ...]:
    core = v.split("+")[0].split("rc")[0]
    out = []
    for part in core.split("."):
        if part.isdigit():
            out.append(int(part))
        else:
            break
    return tuple(out)
