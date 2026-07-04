# SPDX-License-Identifier: Apache-2.0
"""value_steer: inference-time value steering for vLLM.

Two decode-time interventions sharing one value probe and feature contract:
  * dynamic abstention (Paper 1) -- gate to EOS when the value is low
  * value-filtered decoding (Paper 2) -- filter K candidates by a safety value

This top-level package imports only the vLLM-FREE modules (the value head, the pure
decode ops, calibration), so `import value_steer` works without vLLM installed --
useful for training/calibration boxes. The runners and the --worker-cls entry point
import vLLM and live in their own modules; import them explicitly where needed:
    from value_steer.worker import ValueSteerWorker
"""

__version__ = "0.1.1"

from . import calibration, steering_ops, verifiers
from .value_probe import DEFAULT_SPEC, FeatureSpec, ValueHead, load_value_head, request_threshold

__all__ = [
    "ValueHead", "FeatureSpec", "DEFAULT_SPEC", "load_value_head", "request_threshold",
    "steering_ops", "calibration", "verifiers", "__version__",
]
