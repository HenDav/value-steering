# SPDX-License-Identifier: Apache-2.0
"""
Mutation test: prove the GPU behavioral suite actually CATCHES the two VFD prefix bugs.

Reverts each fix in turn (in a backup-protected copy of vfd_model_runner.py), runs the
strict regression test (test_vfd_candidate_forward_shapes -- VFD greedy == base greedy
token-for-token), and confirms it FAILS, then PASSES once restored. If a mutated run still
passes, the test has no teeth for that bug.

    VALUE_STEER_TEST_MODEL=facebook/opt-125m python scripts/mutation_test.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("value_steer/vfd_model_runner.py")
BAK = Path("/tmp/vfd_runner_backup.py")
TEST = "tests/test_gpu_behavioral.py::test_vfd_candidate_forward_shapes"

MUTATIONS = {
    # Reintroduce the bootstrap position bug: anchor generation at 0 instead of the prompt
    # length (the original symptom -- first token's KV overwrote the prompt).
    "position_anchor_to_0": lambda s: s.replace(
        "self._next_pos[r] = int(self.input_batch.num_prompt_tokens[idx])",
        "self._next_pos[r] = 0", 1),
    # Reintroduce the slot_mapping bug: drop the per-layer slot_mapping from
    # set_forward_context, so candidate KV is written to a stale slot.
    "drop_slot_mapping": lambda s: re.sub(
        r",\s*slot_mapping=slot_mapping_by_layer", "", s, count=1),
}


def run_test() -> tuple[int, str]:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-m", "gpu", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    return r.returncode, (lines[-1] if lines else "<no output>")


def main() -> int:
    assert SRC.exists(), f"run from repo root; {SRC} not found"
    shutil.copy(SRC, BAK)
    verdicts = {}
    try:
        rc, last = run_test()
        print(f"[baseline      ] rc={rc} expect=PASS  ::  {last}", flush=True)
        verdicts["baseline_pass"] = rc == 0

        for name, mutate in MUTATIONS.items():
            s = SRC.read_text()
            s2 = mutate(s)
            assert s2 != s, f"{name}: mutation target not found (code changed?)"
            SRC.write_text(s2)
            try:
                rc, last = run_test()
            finally:
                shutil.copy(BAK, SRC)        # always restore
            print(f"[{name:14}] rc={rc} expect=FAIL  ::  {last}", flush=True)
            verdicts[f"{name}_caught"] = rc != 0

        rc, last = run_test()
        print(f"[restored      ] rc={rc} expect=PASS  ::  {last}", flush=True)
        verdicts["restored_pass"] = rc == 0
    finally:
        shutil.copy(BAK, SRC)                # paranoia: restore on any exit path

    print("\n=== MUTATION VERDICTS ===", flush=True)
    for k, v in verdicts.items():
        print(f"  {'OK ' if v else 'BAD'}  {k} = {v}", flush=True)
    all_ok = all(verdicts.values())
    print(f"\nMUTATION TEST {'PASSED -- the suite has teeth' if all_ok else 'FAILED'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
