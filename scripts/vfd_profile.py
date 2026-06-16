# SPDX-License-Identifier: Apache-2.0
"""Per-phase profiling of VFD's steady decode step (VFD_PROFILE path in vfd_model_runner).

Measures where the per-step time actually goes -- copy / forward / score / commit / seed --
so further optimization (e.g. cudagraph-capturing the per-layer KV loops) targets the phase
that dominates instead of an estimate. Runs with cudagraphs ON (the realistic regime now that
the PIECEWISE compile path is wired). opt-125m is deliberate: a small backbone makes the
ORCHESTRATION the largest fraction, so this is the UPPER BOUND on what capturing the eager
phases can buy -- if a phase is negligible here it is negligible on a 7B too.

    VFD_PROFILE=1 VALUE_STEER_TEST_MODEL=facebook/opt-125m python scripts/vfd_profile.py
"""
from __future__ import annotations
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ["VFD_PROFILE"] = "1"
os.environ.setdefault("VFD_PROFILE_EVERY", "64")

MODEL = os.environ.get("VALUE_STEER_TEST_MODEL", "facebook/opt-125m")
K = int(os.environ.get("VFD_K", "4"))
N = int(os.environ.get("SPEED_TOKENS", "256"))
UTIL = float(os.environ.get("VALUE_STEER_UTIL", "0.45"))   # 0.10 OOMs a 7B (weights alone ~14GB)


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        worker_cls="value_steer.worker.ValueSteerWorker",
        additional_config={"vfd": {"enabled": True, "threshold": 2.0,
                                   "num_candidates": K, "strict": True}},
        async_scheduling=False,          # cudagraphs ON (no enforce_eager)
        gpu_memory_utilization=UTIL,
        max_num_seqs=8,
        max_model_len=max(2048, N + 64),
    )
    prompt = "Write a detailed explanation of how photosynthesis works in plants."
    print(f"\n# vfd-profile model={MODEL} K={K} tokens={N} (cudagraphs ON)\n", flush=True)
    # One long generation: the [VFD-PROFILE] lines are printed every VFD_PROFILE_EVERY
    # steady steps by the runner (mean per-phase ms + % of summed phase time).
    llm.generate([prompt], SamplingParams(max_tokens=N, temperature=0.0, ignore_eos=True))
    print("\n[OK] vfd-profile done", flush=True)


if __name__ == "__main__":
    main()
