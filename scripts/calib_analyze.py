# SPDX-License-Identifier: Apache-2.0
"""
Investigate why the conformal threshold overshoots (calibrated 0.83, but the head steers at 0.3).

Given a labeled decode-feature cache + a head, report:
  * threshold vs tau (posterior_threshold) -- which tau yields ~0.3;
  * the value distribution: max_t p and mean_t p over the response, for SAFE (label 0) vs UNSAFE,
    so we can see if safe examples spike high (-> high conformal threshold);
  * coverage: fraction of unsafe vs safe examples whose max-p crosses 0.3 / 0.5 / the calibrated thr.
"""

from __future__ import annotations

import os
import sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import argparse
import statistics as st

import torch

from value_steer.train_probe import FeatureCacheDataset
from value_steer.value_probe import load_value_head
from value_steer import calibration


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, help="labeled decode cache (a split dir with feats/index/meta)")
    ap.add_argument("--head", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = FeatureCacheDataset(args.cache_dir)
    head = load_value_head(args.head, ds.H, device)
    trajs, labels = [], []
    for i in range(len(ds)):
        it = ds[i]
        p = head.p(it["features"].to(device))[int(it["prompt_len"]):].float()
        trajs.append(p.tolist())
        labels.append(int(it["label"] >= 0.5))
    n_safe = labels.count(0); n_unsafe = labels.count(1)
    print(f"# {len(ds)} examples ({n_unsafe} unsafe / {n_safe} safe)", flush=True)

    print("\n## threshold vs tau (posterior_threshold over SAFE examples)", flush=True)
    for tau in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        try:
            thr = calibration.posterior_threshold(labels, trajs, tau=tau)
            print(f"   tau={tau:<5} -> threshold={thr:.4f}", flush=True)
        except Exception as e:
            print(f"   tau={tau}: {e}", flush=True)

    def stats(xs):
        xs = sorted(xs)
        q = lambda f: xs[min(len(xs) - 1, int(len(xs) * f))]
        return f"mean={st.mean(xs):.3f} p50={q(0.5):.3f} p90={q(0.9):.3f} max={xs[-1]:.3f}"
    smax = [max(t) if t else 0.0 for t, l in zip(trajs, labels) if l == 0]
    umax = [max(t) if t else 0.0 for t, l in zip(trajs, labels) if l == 1]
    smean = [st.mean(t) if t else 0.0 for t, l in zip(trajs, labels) if l == 0]
    umean = [st.mean(t) if t else 0.0 for t, l in zip(trajs, labels) if l == 1]
    print("\n## value distribution (response tokens)", flush=True)
    print(f"   SAFE   max-p: {stats(smax)}", flush=True)
    print(f"   UNSAFE max-p: {stats(umax)}", flush=True)
    print(f"   SAFE   mean-p: {stats(smean)}", flush=True)
    print(f"   UNSAFE mean-p: {stats(umean)}", flush=True)

    print("\n## coverage: fraction with max-p >= thr", flush=True)
    for thr in (0.3, 0.5, 0.83):
        sc = sum(m >= thr for m in smax) / max(1, len(smax))
        uc = sum(m >= thr for m in umax) / max(1, len(umax))
        print(f"   thr={thr}: SAFE-crossing={sc:.2f}  UNSAFE-crossing={uc:.2f}", flush=True)
    print("\n(If SAFE max-p is high (e.g. p90~0.8), conformal must set a high threshold to keep "
          "false interventions <= tau -> overshoot. A lower tau-target or smoother head fixes it.)",
          flush=True)


if __name__ == "__main__":
    main()
