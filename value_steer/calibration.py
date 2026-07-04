# SPDX-License-Identifier: Apache-2.0
"""
Conformal threshold calibration for the value-steering plugin.

Ports the statistical core of `calibrate_thresh` (llm_safety/calibration_utils.py),
decoupled from the model/data plumbing. Given a CALIBRATION SET of held-out examples --
each a binary safety label and the value head's per-step probability trajectory -- these
pick the decode threshold with a finite-sample guarantee on FALSE INTERVENTIONS (a safe
generation is intervened on with prob <= tau). This is the value-filtered-decoding /
safety-line calibration; the dynamic-abstention paper calibrates its threshold
empirically (quantile + isotonic) and carries no such conformal bound. The guarantee
makes the threshold principled rather than hand-tuned.

Pure numpy: no torch, no vLLM. Trajectories are produced upstream by running the
trained value head over held-out generations (the data step, which needs a GPU);
these functions are the threshold step and are unit-testable on CPU.

Guarantee (posterior variant), at level tau:
    P_H0( max_t p_t >= thr ) <= tau
on truly-safe sequences -- a safe generation triggers intervention with prob <= tau.

Direction: the returned threshold is for "intervene/reject when the per-step score
>= thr". VFD's runner compares `p_unsafe >= vfd_threshold`, which matches directly.
Abstention's direction depends on its probe's label semantics (see that runner's
direction note); calibrate on the same score the runner compares.
"""

from __future__ import annotations

import numpy as np


def conformal_plus_one_threshold(scores, tau: float = 0.05) -> float:
    """Core conformal (+1) quantile. Given per-safe-example scores S_i and target
    type-I level tau, return thr with finite-sample P(S >= thr) <= tau:
        k = ceil((n + 1) * (1 - tau));  thr = k-th smallest score (k clamped to [1, n]).
    When tau < 1/(n+1) the ideal k exceeds n; we clamp to the max observed score
    (the paper's pragmatic choice -- the bound then holds only up to sampling error,
    so calibrate with enough safe examples that n >= 1/tau - 1)."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    if n == 0:
        raise ValueError("no safe-example scores to calibrate on")
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    k = int(np.ceil((n + 1) * (1.0 - tau)))
    k = max(1, min(n, k))
    return float(s[k - 1])


def posterior_threshold(safe_labels, trajectories, tau: float = 0.05, eps: float = 1e-12) -> float:
    """Calibrate the per-step intervention threshold for VFD (and any posterior-based
    filter). Score per SAFE example (label 0) is S_i = max_t p_t over its trajectory;
    threshold is the conformal +1 quantile of those scores. Intervene when p_t >= thr.
    Mirrors calibrate_posterior_threshold."""
    y = _labels_1d(safe_labels, trajectories)
    safe_idx = np.where(y == 0)[0]
    if safe_idx.size == 0:
        raise ValueError("no safe examples (label 0) to calibrate on")
    scores = np.array([_traj_max(trajectories[i], eps) for i in safe_idx], dtype=np.float64)
    return conformal_plus_one_threshold(scores, tau)


def martingale_threshold(
    safe_labels, priors, trajectories, tau: float = 0.05, eps: float = 1e-12
) -> float:
    """Calibrate the threshold on max_t M_t for the time-to-unsafe-sampling martingale
        M_t = [p_t / (1 - p_t)] * [(1 - p0) / p0],
    with prior p0 = P(H1 | X) and posterior p_t = P(H1 | X, Y_{1:t}). Reject/abstain
    when M_t >= thr. Computed in log space for stability. Mirrors
    calibrate_martingale_threshold."""
    y = _labels_1d(safe_labels, trajectories)
    p0 = np.asarray(priors, dtype=np.float64)
    if p0.ndim != 1 or len(p0) != len(y):
        raise ValueError("priors must be 1D and match safe_labels length")
    safe_idx = np.where(y == 0)[0]
    if safe_idx.size == 0:
        raise ValueError("no safe examples (label 0) to calibrate on")
    scores = np.empty(safe_idx.size, dtype=np.float64)
    for j, i in enumerate(safe_idx):
        posts = np.asarray(trajectories[i], dtype=np.float64)
        if posts.size == 0:
            scores[j] = 0.0
            continue
        posts = _clip01(posts, eps)
        p0i = float(_clip01(p0[i], eps))
        log_m = (np.log(posts) - np.log1p(-posts)) + (np.log1p(-p0i) - np.log(p0i))
        scores[j] = float(np.exp(np.max(log_m)))
    return conformal_plus_one_threshold(scores, tau)


# --------------------------------------------------------------------------- #
def _labels_1d(safe_labels, trajectories) -> np.ndarray:
    y = np.asarray(safe_labels, dtype=int)
    if y.ndim != 1:
        raise ValueError("safe_labels must be 1D")
    if len(y) != len(trajectories):
        raise ValueError("safe_labels must match len(trajectories)")
    return y


def _clip01(p, eps):
    return np.clip(p, eps, 1.0 - eps)


def _traj_max(traj, eps) -> float:
    t = np.asarray(traj, dtype=np.float64)
    if t.size == 0:
        return 0.0          # a safe example that generated nothing never crosses
    return float(np.max(_clip01(t, eps)))
