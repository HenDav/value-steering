# SPDX-License-Identifier: Apache-2.0
"""
CPU unit tests for calibration.py (pure numpy; no torch, no vLLM).

Run:  pytest test_calibration.py -q
"""

import numpy as np
import pytest

from value_steer.calibration import (
    conformal_plus_one_threshold,
    martingale_threshold,
    posterior_threshold,
)


# --------------------------------------------------------------------------- #
# conformal_plus_one_threshold                                                #
# --------------------------------------------------------------------------- #
def test_conformal_quantile_deterministic():
    scores = np.arange(1, 11) / 10.0          # 0.1 .. 1.0, n = 10
    # tau=0.2 -> k = ceil(11*0.8) = 9 -> 9th smallest = 0.9
    assert conformal_plus_one_threshold(scores, 0.2) == pytest.approx(0.9)
    # tau=0.5 -> k = ceil(11*0.5) = 6 -> 6th smallest = 0.6
    assert conformal_plus_one_threshold(scores, 0.5) == pytest.approx(0.6)


def test_conformal_small_tau_clamps_to_max():
    scores = np.array([0.2, 0.4, 0.6])        # n=3; tau<1/4 forces k>n -> clamp to max
    assert conformal_plus_one_threshold(scores, 0.01) == pytest.approx(0.6)


def test_conformal_lower_tau_gives_higher_threshold():
    rng = np.random.default_rng(0)
    s = rng.random(200)
    assert conformal_plus_one_threshold(s, 0.01) >= conformal_plus_one_threshold(s, 0.2)


def test_conformal_empirical_coverage():
    # On i.i.d. safe scores, P(future safe score >= thr) should be ~<= tau.
    rng = np.random.default_rng(1)
    tau = 0.1
    exceed = []
    for _ in range(300):
        calib = rng.random(100)
        thr = conformal_plus_one_threshold(calib, tau)
        test = rng.random(2000)
        exceed.append((test >= thr).mean())
    # mean exceedance should sit at/under tau (conformal is slightly conservative)
    assert np.mean(exceed) <= tau + 0.02


def test_conformal_bad_tau_raises():
    with pytest.raises(ValueError):
        conformal_plus_one_threshold([0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        conformal_plus_one_threshold([], 0.1)


# --------------------------------------------------------------------------- #
# posterior_threshold                                                         #
# --------------------------------------------------------------------------- #
def test_posterior_uses_safe_maxes():
    labels = [0, 1, 0]
    trajs = [[0.2, 0.4], [0.9], [0.1, 0.3, 0.5]]   # safe maxes: 0.4, 0.5
    # n_safe=2, tau=0.5 -> k = ceil(3*0.5)=2 -> 2nd smallest of [0.4,0.5] = 0.5
    assert posterior_threshold(labels, trajs, 0.5) == pytest.approx(0.5)


def test_posterior_ignores_unsafe_examples():
    labels = [0, 1, 0]
    base = [[0.2, 0.4], [0.9], [0.1, 0.5]]
    changed = [[0.2, 0.4], [0.999, 0.999], [0.1, 0.5]]   # only the unsafe traj changed
    assert posterior_threshold(labels, base, 0.3) == posterior_threshold(labels, changed, 0.3)


def test_posterior_empty_safe_trajectory_scores_zero():
    labels = [0, 0]
    trajs = [[], [0.7]]                              # maxes: 0.0, 0.7
    # n=2, tau=0.5 -> k=2 -> 2nd smallest of [0.0,0.7] = 0.7
    assert posterior_threshold(labels, trajs, 0.5) == pytest.approx(0.7)


def test_posterior_no_safe_raises():
    with pytest.raises(ValueError):
        posterior_threshold([1, 1], [[0.5], [0.6]], 0.1)


# --------------------------------------------------------------------------- #
# martingale_threshold                                                        #
# --------------------------------------------------------------------------- #
def test_martingale_hand_computed():
    # one safe example, prior 0.5 -> M_t = p_t/(1-p_t). max over [0.5,0.9]:
    # 0.9/0.1 = 9.0. n=1, tau=0.5 -> k=1 -> threshold = 9.0
    thr = martingale_threshold([0], [0.5], [[0.5, 0.9]], 0.5)
    assert thr == pytest.approx(9.0, rel=1e-6)


def test_martingale_clips_extremes_no_inf():
    # p_t == 1.0 would blow up without clipping; must stay finite.
    thr = martingale_threshold([0], [0.5], [[1.0]], 0.5)
    assert np.isfinite(thr) and thr > 0


def test_martingale_length_mismatch_raises():
    with pytest.raises(ValueError):
        martingale_threshold([0, 0], [0.5], [[0.5], [0.6]], 0.1)   # priors too short
